"""
# Supervised Fine-Tuning of Llama Models with PDF Documents

This script demonstrates the complete process of fine-tuning a Llama model using custom PDF documents.
The process includes:
1. Extracting text from PDF files
2. Preprocessing the text data
3. Creating training data in the appropriate format
4. Fine-tuning a Llama model using the processed data
5. Evaluating and saving the fine-tuned model

"""

import os
import re
import json
import random
import argparse
import torch
import numpy as np
from typing import List, Dict, Union, Optional
from tqdm import tqdm
from pathlib import Path

# PDF processing
import pypdf

# Hugging Face libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    PeftModel
)

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a Llama model with PDF documents")
    
    # PDF and data processing arguments
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data and fine-tuned model")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for training")
    parser.add_argument("--train_val_split", type=float, default=0.9, help="Proportion of data for training (rest for validation)")
    
    # Model and training arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Hugging Face model name/path")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Additional arguments
    parser.add_argument("--format", type=str, default="instruction", choices=["instruction", "completion"], 
                        help="Training format: instruction-following or completion")
    
    return parser.parse_args()

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    print(f"Processing PDF: {pdf_path}")
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        text = []
        
        # Extract text from each page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text.append(page.extract_text())
    
    return "\n".join(text)

def process_pdfs(pdf_dir: str) -> List[str]:
    """
    Process all PDFs in a directory and extract text.
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        List of extracted text documents
    """
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Extract text from all PDFs
    documents = []
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            text = extract_text_from_pdf(str(pdf_file))
            if text.strip():  # Only add non-empty documents
                documents.append(text)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    
    print(f"Successfully processed {len(documents)} documents")
    return documents

def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw text extracted from PDF
    
    Returns:
        Cleaned text
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers (simple heuristic)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    return text.strip()

def chunk_document(document: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split a document into chunks of specified maximum length with overlap.
    
    Args:
        document: Document text
        max_length: Maximum chunk length (in characters)
        overlap: Overlap between chunks (in characters)
        
    Returns:
        List of document chunks
    """
    chunks = []
    start = 0
    doc_length = len(document)
    
    while start < doc_length:
        end = min(start + max_length, doc_length)
        
        # If not at the beginning of the document and not at the end,
        # try to find a good breaking point (period, newline)
        if start > 0 and end < doc_length:
            # Look for a good breaking point (period followed by space or newline)
            break_point = document.rfind('. ', start, end)
            if break_point == -1:
                break_point = document.rfind('\n', start, end)
            
            if break_point > start:
                end = break_point + 1  # Include the period
        
        chunks.append(document[start:end].strip())
        start = end - overlap  # Move start position with overlap
        
        # Avoid getting stuck in an infinite loop
        if start >= doc_length or end == doc_length:
            break
    
    return chunks

def prepare_instruction_dataset(documents: List[str], task_description: str) -> List[Dict]:
    """
    Prepare an instruction-following dataset from documents.
    
    Args:
        documents: List of document texts
        task_description: Description of the task for instruction
        
    Returns:
        List of instruction-response pairs
    """
    instruction_data = []
    
    # Create instruction-response pairs
    for doc in documents:
        # Split document into chunks to create multiple training examples
        chunks = chunk_document(doc)
        
        for chunk in chunks:
            # Create a sample instruction-response pair
            # This is a simplified approach - in a real scenario, you'd want to create
            # meaningful instruction-response pairs based on your specific use case
            instruction = f"{task_description}\n\nText: {chunk[:100]}..."
            response = chunk
            
            instruction_data.append({
                "instruction": instruction,
                "response": response
            })
    
    return instruction_data

def prepare_completion_dataset(documents: List[str], prefix_length: int = 100) -> List[Dict]:
    """
    Prepare a completion dataset from documents.
    
    Args:
        documents: List of document texts
        prefix_length: Length of prefix to use as prompt
        
    Returns:
        List of prompt-completion pairs
    """
    completion_data = []
    
    for doc in documents:
        chunks = chunk_document(doc)
        
        for chunk in chunks:
            if len(chunk) > prefix_length + 50:  # Ensure chunk is long enough
                prompt = chunk[:prefix_length]
                completion = chunk[prefix_length:]
                
                completion_data.append({
                    "prompt": prompt,
                    "completion": completion
                })
    
    return completion_data

def format_instruction_data(examples: Dict) -> Dict:
    """Format instruction data for training."""
    INSTRUCTION_FORMAT = """### Instruction:
{instruction}

### Response:
{response}"""

    texts = []
    for i in range(len(examples["instruction"])):
        text = INSTRUCTION_FORMAT.format(
            instruction=examples["instruction"][i],
            response=examples["response"][i]
        )
        texts.append(text)
    
    return {"text": texts}

def format_completion_data(examples: Dict) -> Dict:
    """Format completion data for training."""
    texts = []
    for i in range(len(examples["prompt"])):
        text = examples["prompt"][i] + examples["completion"][i]
        texts.append(text)
    
    return {"text": texts}

def tokenize_function(examples: Dict, tokenizer, max_length: int) -> Dict:
    """Tokenize the examples and prepare them for training."""
    # Tokenize and prepare the text
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    
    # Set labels same as input_ids for causal language modeling
    result["labels"] = result["input_ids"].copy()
    
    return result

def create_dataset(data: List[Dict], tokenizer, max_length: int, format_type: str = "instruction") -> Dataset:
    """Create a Hugging Face dataset from the prepared data."""
    # Convert to Hugging Face dataset
    dataset = Dataset.from_dict({k: [example[k] for example in data] for k in data[0].keys()})
    
    # Format the data based on the specified format
    if format_type == "instruction":
        dataset = dataset.map(format_instruction_data, batched=True)
    else:
        dataset = dataset.map(format_completion_data, batched=True)
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def load_model_and_tokenizer(args):
    """Load the model and tokenizer with appropriate configuration."""
    print(f"Loading model: {args.model_name}")
    
    # Configure quantization if requested
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        padding_side="right",
        trust_remote_code=True
    )
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Prepare model for training if using quantization
    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Wrap model with LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters information
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, eval_dataset, args):
    """Train the model using the prepared datasets."""
    print("Starting training...")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none",
    )
    
    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()
    
    # Save the model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    return trainer

def evaluate_model(trainer, model, tokenizer, eval_dataset, args):
    """Evaluate the fine-tuned model."""
    print("Evaluating the model...")
    
    # Run evaluation
    eval_results = trainer.evaluate()
    
    print(f"Evaluation results: {eval_results}")
    
    # Save evaluation results
    with open(f"{args.output_dir}/eval_results.json", "w") as f:
        json.dump(eval_results, f)
    
    # Sample generation
    print("Generating sample outputs...")
    sample_texts = []
    
    # Get a few samples from the evaluation dataset
    for i in range(min(3, len(eval_dataset))):
        input_ids = eval_dataset[i]["input_ids"]
        # Take the first half as prompt
        prompt_length = len(input_ids) // 2
        prompt = input_ids[:prompt_length]
        
        # Generate text
        inputs = torch.tensor([prompt]).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sample_texts.append(generated_text)
    
    # Save sample outputs
    with open(f"{args.output_dir}/sample_outputs.txt", "w") as f:
        for i, text in enumerate(sample_texts):
            f.write(f"Sample {i+1}:\n")
            f.write(text)
            f.write("\n\n" + "-"*50 + "\n\n")
    
    return eval_results

def main():
    """Main function to execute the fine-tuning pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process PDF documents
    print("Processing PDF documents...")
    raw_documents = process_pdfs(args.pdf_dir)
    
    # Clean and preprocess the documents
    print("Cleaning and preprocessing documents...")
    cleaned_documents = [clean_text(doc) for doc in raw_documents]
    
    # Prepare dataset based on the specified format
    print(f"Preparing {args.format} dataset...")
    if args.format == "instruction":
        task_description = "Please provide information based on the given text."
        data = prepare_instruction_dataset(cleaned_documents, task_description)
    else:
        data = prepare_completion_dataset(cleaned_documents)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Split data into train and validation sets
    random.shuffle(data)
    split_idx = int(len(data) * args.train_val_split)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(eval_data)}")
    
    # Create datasets
    train_dataset = create_dataset(train_data, tokenizer, args.max_length, args.format)
    eval_dataset = create_dataset(eval_data, tokenizer, args.max_length, args.format)
    
    # Save the datasets
    train_dataset.save_to_disk(f"{args.output_dir}/train_dataset")
    eval_dataset.save_to_disk(f"{args.output_dir}/eval_dataset")
    
    # Train the model
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset, args)
    
    # Evaluate the model
    evaluate_model(trainer, model, tokenizer, eval_dataset, args)
    
    print(f"Fine-tuning complete! Model saved to {args.output_dir}")
    print("You can load the fine-tuned model with:")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_dir}')")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_dir}')")
    
    # If using PEFT/LoRA
    print("For PEFT/LoRA model:")
    print(f"  base_model = AutoModelForCausalLM.from_pretrained('{args.model_name}')")
    print(f"  model = PeftModel.from_pretrained(base_model, '{args.output_dir}')")

if __name__ == "__main__":
    main()