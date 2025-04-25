# Supervised Fine-Tuning of Llama Models with PDF Documents: Usage Guide

This guide explains how to use the provided code to fine-tune Llama models on your PDF documents.

## Setup

First, install the required dependencies:

```bash
pip install torch transformers datasets peft accelerate bitsandbytes PyPDF2 tqdm
```

## Hugging Face Access Setup

To use Llama models, you'll need to:

1. Create a Hugging Face account if you don't have one

2. Request access to the Meta Llama models at https://huggingface.co/meta-llama

3. Generate and set up a Hugging Face token:
   
   ```bash
   huggingface-cli login# Enter your token when prompted
   ```

## GPU Requirements

Fine-tuning Llama models requires significant computational resources:

- For 7B models: At least 16GB GPU VRAM (using 4-bit quantization)
- For 13B models: At least 24GB GPU VRAM (using 4-bit quantization)
- For larger models: Multiple GPUs or specialized hardware

## Preparing Your Data

1. Organize your PDF documents in a folder structure
2. Consider data quality and relevance to your fine-tuning objective
3. Make sure your PDFs are readable (not scanned images without OCR)

## Fine-Tuning Options

The script provides two training formats:

1. **Instruction format** (default): Creates instruction-response pairs from your documents
   
   - Good for: Creating assistants that follow instructions based on document content
   - Use with: `--format instruction`

2. **Completion format**: Creates prompt-completion pairs for text generation
   
   - Good for: Creating models that can generate text similar to your documents
   - Use with: `--format completion`

## Running the Fine-Tuning

### Basic Usage

```bash
python llama_finetuning.py \
  --pdf_dir /path/to/your/pdfs \
  --output_dir /path/to/save/model \
  --model_name meta-llama/Llama-2-7b-hf
```

### Advanced Usage

```bash
python llama_finetuning.py \
  --pdf_dir /path/to/your/pdfs \
  --output_dir /path/to/save/model \
  --model_name meta-llama/Llama-2-7b-hf \
  --load_in_4bit \
  --lora_r 16 \
  --lora_alpha 32 \
  --learning_rate 2e-4 \
  --batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_epochs 5 \
  --max_length 1024 \
  --format instruction
```

## Parameter Explanation

- `--pdf_dir`: Directory containing your PDF files
- `--output_dir`: Directory to save the fine-tuned model and data
- `--model_name`: Base model to fine-tune (e.g., `meta-llama/Llama-2-7b-hf`)
- `--load_in_4bit` or `--load_in_8bit`: Use 4-bit or 8-bit quantization to reduce memory usage
- `--lora_r`: LoRA rank parameter (higher = more capacity but more memory)
- `--lora_alpha`: LoRA alpha parameter (higher = stronger adaptation)
- `--learning_rate`: Learning rate for fine-tuning
- `--batch_size`: Batch size for training
- `--gradient_accumulation_steps`: Accumulate gradients over multiple batches
- `--num_epochs`: Number of training epochs
- `--max_length`: Maximum sequence length for training
- `--format`: Training format: `instruction` or `completion`

## Using the Fine-Tuned Model

After fine-tuning, you can use your model like this:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Option 1: Load directly if you saved the full model
tokenizer = AutoTokenizer.from_pretrained("/path/to/save/model")
model = AutoModelForCausalLM.from_pretrained("/path/to/save/model")

# Option 2: Load with PEFT/LoRA (more efficient)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto", 
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "/path/to/save/model")

# Generate text with your model
prompt = "Please provide information about..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Tips for Better Results

1. **Preprocess your PDFs**: Ensure they're text-based and not just scanned images
2. **Consider data quality**: Clean and relevant data yields better results
3. **Adjust LoRA parameters**:
   - Higher `lora_r` = more capacity but more memory usage
   - Higher `lora_alpha` = stronger adaptation
4. **Experiment with instruction formats**: Create clear, consistent instruction-response pairs
5. **Start with a few epochs**: Monitor validation loss and avoid overfitting
6. **Use quantization**: `--load_in_4bit` reduces memory requirements with minimal quality loss

## Troubleshooting

- **Memory errors**: Try reducing batch size, using quantization, or reducing model size
- **Poor results**: Check data quality, increase training time, adjust learning rate
- **PDF extraction issues**: Ensure PDFs contain actual text and not just images
- **Slow training**: Consider using a more powerful GPU or cloud services

## Further Customization

- Adjust the chunking strategy in `chunk_document()` to better suit your documents
- Modify the instruction templates in `prepare_instruction_dataset()` for your use case
- Add custom evaluation metrics in `evaluate_model()`
