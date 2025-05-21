# Using VLLM for LLaMA Model Inference

VLLM is a high-throughput and memory-efficient inference engine for LLMs like LLaMA. Let me walk you through how to set it up and use it for inference with detailed code examples.

## Basic Setup

First, you'll need to install the vllm package:

```python
pip install vllm
```

## Loading a LLaMA Model

Here's a comprehensive example of loading and using a LLaMA model with VLLM:

```python
from vllm import LLM, SamplingParams
import time

# Initialize the LLM
# Replace "meta-llama/Llama-2-7b-chat-hf" with your specific LLaMA model
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,  # Number of GPUs to use for tensor parallelism
    gpu_memory_utilization=0.85,  # Control memory usage (0-1)
    max_model_len=4096,  # Max sequence length to process
    quantization="awq",  # Optional: quantization method (e.g., "awq", "gptq", None)
    # seed=42,  # Optional: for reproducibility
    # trust_remote_code=False,  # Set to True if loading a model with custom code
    # download_dir="/path/to/models",  # Custom path for downloaded models
    # revision="main",  # Branch/tag/commit to use when downloading
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,  # Higher = more random, lower = more deterministic
    top_p=0.95,  # Nucleus sampling parameter
    top_k=50,  # Limit sampling to top k tokens
    max_tokens=512,  # Maximum number of tokens to generate
    stop=["</s>", "Human:", "\n\n"],  # Stop sequences
    # frequency_penalty=1.0,  # Penalize repeated tokens
    # presence_penalty=0.0,  # Penalize tokens already in the prompt
    # skip_special_tokens=True,  # Skip special tokens in output
    # logprobs=None,  # Number of top logprobs to return per token
    # logits_processors=[],  # Custom logits processors
)

# Prepare prompt
# Format may vary depending on your specific LLaMA version
prompt = """<s>[INST] Write a short poem about AI [/INST]"""

# Generate response
start_time = time.time()
outputs = llm.generate([prompt], sampling_params)
end_time = time.time()

# Process and display response
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text}")
    print(f"Generation time: {end_time - start_time:.2f} seconds")
    print(f"Token count: {len(output.outputs[0].token_ids)}")
```

## Using VLLM as a Server

You can also run VLLM as a server and make API calls to it:

```python
# Terminal command to start the server
# vllm --model meta-llama/Llama-2-7b-chat-hf --tensor-parallel-size 1 --host 0.0.0.0 --port 8000
```

Then, to query the server:

```python
import requests
import json

# API endpoint (local server or remote endpoint)
endpoint_url = "http://localhost:8000/generate"

# Request payload
payload = {
    "prompt": "<s>[INST] Write a short poem about AI [/INST]",
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "max_tokens": 512,
    "stop": ["</s>", "Human:", "\n\n"]
}

# Send request
response = requests.post(endpoint_url, json=payload)
result = json.loads(response.text)

# Print the generated text
print(result["text"])
```

## Advanced: Batch Processing for Multiple Prompts

For processing multiple prompts efficiently:

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# Prepare multiple prompts
prompts = [
    "<s>[INST] What is artificial intelligence? [/INST]",
    "<s>[INST] Explain quantum computing [/INST]",
    "<s>[INST] Write a haiku about programming [/INST]"
]

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.8, max_tokens=256)

# Generate responses in batch
outputs = llm.generate(prompts, sampling_params)

# Process outputs
for i, output in enumerate(outputs):
    print(f"Prompt {i+1}: {prompts[i]}")
    print(f"Response: {output.outputs[0].text}")
    print("-" * 50)
```

## Parameters Reference

Here's a comprehensive list of parameters you can use with VLLM:

### LLM Initialization Parameters:
- `model`: Model name or path
- `tokenizer`: Optional tokenizer name or path (defaults to model)
- `tokenizer_mode`: "auto", "slow" or "fast"
- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `dtype`: "auto", "half", "float16", "bfloat16", "float", "float32"
- `gpu_memory_utilization`: Target GPU memory usage (0-1)
- `max_model_len`: Maximum sequence length
- `quantization`: Quantization method ("awq", "gptq", "squeezellm", None)
- `revision`: Model revision/version tag
- `trust_remote_code`: Whether to trust custom code in model
- `download_dir`: Directory for downloading models
- `seed`: Random seed for reproducibility 
- `enforce_eager`: Force eager execution
- `max_context_len_to_capture`: Maximum context length to capture

### SamplingParams:
- `n`: Number of output sequences per prompt
- `best_of`: Number of candidates to generate per prompt
- `temperature`: Temperature for sampling
- `top_p`: Top-p sampling parameter
- `top_k`: Top-k sampling parameter
- `min_p`: Minimum probability for nucleus sampling
- `typical_p`: Parameter for typical sampling
- `frequency_penalty`: Frequency penalty for repeated tokens
- `presence_penalty`: Presence penalty for tokens in prompt
- `repetition_penalty`: Penalty for repeated tokens
- `max_tokens`: Maximum number of tokens to generate
- `stop`: Stop sequences to end generation
- `ignore_eos`: Whether to ignore EOS token
- `logprobs`: Number of token log probabilities to return
- `prompt_logprobs`: Whether to return prompt token log probabilities
- `skip_special_tokens`: Whether to skip special tokens in output

Would you like me to explain any specific part of this setup in more detail?