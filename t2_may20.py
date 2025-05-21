# Setting Up a Model Serving API with vLLM

vLLM is a high-performance, open-source library for LLM inference and serving that offers significant throughput improvements compared to other solutions. Here's a comprehensive guide to setting up a model serving API with vLLM:

## Installation

First, install vLLM:

```bash
pip install vllm
```

For GPU acceleration, you'll need CUDA-compatible hardware and the appropriate CUDA version installed.

## Basic API Server Deployment

The simplest way to start a vLLM API server is:

```bash
python -m vllm.entrypoints.api_server --model [MODEL_NAME_OR_PATH]
```

## Launch Parameters

vLLM offers extensive configuration options when launching your model server:

### Model Loading Parameters
- `--model`: Model name or path (required)
- `--tokenizer`: Tokenizer name or path (defaults to model name/path)
- `--revision`: Model revision/version to use
- `--tokenizer-revision`: Tokenizer revision to use
- `--tokenizer-mode`: Mode for tokenizer operations (`auto`, `slow`, or `fast`)
- `--trust-remote-code`: Boolean flag to trust remote code for model loading
- `--download-dir`: Directory for downloading models
- `--load-format`: Model loading format (`auto`, `pt`, `safetensors`, `npcache`, or `dummy`)
- `--dtype`: Data type for model weights (`auto`, `half`, `float16`, `bfloat16`, `float`, etc.)
- `--max-model-len`: Maximum sequence length for model context
- `--worker-use-ray`: Use Ray for worker management
- `--seed`: Random seed for reproducibility

### Quantization Parameters
- `--quantization`: Quantization method (`awq`, `gptq`, `squeezellm`, etc.)
- `--kv-cache-dtype`: KV cache data type
- `--enforce-eager`: Force PyTorch eager execution

### Serving Configuration
- `--host`: Host address to bind the server (default: `localhost`)
- `--port`: Port to run the server on (default: `8000`)
- `--api-key`: Optional API key for authentication
- `--served-model-name`: Name for the model in API responses
- `--engine-use-ray`: Use Ray for engine management
- `--disable-log-stats`: Disable throughput logging
- `--disable-log-requests`: Disable request logging

### Hardware Utilization
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--max-parallel-loading-workers`: Maximum workers for parallel loading
- `--block-size`: Token block size for continuous batching
- `--swap-space`: GPU VRAM allocation for swapping
- `--max-num-batched-tokens`: Maximum tokens to process in parallel
- `--max-num-seqs`: Maximum number of sequences to process in parallel
- `--max-paddings`: Maximum padding tokens in a batch
- `--disable-custom-all-reduce`: Disable custom all-reduce implementation

### GPU Memory Management
- `--gpu-memory-utilization`: Fraction of GPU memory to use (0.0-1.0)
- `--max-cpu-memory`: Maximum CPU memory to use for offloading
- `--num-gpu-blocks`: Number of GPU blocks to allocate
- `--num-cpu-blocks`: Number of CPU blocks to allocate

### Specific Optimizations
- `--disable-quant-padding`: Disable quantized padding
- `--enable-chunked-prefill`: Enable prefill request chunking
- `--max-prefill-chunk-size`: Maximum chunk size for prefill request
- `--enforce-typing`: Enforce dtype integrity for model inputs

## Example Configurations

### Basic OpenAI-compatible API Server

```bash
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000
```

### High-performance Server with Multi-GPU

```bash
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000
```

### Quantized Model for Memory Efficiency

```bash
python -m vllm.entrypoints.api_server \
    --model TheBloke/Llama-2-13B-AWQ \
    --quantization awq \
    --dtype half \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --host 0.0.0.0 \
    --port 8000
```

## Using the API

Once the server is running, you can send requests to it:

### OpenAI-compatible Endpoint

vLLM provides an OpenAI-compatible API at `/v1/completions` and `/v1/chat/completions`.

Example curl request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": "San Francisco is",
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.95
    }'
```

## Programmatic Usage

You can also use vLLM's Python API directly:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)
outputs = llm.generate("San Francisco is", sampling_params)
print(outputs[0].outputs[0].text)
```

Would you like me to go into more detail about any specific aspect of vLLM setup or configuration?