# Creating an API with VLLM for LLaMA Model Inference

Let me walk you through creating a complete API service using VLLM for LLaMA model inference. I'll show you two approaches: using VLLM's built-in server and creating a custom FastAPI application.

## Option 1: VLLM's Built-in Server

VLLM provides a built-in OpenAI-compatible server that you can launch with a single command:

```python
# Install requirements
pip install vllm uvicorn fastapi

# Start the server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85
```

This creates an OpenAI-compatible API server that you can query:

```python
import openai

# Configure client
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # The API key is not actually checked
)

# Make a completion request
completion = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",  # This can be any string
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about AI."}
    ],
    temperature=0.7,
    max_tokens=512
)

# Print the response
print(completion.choices[0].message.content)
```

## Option 2: Custom FastAPI Application

For more control and customization, you can create your own FastAPI application:

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
from vllm import LLM, SamplingParams
import uvicorn
import json
import time
import asyncio
from contextlib import asynccontextmanager

# Initialize LLM in a context manager to ensure proper loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model before server starts
    app.state.llm = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        # Optional: quantization settings
        # quantization="awq",
    )
    yield
    # Clean up resources if needed when server shuts down
    # No specific cleanup needed for VLLM

# Create FastAPI app
app = FastAPI(lifespan=lifespan)

# Define request and response models
class GenerationRequest(BaseModel):
    prompt: str
    temperature: float = Field(0.7, ge=0, le=2.0)
    top_p: float = Field(0.95, ge=0, le=1.0)
    top_k: int = Field(50, ge=0)
    max_tokens: int = Field(512, ge=1, le=4096)
    stop: Optional[List[str]] = None
    stream: bool = False
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    
class GenerationResponse(BaseModel):
    text: str
    usage: Dict[str, int]
    finish_reason: str
    latency: float

# Define the API endpoints
@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    try:
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop if request.stop else None,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty
        )
        
        # Generate the response
        start_time = time.time()
        outputs = app.state.llm.generate([request.prompt], sampling_params)
        end_time = time.time()
        latency = end_time - start_time
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text
        input_token_count = len(outputs[0].prompt_token_ids)
        output_token_count = len(outputs[0].outputs[0].token_ids)
        
        # Prepare response
        response = {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_token_count,
                "completion_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count
            },
            "finish_reason": "stop" if outputs[0].outputs[0].finish_reason == "stop" else "length",
            "latency": latency
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "meta-llama/Llama-2-7b-chat-hf"}

# Run the server
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
```

Save this to a file named `api.py` and run it with `python api.py`.

## Option 3: Advanced FastAPI with Additional Features

For a more comprehensive API with streaming support, batch processing, and more:

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any, Generator
from vllm import LLM, SamplingParams
import uvicorn
import json
import time
import asyncio
from contextlib import asynccontextmanager
import logging
import os
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vllm-api")

# Initialize LLM in a context manager to ensure proper loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model before server starts
    logger.info(f"Loading model: {os.environ.get('MODEL_ID', 'meta-llama/Llama-2-7b-chat-hf')}")
    app.state.llm = LLM(
        model=os.environ.get("MODEL_ID", "meta-llama/Llama-2-7b-chat-hf"),
        tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")),
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "4096")),
        quantization=os.environ.get("QUANTIZATION", None),
        trust_remote_code=os.environ.get("TRUST_REMOTE_CODE", "False").lower() == "true"
    )
    app.state.request_lock = Lock()  # Add a lock for thread-safe operations
    logger.info("Model loaded successfully")
    yield
    # No specific cleanup needed for VLLM
    logger.info("Shutting down server")

# Create FastAPI app
app = FastAPI(
    title="VLLM LLaMA API",
    description="API for generating text with LLaMA models using VLLM",
    version="1.0.0",
    lifespan=lifespan
)

# Define request and response models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = Field(0.7, ge=0, le=2.0)
    top_p: float = Field(0.95, ge=0, le=1.0)
    top_k: int = Field(50, ge=0)
    max_tokens: int = Field(512, ge=1, le=4096)
    stop: Optional[List[str]] = None
    stream: bool = False
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)

class GenerationRequest(BaseModel):
    prompt: str
    temperature: float = Field(0.7, ge=0, le=2.0)
    top_p: float = Field(0.95, ge=0, le=1.0)
    top_k: int = Field(50, ge=0)
    max_tokens: int = Field(512, ge=1, le=4096)
    stop: Optional[List[str]] = None
    stream: bool = False
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)

class BatchGenerationRequest(BaseModel):
    prompts: List[str]
    temperature: float = Field(0.7, ge=0, le=2.0)
    top_p: float = Field(0.95, ge=0, le=1.0)
    top_k: int = Field(50, ge=0)
    max_tokens: int = Field(512, ge=1, le=4096)
    stop: Optional[List[str]] = None

class GenerationResponse(BaseModel):
    text: str
    usage: Dict[str, int]
    finish_reason: str
    latency: float

class BatchGenerationResponse(BaseModel):
    responses: List[GenerationResponse]
    total_latency: float

# Helper function to format chat messages into a prompt
def format_chat_messages(messages: List[Message]) -> str:
    """Convert chat messages to a prompt format suitable for LLaMA models."""
    formatted_prompt = ""
    
    for msg in messages:
        if msg.role == "system":
            # System message typically goes at the beginning for LLaMA-2-chat
            formatted_prompt = f"<s>[INST] <<SYS>>\n{msg.content}\n<</SYS>>\n\n"
        elif msg.role == "user":
            # If we already have content, close previous instruction and start new one
            if formatted_prompt and not formatted_prompt.endswith("[INST] "):
                formatted_prompt += " [/INST] "
                formatted_prompt += f"<s>[INST] {msg.content}"
            else:
                # First user message
                formatted_prompt += f"{msg.content}"
        elif msg.role == "assistant":
            # Assistant responses come after [/INST]
            formatted_prompt += f" [/INST] {msg.content} "
    
    # Make sure we close the final instruction if it's open
    if formatted_prompt and not formatted_prompt.endswith("[/INST] "):
        formatted_prompt += " [/INST] "
    
    return formatted_prompt

# Streaming response generator
def generate_stream_response(llm, prompt, sampling_params) -> Generator[str, None, None]:
    """Generate streaming response for text generation."""
    
    # For streaming, we must use the streaming API
    for output in llm.generate_stream([prompt], sampling_params):
        generated_text = ""
        for i, generated_text in enumerate(output):
            if i == 0:  # First response contains the prompt
                continue
            
            # Get the newly generated text
            generated_chunk = generated_text.outputs[0].text
            
            # Yield the chunk as a SSE event
            yield f"data: {json.dumps({'text': generated_chunk})}\n\n"
    
    # Signal the end of the stream
    yield f"data: {json.dumps({'finish': True})}\n\n"

# API endpoints
@app.post("/v1/chat/completions", response_model=GenerationResponse)
async def chat_completions(request: ChatRequest):
    """Generate text based on chat messages."""
    try:
        # Format the chat messages into a prompt
        prompt = format_chat_messages(request.messages)
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop if request.stop else None,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty
        )
        
        # For streaming responses
        if request.stream:
            return StreamingResponse(
                generate_stream_response(app.state.llm, prompt, sampling_params),
                media_type="text/event-stream"
            )
        
        # For regular responses
        with app.state.request_lock:
            start_time = time.time()
            outputs = app.state.llm.generate([prompt], sampling_params)
            end_time = time.time()
            latency = end_time - start_time
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text
        input_token_count = len(outputs[0].prompt_token_ids)
        output_token_count = len(outputs[0].outputs[0].token_ids)
        
        # Prepare response
        response = {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_token_count,
                "completion_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count
            },
            "finish_reason": "stop" if outputs[0].outputs[0].finish_reason == "stop" else "length",
            "latency": latency
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions", response_model=GenerationResponse)
async def completions(request: GenerationRequest):
    """Generate text based on a prompt."""
    try:
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop if request.stop else None,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty
        )
        
        # For streaming responses
        if request.stream:
            return StreamingResponse(
                generate_stream_response(app.state.llm, request.prompt, sampling_params),
                media_type="text/event-stream"
            )
        
        # For regular responses
        with app.state.request_lock:
            start_time = time.time()
            outputs = app.state.llm.generate([request.prompt], sampling_params)
            end_time = time.time()
            latency = end_time - start_time
        
        # Extract the generated text
        generated_text = outputs[0].outputs[0].text
        input_token_count = len(outputs[0].prompt_token_ids)
        output_token_count = len(outputs[0].outputs[0].token_ids)
        
        # Prepare response
        response = {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_token_count,
                "completion_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count
            },
            "finish_reason": "stop" if outputs[0].outputs[0].finish_reason == "stop" else "length",
            "latency": latency
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error in completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/batch_completions", response_model=BatchGenerationResponse)
async def batch_completions(request: BatchGenerationRequest):
    """Generate text for multiple prompts in a batch."""
    try:
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop if request.stop else None
        )
        
        # Generate responses for all prompts in batch
        with app.state.request_lock:
            start_time = time.time()
            outputs = app.state.llm.generate(request.prompts, sampling_params)
            end_time = time.time()
            total_latency = end_time - start_time
        
        # Process each response
        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            input_token_count = len(output.prompt_token_ids)
            output_token_count = len(output.outputs[0].token_ids)
            
            response = {
                "text": generated_text,
                "usage": {
                    "prompt_tokens": input_token_count,
                    "completion_tokens": output_token_count,
                    "total_tokens": input_token_count + output_token_count
                },
                "finish_reason": "stop" if output.outputs[0].finish_reason == "stop" else "length",
                "latency": 0  # Individual latency not available in batch mode
            }
            responses.append(response)
        
        return {
            "responses": responses,
            "total_latency": total_latency
        }
    
    except Exception as e:
        logger.error(f"Error in batch completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": os.environ.get("MODEL_ID", "meta-llama/Llama-2-7b-chat-hf"),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "VLLM LLaMA API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/v1/completions",
            "/v1/chat/completions",
            "/v1/batch_completions",
            "/health"
        ]
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "api:app", 
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
        workers=1  # VLLM already uses multiple workers internally
    )

## Deploying the API

For a production deployment, you'll want to containerize your API. Here's a Dockerfile example:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_ID="meta-llama/Llama-2-7b-chat-hf" \
    TENSOR_PARALLEL_SIZE=1 \
    GPU_MEMORY_UTILIZATION=0.85 \
    MAX_MODEL_LEN=4096

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .

# Command to run the API server
CMD ["python3", "api.py"]
```

Create a `requirements.txt` file:

```
fastapi>=0.95.0
uvicorn>=0.22.0
vllm>=0.2.0
pydantic>=2.0.0
```

## Making Requests to the API

Here are examples of how to interact with your API:

### Basic Text Generation
```python
import requests
import json

url = "http://localhost:8000/v1/completions"
payload = {
    "prompt": "<s>[INST] Write a short poem about artificial intelligence [/INST]",
    "max_tokens": 250,
    "temperature": 0.7
}

response = requests.post(url, json=payload)
result = response.json()
print(result["text"])
```

### Chat Completion
```python
import requests

url = "http://localhost:8000/v1/chat/completions"
payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about AI."}
    ],
    "temperature": 0.7,
    "max_tokens": 250
}

response = requests.post(url, json=payload)
result = response.json()
print(result["text"])
```

### Batch Processing
```python
import requests

url = "http://localhost:8000/v1/batch_completions"
payload = {
    "prompts": [
        "<s>[INST] What is artificial intelligence? [/INST]",
        "<s>[INST] Write a haiku about programming [/INST]"
    ],
    "temperature": 0.7,
    "max_tokens": 150
}

response = requests.post(url, json=payload)
result = response.json()
for i, resp in enumerate(result["responses"]):
    print(f"Response {i+1}: {resp['text']}")
```

### Streaming Example (JavaScript/Client-side)
```javascript
// Example usage in JavaScript with EventSource
const eventSource = new EventSource('/v1/completions?stream=true&prompt=Tell me a story');

eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.finish) {
    eventSource.close();
  } else {
    document.getElementById('output').innerHTML += data.text;
  }
};

eventSource.onerror = function() {
  eventSource.close();
};
```

## Key Features of the Advanced API

1. **Environment Variable Configuration**: Allows easy configuration through environment variables
2. **OpenAI-like Endpoints**: Compatible with OpenAI client libraries
3. **Streaming Support**: Server-sent events for streaming responses
4. **Batch Processing**: Handle multiple prompts i