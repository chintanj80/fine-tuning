import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import time

def run_inference(
    prompt,
    server_url="localhost:8000",
    model_name="vllm_model",
    max_tokens=512,
    temperature=0.7,
    top_p=0.9
):
    """
    Run inference using Triton's HTTP client to query a vLLM model.
    
    Args:
        prompt (str): The input prompt to send to the model
        server_url (str): URL of the Triton server
        model_name (str): Name of the model as configured in Triton
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature parameter for sampling
        top_p (float): Top-p parameter for nucleus sampling
        
    Returns:
        dict: Response containing generated text and metadata
    """
    # Create the Triton HTTP client
    client = httpclient.InferenceServerClient(
        url=server_url,
        verbose=False
    )
    
    # Check if server and model are ready
    if not client.is_server_ready():
        return {"error": f"Triton server at {server_url} is not ready", "success": False}
    
    if not client.is_model_ready(model_name):
        return {"error": f"Model {model_name} is not ready on the server", "success": False}
    
    # Create the inputs list
    inputs = []
    
    # Create input for the prompt
    prompt_data = np.array([prompt], dtype=object)
    input_prompt = httpclient.InferInput("prompt", prompt_data.shape, np_to_triton_dtype(prompt_data.dtype))
    input_prompt.set_data_from_numpy(prompt_data)
    inputs.append(input_prompt)
    
    # Create input for max_tokens
    max_tokens_data = np.array([max_tokens], dtype=np.int32)
    input_max_tokens = httpclient.InferInput("max_tokens", max_tokens_data.shape, np_to_triton_dtype(max_tokens_data.dtype))
    input_max_tokens.set_data_from_numpy(max_tokens_data)
    inputs.append(input_max_tokens)
    
    # Create input for temperature
    temperature_data = np.array([temperature], dtype=np.float32)
    input_temperature = httpclient.InferInput("temperature", temperature_data.shape, np_to_triton_dtype(temperature_data.dtype))
    input_temperature.set_data_from_numpy(temperature_data)
    inputs.append(input_temperature)
    
    # Create input for top_p
    top_p_data = np.array([top_p], dtype=np.float32)
    input_top_p = httpclient.InferInput("top_p", top_p_data.shape, np_to_triton_dtype(top_p_data.dtype))
    input_top_p.set_data_from_numpy(top_p_data)
    inputs.append(input_top_p)
    
    # Define the output
    outputs = [httpclient.InferRequestedOutput("generated_text")]
    
    # Measure inference time
    start_time = time.time()
    
    try:
        # Send the inference request
        response = client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            request_id="request1"
        )
        
        # Get the output
        output_data = response.as_numpy("generated_text")
        generated_text = output_data[0].decode('utf-8')
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        return {
            "generated_text": generated_text,
            "latency_ms": latency_ms,
            "success": True
        }
    
    except Exception as e:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        return {
            "error": str(e),
            "latency_ms": latency_ms,
            "success": False
        }

def run_streaming_inference(
    prompt,
    server_url="localhost:8000",
    model_name="vllm_model",
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    print_output=True
):
    """
    Run streaming inference using Triton's HTTP client to query a vLLM model.
    
    Args:
        prompt (str): The input prompt to send to the model
        server_url (str): URL of the Triton server
        model_name (str): Name of the model as configured in Triton
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature parameter for sampling
        top_p (float): Top-p parameter for nucleus sampling
        print_output (bool): Whether to print tokens as they arrive
        
    Returns:
        str: The full generated text
    """
    # Create the Triton HTTP client
    client = httpclient.InferenceServerClient(
        url=server_url,
        verbose=False
    )
    
    # Check if server and model are ready
    if not client.is_server_ready() or not client.is_model_ready(model_name):
        print(f"Error: Server or model not ready")
        return ""
    
    # Create the inputs list
    inputs = []
    
    # Create input for the prompt
    prompt_data = np.array([prompt], dtype=object)
    input_prompt = httpclient.InferInput("prompt", prompt_data.shape, np_to_triton_dtype(prompt_data.dtype))
    input_prompt.set_data_from_numpy(prompt_data)
    inputs.append(input_prompt)
    
    # Create input for max_tokens
    max_tokens_data = np.array([max_tokens], dtype=np.int32)
    input_max_tokens = httpclient.InferInput("max_tokens", max_tokens_data.shape, np_to_triton_dtype(max_tokens_data.dtype))
    input_max_tokens.set_data_from_numpy(max_tokens_data)
    inputs.append(input_max_tokens)
    
    # Create input for temperature
    temperature_data = np.array([temperature], dtype=np.float32)
    input_temperature = httpclient.InferInput("temperature", temperature_data.shape, np_to_triton_dtype(temperature_data.dtype))
    input_temperature.set_data_from_numpy(temperature_data)
    inputs.append(input_temperature)
    
    # Create input for top_p
    top_p_data = np.array([top_p], dtype=np.float32)
    input_top_p = httpclient.InferInput("top_p", top_p_data.shape, np_to_triton_dtype(top_p_data.dtype))
    input_top_p.set_data_from_numpy(top_p_data)
    inputs.append(input_top_p)
    
    # Enable streaming mode
    stream_data = np.array([True], dtype=bool)
    input_stream = httpclient.InferInput("stream", stream_data.shape, np_to_triton_dtype(stream_data.dtype))
    input_stream.set_data_from_numpy(stream_data)
    inputs.append(input_stream)
    
    full_output = ""
    
    try:
        # Send the streaming inference request
        response_stream = client.stream_infer(
            model_name=model_name,
            inputs=inputs,
            request_id="stream_request"
        )
        
        # Process streaming responses
        for response in response_stream:
            if response.get_response() is not None:
                result = response.get_response()
                
                try:
                    # Extract token from response
                    token = result.outputs[0].data[0].decode('utf-8')
                    full_output += token
                    
                    if print_output:
                        print(token, end="", flush=True)
                        
                except Exception as e:
                    print(f"\nError processing token: {e}")
        
        if print_output:
            print("\n")
            
        return full_output
    
    except Exception as e:
        print(f"Streaming inference failed: {e}")
        return full_output


if __name__ == "__main__":
    # Example usage
    prompt = "Explain how neural networks work:"
    
    print("=== Standard Inference ===")
    result = run_inference(
        prompt=prompt,
        server_url="localhost:8000",  # Change to your Triton server address
        model_name="vllm_model",      # Change to your model name
        max_tokens=256,
        temperature=0.7,
        top_p=0.9
    )
    
    if result["success"]:
        print(f"Generated text:\n{result['generated_text']}")
        print(f"Latency: {result['latency_ms']:.2f}ms")
    else:
        print(f"Inference failed: {result['error']}")
    
    print("\n=== Streaming Inference ===")
    print("Generated text:")
    full_text = run_streaming_inference(
        prompt="Write a short story about AI:",
        server_url="localhost:8000",  # Change to your Triton server address
        model_name="vllm_model",      # Change to your model name
        max_tokens=256,
        temperature=0.7,
        top_p=0.9
    )