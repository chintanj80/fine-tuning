import httpx
import json
import numpy as np
import time
from typing import Dict, List, Any, Optional


class TritonClient:
    """Client for communicating with Triton Inference Server using HTTPX."""
    
    def __init__(self, url: str = "http://localhost:8000", model_name: str = "qa_model", model_version: str = "1"):
        """
        Initialize the Triton client.
        
        Args:
            url: The URL of the Triton server.
            model_name: The name of the model to use.
            model_version: The version of the model to use.
        """
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self.client = httpx.Client(timeout=30.0)  # 30 second timeout
        
    def health_check(self) -> bool:
        """Check if the Triton server is ready."""
        try:
            response = self.client.get(f"{self.url}/v2/health/ready")
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def get_model_metadata(self) -> Dict:
        """Get metadata about the model."""
        try:
            response = self.client.get(f"{self.url}/v2/models/{self.model_name}/versions/{self.model_version}")
            return response.json()
        except Exception as e:
            print(f"Failed to get model metadata: {e}")
            return {}
    
    def infer(self, question: str, context: str) -> Dict[str, Any]:
        """
        Make an inference request to the Triton server for question answering.
        
        Args:
            question: The question to answer.
            context: The context from which to extract the answer.
            
        Returns:
            The server's response containing the answer.
        """
        # Prepare the inference inputs
        inputs = [
            {
                "name": "question",
                "datatype": "BYTES",
                "shape": [1],
                "data": [question]
            },
            {
                "name": "context",
                "datatype": "BYTES",
                "shape": [1],
                "data": [context]
            }
        ]
        
        # Create the request payload
        payload = {
            "inputs": inputs,
            "outputs": [
                {
                    "name": "answer",
                    "parameters": {
                        "binary_data": False
                    }
                },
                {
                    "name": "score",
                    "parameters": {
                        "binary_data": False
                    }
                }
            ]
        }
        
        try:
            # Make the HTTP POST request to the Triton server
            start_time = time.time()
            response = self.client.post(
                f"{self.url}/v2/models/{self.model_name}/versions/{self.model_version}/infer",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            inference_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the answer and score from the response
                answer = None
                score = None
                
                for output in result.get("outputs", []):
                    if output["name"] == "answer":
                        answer = output["data"][0]
                    elif output["name"] == "score":
                        score = float(output["data"][0])
                
                return {
                    "answer": answer,
                    "score": score,
                    "inference_time": inference_time,
                    "success": True
                }
            else:
                return {
                    "error": f"Inference failed with status code {response.status_code}",
                    "details": response.text,
                    "success": False
                }
                
        except Exception as e:
            return {
                "error": f"Exception during inference: {str(e)}",
                "success": False
            }
    
    def close(self):
        """Close the client connection."""
        self.client.close()


def main():
    """Example usage of the TritonClient for question answering."""
    
    # Initialize the client
    client = TritonClient(
        url="http://localhost:8000",
        model_name="qa_model",
        model_version="1"
    )
    
    # Check if the server is healthy
    if not client.health_check():
        print("Triton server is not ready. Please check the server status.")
        return
    
    # Get model metadata
    metadata = client.get_model_metadata()
    print(f"Model metadata: {json.dumps(metadata, indent=2)}")
    
    # Example question and context
    context = """
    Triton Inference Server provides a cloud and edge inferencing solution optimized for both CPUs and GPUs.
    It supports multiple frameworks including TensorRT, TensorFlow, PyTorch, ONNX Runtime, OpenVINO, Python,
    and custom C++ backends. Triton includes HTTP and gRPC endpoints that implement the KServe inference
    protocol, allowing remote clients to request inference for any model being managed by the server.
    """
    
    question = "What frameworks does Triton support?"
    
    # Make an inference request
    result = client.infer(question, context)
    
    if result["success"]:
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence score: {result['score']:.4f}")
        print(f"Inference time: {result['inference_time']:.4f} seconds")
    else:
        print(f"Error: {result.get('error')}")
        print(f"Details: {result.get('details', 'No additional details')}")
    
    # Close the client connection
    client.close()


if __name__ == "__main__":
    main()