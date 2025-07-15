import openai
import json
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the client for local LLM
client = openai.OpenAI(
    api_key="not-needed",  # Many local APIs don't require real API keys
    base_url="http://localhost:1234/v1"  # Common local LLM server URL
)

# Define multiple tools/functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '2 + 2' or 'sqrt(16)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (optional), e.g. 'UTC', 'US/Eastern'"
                    }
                },
                "required": []
            }
        }
    }
]

# Function implementations
def get_weather(location, unit="fahrenheit"):
    """Get weather for a location (mock implementation)"""
    return f"The weather in {location} is 72°{unit[0].upper()}, partly cloudy with light winds"

def calculate_math(expression):
    """Safely evaluate mathematical expressions"""
    try:
        # Simple safe evaluation - in production, use a proper math parser
        import math
        import re
        
        # Only allow safe mathematical operations
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "pow": pow,
            "max": max,
            "min": min
        }
        
        # Basic safety check - only allow numbers, operators, and safe functions
        if re.match(r'^[0-9+\-*/().sqrt sincotan log pi e abs round pow max min\s]+

# Metrics tracking
class MetricsTracker:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "tool_calls_made": 0,
            "total_tokens": 0,
            "total_time": 0,
            "requests": []
        }
    
    def log_request(self, request_type, response_time, tokens_used, tool_calls=0):
        self.metrics["total_requests"] += 1
        self.metrics["tool_calls_made"] += tool_calls
        self.metrics["total_tokens"] += tokens_used if tokens_used else 0
        self.metrics["total_time"] += response_time
        
        request_data = {
            "timestamp": datetime.now().isoformat(),
            "type": request_type,
            "response_time": response_time,
            "tokens_used": tokens_used,
            "tool_calls": tool_calls
        }
        self.metrics["requests"].append(request_data)
        
        logger.info(f"Request: {request_type} | Time: {response_time:.2f}s | Tokens: {tokens_used} | Tool calls: {tool_calls}")
    
    def print_summary(self):
        print("\n=== METRICS SUMMARY ===")
        print(f"Total requests: {self.metrics['total_requests']}")
        print(f"Tool calls made: {self.metrics['tool_calls_made']}")
        print(f"Total tokens used: {self.metrics['total_tokens']}")
        print(f"Total time: {self.metrics['total_time']:.2f}s")
        print(f"Average response time: {self.metrics['total_time'] / max(1, self.metrics['total_requests']):.2f}s")
        print("=======================\n")

# Initialize metrics tracker
tracker = MetricsTracker()

def main():
    """Main function to run the tool calling example"""
    
    # Get user input instead of hardcoded question
    user_question = input("Ask me anything (weather, math, time, or general questions): ")
    
    # Make the API call with metrics tracking
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use whatever model your local LLM serves
            messages=[
                {"role": "user", "content": user_question}
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        response_time = time.time() - start_time
        tokens_used = getattr(response, 'usage', None)
        tokens_count = tokens_used.total_tokens if tokens_used else None
        
        # Log initial request
        tracker.log_request("initial_request", response_time, tokens_count)
        
    except Exception as e:
        logger.error(f"Error in initial request: {e}")
        return

    # Check if the model wants to call a function
    message = response.choices[0].message

    if message.tool_calls:
        # The model wants to call a function
        tool_call_count = len(message.tool_calls)
        logger.info(f"Model requested {tool_call_count} tool call(s)")
        
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            logger.info(f"Calling function: {function_name} with args: {function_args}")
            
            # Call the appropriate function using dispatcher
            function_start = time.time()
            result = call_function(function_name, function_args)
            function_time = time.time() - function_start
            
            logger.info(f"Function executed in {function_time:.2f}s: {result}")
            
            # Send the function result back to the model
            messages = [
                {"role": "user", "content": user_question},
                message,  # The assistant's message with tool call
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.id
                }
            ]
            
            # Get the final response with metrics
            final_start = time.time()
            try:
                final_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                
                final_time = time.time() - final_start
                final_tokens = getattr(final_response, 'usage', None)
                final_tokens_count = final_tokens.total_tokens if final_tokens else None
                
                # Log final request
                tracker.log_request("final_request", final_time, final_tokens_count, tool_call_count)
                
                print("\nFinal Response:")
                print(final_response.choices[0].message.content)
                
            except Exception as e:
                logger.error(f"Error in final request: {e}")
                
    else:
        # No tool call, just print the regular response
        print("\nDirect Response (no tools used):")
        print(message.content)

    # Print metrics summary
    tracker.print_summary()


if __name__ == "__main__":
    main(), expression):
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"The result of {expression} is {result}"
        else:
            return "Invalid mathematical expression"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

def get_current_time(timezone="UTC"):
    """Get current date and time"""
    from datetime import datetime
    import pytz
    
    try:
        if timezone == "UTC":
            dt = datetime.utcnow()
            return f"Current UTC time: {dt.strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            # For simplicity, just return UTC time with timezone note
            dt = datetime.utcnow()
            return f"Current time (UTC): {dt.strftime('%Y-%m-%d %H:%M:%S')} (Timezone conversion not implemented)"
    except Exception as e:
        return f"Error getting time: {str(e)}"

# Function dispatcher
def call_function(function_name, function_args):
    """Call the appropriate function based on name"""
    if function_name == "get_weather":
        return get_weather(**function_args)
    elif function_name == "calculate_math":
        return calculate_math(**function_args)
    elif function_name == "get_current_time":
        return get_current_time(**function_args)
    else:
        return f"Unknown function: {function_name}"

# Metrics tracking
class MetricsTracker:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "tool_calls_made": 0,
            "total_tokens": 0,
            "total_time": 0,
            "requests": []
        }
    
    def log_request(self, request_type, response_time, tokens_used, tool_calls=0):
        self.metrics["total_requests"] += 1
        self.metrics["tool_calls_made"] += tool_calls
        self.metrics["total_tokens"] += tokens_used if tokens_used else 0
        self.metrics["total_time"] += response_time
        
        request_data = {
            "timestamp": datetime.now().isoformat(),
            "type": request_type,
            "response_time": response_time,
            "tokens_used": tokens_used,
            "tool_calls": tool_calls
        }
        self.metrics["requests"].append(request_data)
        
        logger.info(f"Request: {request_type} | Time: {response_time:.2f}s | Tokens: {tokens_used} | Tool calls: {tool_calls}")
    
    def print_summary(self):
        print("\n=== METRICS SUMMARY ===")
        print(f"Total requests: {self.metrics['total_requests']}")
        print(f"Tool calls made: {self.metrics['tool_calls_made']}")
        print(f"Total tokens used: {self.metrics['total_tokens']}")
        print(f"Total time: {self.metrics['total_time']:.2f}s")
        print(f"Average response time: {self.metrics['total_time'] / max(1, self.metrics['total_requests']):.2f}s")
        print("=======================\n")

# Initialize metrics tracker
tracker = MetricsTracker()

def main():
    """Main function to run the tool calling example"""
    
    # Make the API call with metrics tracking
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use whatever model your local LLM serves
            messages=[
                {"role": "user", "content": "What's the weather like in New York?"}
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        response_time = time.time() - start_time
        tokens_used = getattr(response, 'usage', None)
        tokens_count = tokens_used.total_tokens if tokens_used else None
        
        # Log initial request
        tracker.log_request("initial_request", response_time, tokens_count)
        
    except Exception as e:
        logger.error(f"Error in initial request: {e}")
        return

    # Check if the model wants to call a function
    message = response.choices[0].message

    if message.tool_calls:
        # The model wants to call a function
        tool_call_count = len(message.tool_calls)
        logger.info(f"Model requested {tool_call_count} tool call(s)")
        
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)  # Safer than eval
            
            logger.info(f"Calling function: {function_name} with args: {function_args}")
            
            # Call the actual function
            if function_name == "get_weather":
                function_start = time.time()
                result = get_weather(**function_args)
                function_time = time.time() - function_start
                
                logger.info(f"Function executed in {function_time:.2f}s: {result}")
                
                # Send the function result back to the model
                messages = [
                    {"role": "user", "content": "What's the weather like in New York?"},
                    message,  # The assistant's message with tool call
                    {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id
                    }
                ]
                
                # Get the final response with metrics
                final_start = time.time()
                try:
                    final_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages
                    )
                    
                    final_time = time.time() - final_start
                    final_tokens = getattr(final_response, 'usage', None)
                    final_tokens_count = final_tokens.total_tokens if final_tokens else None
                    
                    # Log final request
                    tracker.log_request("final_request", final_time, final_tokens_count, tool_call_count)
                    
                    print("Final Response:")
                    print(final_response.choices[0].message.content)
                    
                except Exception as e:
                    logger.error(f"Error in final request: {e}")
                    
    else:
        # No tool call, just print the regular response
        print("Direct Response:")
        print(message.content)

    # Print metrics summary
    tracker.print_summary()


if __name__ == "__main__":
    main()
