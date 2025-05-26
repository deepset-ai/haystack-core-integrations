#!/usr/bin/env python3

import os
import sys
from typing import Dict, Any

from haystack.utils import Secret
from haystack.dataclasses.chat_message import ChatMessage
from haystack.tools import Tool
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator


def weather_function(city: str) -> Dict[str, Any]:
    """Get weather information for a city."""
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
        "London": {"weather": "rainy", "temperature": 5, "unit": "celsius"},
        "Tokyo": {"weather": "clear", "temperature": 12, "unit": "celsius"},
    }
    return weather_info.get(city, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


def calculator_function(expression: str) -> Dict[str, Any]:
    """Calculate a mathematical expression."""
    try:
        # Simple evaluation for basic math expressions
        result = eval(expression)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}


def test_tool_calling():
    """Test various tool calling scenarios."""
    
    # Create tools
    weather_tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=weather_function,
    )
    
    calculator_tool = Tool(
        name="calculator",
        description="useful for performing mathematical calculations",
        parameters={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        function=calculator_function,
    )
    
    # Initialize chat generator with tools
    chat_generator = GoogleGenAIChatGenerator(
        model="gemini-2.0-flash",
        api_key=Secret.from_env_var("GOOGLE_API_KEY"),
        tools=[weather_tool, calculator_tool],
        generation_config={
            "temperature": 0.1,  # Low temperature for more deterministic tool calls
            "max_output_tokens": 1000,
        }
    )
    
    test_cases = [
        "What's the weather like in Tokyo?",
        "Calculate 15 * 23 + 7",
        "What's the weather in Berlin and what's 100 divided by 4?",
        "Tell me about the weather in London",
        "What's 2 + 2?",
    ]
    
    print("🧪 Testing Google Gen AI Tool Calling")
    print("=" * 50)
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {query}")
        print("-" * 30)
        
        try:
            # Initial request
            messages = [ChatMessage.from_user(query)]
            response = chat_generator.run(messages=messages)
            assistant_message = response['replies'][0]
            
            print(f"Assistant: {assistant_message.text}")
            
            # Handle tool calls
            if assistant_message.tool_calls:
                print(f"🔧 Tool calls made: {len(assistant_message.tool_calls)}")
                
                # Add assistant message to conversation
                messages.append(assistant_message)
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    print(f"  - Tool: {tool_call.tool_name}")
                    print(f"  - Arguments: {tool_call.arguments}")
                    
                    # Execute the appropriate tool
                    if tool_call.tool_name == "weather":
                        result = weather_function(**tool_call.arguments)
                    elif tool_call.tool_name == "calculator":
                        result = calculator_function(**tool_call.arguments)
                    else:
                        result = {"error": f"Unknown tool: {tool_call.tool_name}"}
                    
                    print(f"  - Result: {result}")
                    
                    # Create tool result message
                    tool_result_message = ChatMessage.from_tool(
                        tool_result=str(result),
                        origin=tool_call
                    )
                    messages.append(tool_result_message)
                
                # Get final response
                final_response = chat_generator.run(messages=messages)
                final_message = final_response['replies'][0]
                print(f"📋 Final response: {final_message.text}")
            else:
                print("ℹ️  No tool calls were made.")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Tool calling tests completed!")


if __name__ == "__main__":
    test_tool_calling() 