#!/usr/bin/env python3

import os
import sys
from typing import Dict, Any

from haystack.utils import Secret
from haystack.dataclasses.chat_message import ChatMessage
from haystack.tools import Tool, Toolset
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


def test_toolset_support():
    """Test Toolset support in GoogleGenAIChatGenerator."""
    
    # Create individual tools
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
    
    # Create a Toolset
    toolset = Toolset([weather_tool, calculator_tool])
    
    print("🧪 Testing Google Gen AI Toolset Support")
    print("=" * 50)
    
    # Test 1: Initialize with Toolset
    print("\n📝 Test 1: Initialize with Toolset")
    print("-" * 30)
    
    try:
        chat_generator = GoogleGenAIChatGenerator(
            model="gemini-2.0-flash",
            api_key=Secret.from_env_var("GOOGLE_API_KEY"),
            tools=toolset,  # Using Toolset instead of List[Tool]
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1000,
            }
        )
        print("✅ Successfully initialized with Toolset")
        
        # Test with weather query
        messages = [ChatMessage.from_user("What's the weather like in Tokyo?")]
        response = chat_generator.run(messages=messages)
        assistant_message = response['replies'][0]
        
        print(f"User: {messages[0].text}")
        print(f"Assistant: {assistant_message.text}")
        
        if assistant_message.tool_calls:
            print(f"🔧 Tool calls made: {len(assistant_message.tool_calls)}")
            for tool_call in assistant_message.tool_calls:
                print(f"  - Tool: {tool_call.tool_name}")
                print(f"  - Arguments: {tool_call.arguments}")
                
                # Execute the tool
                if tool_call.tool_name == "weather":
                    result = weather_function(**tool_call.arguments)
                    print(f"  - Result: {result}")
                    
                    # Continue conversation
                    tool_result_message = ChatMessage.from_tool(
                        tool_result=str(result),
                        origin=tool_call
                    )
                    follow_up_messages = messages + [assistant_message, tool_result_message]
                    final_response = chat_generator.run(messages=follow_up_messages)
                    print(f"📋 Final response: {final_response['replies'][0].text}")
        else:
            print("ℹ️  No tool calls were made.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Pass Toolset to run method
    print("\n📝 Test 2: Pass Toolset to run method")
    print("-" * 30)
    
    try:
        # Initialize without tools
        chat_generator_no_tools = GoogleGenAIChatGenerator(
            model="gemini-2.0-flash",
            api_key=Secret.from_env_var("GOOGLE_API_KEY"),
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1000,
            }
        )
        
        # Pass toolset to run method
        messages = [ChatMessage.from_user("Calculate 25 * 4")]
        response = chat_generator_no_tools.run(messages=messages, tools=toolset)
        assistant_message = response['replies'][0]
        
        print(f"User: {messages[0].text}")
        print(f"Assistant: {assistant_message.text}")
        
        if assistant_message.tool_calls:
            print(f"🔧 Tool calls made: {len(assistant_message.tool_calls)}")
            for tool_call in assistant_message.tool_calls:
                print(f"  - Tool: {tool_call.tool_name}")
                print(f"  - Arguments: {tool_call.arguments}")
                
                # Execute the tool
                if tool_call.tool_name == "calculator":
                    result = calculator_function(**tool_call.arguments)
                    print(f"  - Result: {result}")
                    
                    # Continue conversation
                    tool_result_message = ChatMessage.from_tool(
                        tool_result=str(result),
                        origin=tool_call
                    )
                    follow_up_messages = messages + [assistant_message, tool_result_message]
                    final_response = chat_generator_no_tools.run(messages=follow_up_messages, tools=toolset)
                    print(f"📋 Final response: {final_response['replies'][0].text}")
        else:
            print("ℹ️  No tool calls were made.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Compare List[Tool] vs Toolset behavior
    print("\n📝 Test 3: Compare List[Tool] vs Toolset behavior")
    print("-" * 30)
    
    try:
        # Test with List[Tool]
        chat_gen_list = GoogleGenAIChatGenerator(
            model="gemini-2.0-flash",
            api_key=Secret.from_env_var("GOOGLE_API_KEY"),
            tools=[weather_tool, calculator_tool],  # List[Tool]
        )
        
        # Test with Toolset
        chat_gen_toolset = GoogleGenAIChatGenerator(
            model="gemini-2.0-flash",
            api_key=Secret.from_env_var("GOOGLE_API_KEY"),
            tools=toolset,  # Toolset
        )
        
        test_message = [ChatMessage.from_user("What's the weather in Berlin?")]
        
        # Both should work identically
        response_list = chat_gen_list.run(messages=test_message)
        response_toolset = chat_gen_toolset.run(messages=test_message)
        
        print("✅ Both List[Tool] and Toolset work correctly")
        print(f"List[Tool] tool calls: {len(response_list['replies'][0].tool_calls)}")
        print(f"Toolset tool calls: {len(response_toolset['replies'][0].tool_calls)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ Toolset support tests completed!")


if __name__ == "__main__":
    test_toolset_support() 