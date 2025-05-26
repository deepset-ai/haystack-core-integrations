#!/usr/bin/env python3
"""
Example usage of GoogleGenAIChatGenerator with the new Google Gen AI SDK.

This example demonstrates:
1. Basic chat generation
2. Conversation with context
3. System message usage
"""

import os
import sys
from typing import Dict, Any

from haystack.utils import Secret
from haystack.dataclasses.chat_message import ChatMessage
from haystack.dataclasses import StreamingChunk
from haystack.tools import Tool, Toolset
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator


def weather_function(city: str) -> Dict[str, Any]:
    """Get weather information for a city."""
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(city, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


def main():
    """Demonstrate GoogleGenAIChatGenerator usage."""
    
    # Check if API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("Please set the GOOGLE_API_KEY environment variable")
        return
    
    # Create weather tool
    weather_tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=weather_function,
    )
    
    # Option 1: Use List[Tool]
    # tools = [weather_tool]
    
    # Option 2: Use Toolset (demonstrates Toolset support)
    tools = Toolset([weather_tool])
    
    # Initialize the chat generator with tools
    print("🚀 Initializing GoogleGenAIChatGenerator with gemini-2.0-flash and weather tool...")
    chat_generator = GoogleGenAIChatGenerator(
        model="gemini-2.0-flash",
        api_key=Secret.from_env_var("GOOGLE_API_KEY"),
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 1000,
        },
        tools=tools  # Can be either List[Tool] or Toolset
    )
    
    # # Example 1: Basic chat
    # print("\n📝 Example 1: Basic chat")
    # messages = [ChatMessage.from_user("Explain quantum computing in simple terms")]
    # response = chat_generator.run(messages=messages)
    # print(f"User: {messages[0].text}")
    # print(f"Assistant: {response['replies'][0].text}")
    
    # # Example 2: Conversation with context
    # print("\n💬 Example 2: Conversation with context")
    # messages.append(response['replies'][0])
    # messages.append(ChatMessage.from_user("Can you give me a practical example?"))
    
    # response = chat_generator.run(messages=messages)
    # print(f"User: {messages[-1].text}")
    # print(f"Assistant: {response['replies'][0].text}")
    
    # # Example 3: Using system message
    # print("\n🎭 Example 3: Using system message")
    # system_messages = [
    #     ChatMessage.from_system("You are a helpful assistant that always responds in a poetic style."),
    #     ChatMessage.from_user("Tell me about the benefits of renewable energy")
    # ]
    
    # response = chat_generator.run(messages=system_messages)
    # print(f"System: {system_messages[0].text}")
    # print(f"User: {system_messages[1].text}")
    # print(f"Assistant: {response['replies'][0].text}")
    
    # Example 5: Tool calling
    print("\n🔧 Example 5: Tool calling")
    tool_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
    print(f"User: {tool_messages[0].text}")
    
    response = chat_generator.run(messages=tool_messages)
    assistant_message = response['replies'][0]
    print(f"Assistant: {assistant_message.text}")
    
    # Check if the model made tool calls
    if assistant_message.tool_calls:
        print(f"Tool calls made: {len(assistant_message.tool_calls)}")
        for tool_call in assistant_message.tool_calls:
            print(f"  - Tool: {tool_call.tool_name}")
            print(f"  - Arguments: {tool_call.arguments}")
            
            # Execute the tool call
            tool_result = weather_function(**tool_call.arguments)
            print(f"  - Result: {tool_result}")
            
            # Create a tool result message and continue the conversation
            tool_result_message = ChatMessage.from_tool(
                tool_result=str(tool_result),
                origin=tool_call
            )
            
            # Continue conversation with tool result
            follow_up_messages = tool_messages + [assistant_message, tool_result_message]
            final_response = chat_generator.run(messages=follow_up_messages)
            print(f"Final response: {final_response['replies'][0].text}")
    else:
        print("No tool calls were made.")

if __name__ == "__main__":
    main() 