#!/usr/bin/env python3

import os
import sys
import time

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from haystack.utils import Secret
from haystack.dataclasses.chat_message import ChatMessage
from haystack.dataclasses import StreamingChunk
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

def streaming_callback(chunk: StreamingChunk):
    """
    Callback function to handle streaming chunks.
    Prints each chunk as it arrives to simulate real-time streaming.
    """
    print(chunk.content, end='', flush=True)

def test_streaming_vs_non_streaming():
    """Test both streaming and non-streaming modes."""
    
    # Check if API key is available
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        return False
    
    try:
        # Initialize the chat generator
        print("🚀 Initializing GoogleGenAIChatGenerator...")
        chat_generator = GoogleGenAIChatGenerator(
            model="gemini-2.0-flash",
            api_key=Secret.from_env_var("GOOGLE_API_KEY")
        )
        print("✓ Chat generator initialized successfully\n")
        
        # Test 1: Non-streaming mode
        print("📝 Test 1: Non-streaming mode")
        print("=" * 50)
        messages = [ChatMessage.from_user("Write a short story about a robot learning to paint. Keep it under 200 words.")]
        
        start_time = time.time()
        response = chat_generator.run(messages=messages)
        end_time = time.time()
        
        print(f"Response: {response['replies'][0].text}")
        print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds\n")
        
        # Test 2: Streaming mode
        print("🌊 Test 2: Streaming mode")
        print("=" * 50)
        print("Streaming response (watch it appear in real-time):")
        print("-" * 50)
        
        start_time = time.time()
        response = chat_generator.run(
            messages=messages,
            streaming_callback=streaming_callback
        )
        end_time = time.time()
        
        print(f"\n-" * 50)
        print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
        print(f"✓ Final response length: {len(response['replies'][0].text)} characters\n")
        
        # Test 3: Streaming with conversation context
        print("💬 Test 3: Streaming with conversation context")
        print("=" * 50)
        
        # Add the previous response to conversation
        messages.append(response['replies'][0])
        messages.append(ChatMessage.from_user("Now make the story even shorter, just 2 sentences."))
        
        print("Streaming follow-up response:")
        print("-" * 50)
        
        response = chat_generator.run(
            messages=messages,
            streaming_callback=streaming_callback
        )
        
        print(f"\n-" * 50)
        print(f"✓ Follow-up response: {len(response['replies'][0].text)} characters")
        
        # Test 4: Streaming with system message
        print("\n🎭 Test 4: Streaming with system message")
        print("=" * 50)
        
        system_messages = [
            ChatMessage.from_system("You are a helpful assistant that responds in haiku format."),
            ChatMessage.from_user("Describe the beauty of coding")
        ]
        
        print("Streaming haiku response:")
        print("-" * 50)
        
        response = chat_generator.run(
            messages=system_messages,
            streaming_callback=streaming_callback
        )
        
        print(f"\n-" * 50)
        print("✓ Haiku response completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_streaming_callback_variations():
    """Test different streaming callback implementations."""
    
    print("\n🔧 Testing different streaming callback variations")
    print("=" * 60)
    
    # Check if API key is available
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        return False
    
    chat_generator = GoogleGenAIChatGenerator(
        model="gemini-2.0-flash",
        api_key=Secret.from_env_var("GOOGLE_API_KEY")
    )
    
    messages = [ChatMessage.from_user("Count from 1 to 10 with a brief description of each number.")]
    
    # Callback that collects chunks
    collected_chunks = []
    def collecting_callback(chunk: StreamingChunk):
        collected_chunks.append(chunk.content)
        print(f"[CHUNK {len(collected_chunks)}]: {chunk.content}", end='')
    
    print("Collecting chunks with numbered output:")
    print("-" * 50)
    
    response = chat_generator.run(
        messages=messages,
        streaming_callback=collecting_callback
    )
    
    print(f"\n-" * 50)
    print(f"✓ Collected {len(collected_chunks)} chunks")
    print(f"✓ Total response length: {len(response['replies'][0].text)} characters")
    print(f"✓ Chunks combined length: {len(''.join(collected_chunks))} characters")
    
    return True

if __name__ == "__main__":
    print("Testing GoogleGenAIChatGenerator Streaming Functionality")
    print("=" * 60)
    
    success1 = test_streaming_vs_non_streaming()
    success2 = test_streaming_callback_variations()
    
    if success1 and success2:
        print("\n🎉 All streaming tests passed! GoogleGenAIChatGenerator streaming is working correctly.")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 