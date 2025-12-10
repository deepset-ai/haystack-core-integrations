#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
MistralAgentSDK Examples
========================

Comprehensive examples demonstrating all features of the MistralAgentSDK component.

Prerequisites:
    pip install mistralai haystack-ai

Environment Variables:
    MISTRAL_API_KEY - Your Mistral API key
    MISTRAL_AGENT_ID - Your agent ID from the Mistral console

To create an agent:
    1. Go to https://console.mistral.ai/
    2. Navigate to "Agents" section
    3. Create a new agent or use an existing one
    4. Copy the agent ID

Run all examples:
    export MISTRAL_API_KEY="your-api-key"
    export MISTRAL_AGENT_ID="your-agent-id"
    python mistral_agent_sdk_examples.py
"""

import asyncio
import json
import os
import sys
from typing import Optional

from haystack import Pipeline
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall
from haystack.tools import Tool, Toolset

# Import the MistralAgentSDK
# Adjust the import path based on your project structure
try:
    from haystack_integrations.components.agents.mistral.agent_sdk import MistralAgentSDK
except ImportError:
    print("Error: Could not import MistralAgentSDK")
    print("Make sure the integration is installed or the path is correct")
    sys.exit(1)


def get_agent_id() -> str:
    """Get agent ID from environment or raise error."""
    agent_id = os.environ.get("MISTRAL_AGENT_ID")
    if not agent_id:
        print("Error: MISTRAL_AGENT_ID environment variable not set")
        print("Create an agent at https://console.mistral.ai/ and set:")
        print("  export MISTRAL_AGENT_ID='your-agent-id'")
        sys.exit(1)
    return agent_id


# =============================================================================
# EXAMPLE 1: Basic Chat
# =============================================================================


def example_basic_chat():
    """
    Basic example: Simple question-answer with an agent.

    This demonstrates the simplest usage of MistralAgentSDK.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Chat")
    print("=" * 60)

    agent = MistralAgentSDK(agent_id=get_agent_id())

    messages = [
        ChatMessage.from_user("What is the capital of France? Answer briefly.")
    ]

    result = agent.run(messages)

    reply = result["replies"][0]
    print(f"User: What is the capital of France?")
    print(f"Agent: {reply.text}")
    print(f"Model: {reply.meta.get('model')}")
    print(f"Tokens used: {reply.meta.get('usage', {}).get('total_tokens', 'N/A')}")


# =============================================================================
# EXAMPLE 2: Multi-turn Conversation
# =============================================================================


def example_multi_turn_conversation():
    """
    Multi-turn conversation example.

    Shows how to maintain conversation context across multiple turns.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multi-turn Conversation")
    print("=" * 60)

    agent = MistralAgentSDK(agent_id=get_agent_id())
    messages = []

    # Turn 1
    print("\n--- Turn 1 ---")
    messages.append(ChatMessage.from_user("My favorite color is blue. Remember this."))
    result = agent.run(messages)
    assistant_reply = result["replies"][0]
    messages.append(assistant_reply)
    print(f"User: My favorite color is blue. Remember this.")
    print(f"Agent: {assistant_reply.text}")

    # Turn 2
    print("\n--- Turn 2 ---")
    messages.append(ChatMessage.from_user("What's my favorite color?"))
    result = agent.run(messages)
    assistant_reply = result["replies"][0]
    messages.append(assistant_reply)
    print(f"User: What's my favorite color?")
    print(f"Agent: {assistant_reply.text}")

    # Turn 3
    print("\n--- Turn 3 ---")
    messages.append(ChatMessage.from_user("Suggest a car color for me based on my preference."))
    result = agent.run(messages)
    print(f"User: Suggest a car color for me based on my preference.")
    print(f"Agent: {result['replies'][0].text}")


# =============================================================================
# EXAMPLE 3: Streaming
# =============================================================================


def example_streaming():
    """
    Streaming example: Receive tokens as they are generated.

    Useful for real-time UI updates and long responses.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Streaming")
    print("=" * 60)

    # Collector to track all chunks
    all_chunks = []

    def streaming_callback(chunk: StreamingChunk) -> None:
        """Called for each token received."""
        if chunk.content:
            print(chunk.content, end="", flush=True)
            all_chunks.append(chunk)

    agent = MistralAgentSDK(
        agent_id=get_agent_id(),
        streaming_callback=streaming_callback
    )

    messages = [
        ChatMessage.from_user("Write a haiku about programming.")
    ]

    print("User: Write a haiku about programming.")
    print("Agent: ", end="")

    result = agent.run(messages)

    print(f"\n\nTotal chunks received: {len(all_chunks)}")
    print(f"Final text length: {len(result['replies'][0].text or '')}")


# =============================================================================
# EXAMPLE 4: Tool Calling
# =============================================================================


def example_tool_calling():
    """
    Tool calling example: Let the agent use custom functions.

    This shows how to define tools, handle tool calls, and continue the conversation.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Tool Calling")
    print("=" * 60)

    # Define tool functions
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Get current weather for a city."""
        weather_data = {
            "Paris": {"temp": 18, "condition": "partly cloudy", "humidity": 65},
            "London": {"temp": 12, "condition": "rainy", "humidity": 80},
            "Tokyo": {"temp": 22, "condition": "sunny", "humidity": 55},
            "New York": {"temp": 15, "condition": "windy", "humidity": 60},
        }
        data = weather_data.get(city, {"temp": 20, "condition": "unknown", "humidity": 50})

        if unit == "fahrenheit":
            temp = int(data["temp"] * 9/5 + 32)
            unit_str = "°F"
        else:
            temp = data["temp"]
            unit_str = "°C"

        return json.dumps({
            "city": city,
            "temperature": f"{temp}{unit_str}",
            "condition": data["condition"],
            "humidity": f"{data['humidity']}%"
        })

    # Create tool
    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather for a city. Returns temperature, conditions, and humidity.",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit (default: celsius)"
                }
            },
            "required": ["city"]
        },
        function=get_weather,
    )

    # Create agent with tool
    agent = MistralAgentSDK(
        agent_id=get_agent_id(),
        tools=[weather_tool],
        tool_choice="auto"
    )

    # Initial request
    messages = [
        ChatMessage.from_user("What's the weather like in Paris today?")
    ]
    print("User: What's the weather like in Paris today?")

    result = agent.run(messages)
    assistant_message = result["replies"][0]

    if assistant_message.tool_calls:
        print(f"\nAgent requested tool calls:")
        for tc in assistant_message.tool_calls:
            print(f"  - {tc.tool_name}({tc.arguments})")

        # Execute tools and collect results
        messages.append(assistant_message)
        for tool_call in assistant_message.tool_calls:
            # Execute the tool
            tool_result = get_weather(**tool_call.arguments)
            print(f"\nTool result: {tool_result}")

            # Add tool result to messages
            messages.append(
                ChatMessage.from_tool(tool_result=tool_result, origin=tool_call)
            )

        # Get final response
        result = agent.run(messages)
        print(f"\nAgent: {result['replies'][0].text}")
    else:
        print(f"Agent: {assistant_message.text}")


# =============================================================================
# EXAMPLE 5: Multiple Tools
# =============================================================================


def example_multiple_tools():
    """
    Multiple tools example: Agent can choose from several tools.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Multiple Tools")
    print("=" * 60)

    # Define multiple tool functions
    def search_products(query: str, category: Optional[str] = None) -> str:
        """Search for products."""
        products = {
            "laptop": [
                {"name": "MacBook Pro", "price": 1999, "rating": 4.8},
                {"name": "ThinkPad X1", "price": 1599, "rating": 4.6},
            ],
            "phone": [
                {"name": "iPhone 15", "price": 999, "rating": 4.7},
                {"name": "Pixel 8", "price": 699, "rating": 4.5},
            ],
        }
        # Simple search logic
        results = []
        for cat, items in products.items():
            if category and cat != category:
                continue
            for item in items:
                if query.lower() in item["name"].lower() or query.lower() in cat:
                    results.append(item)
        return json.dumps({"results": results[:3], "total": len(results)})

    def get_product_details(product_name: str) -> str:
        """Get detailed info about a product."""
        details = {
            "MacBook Pro": {
                "name": "MacBook Pro",
                "specs": "M3 Pro chip, 18GB RAM, 512GB SSD",
                "warranty": "1 year",
                "in_stock": True
            },
            "iPhone 15": {
                "name": "iPhone 15",
                "specs": "A17 chip, 128GB storage, 6.1\" display",
                "warranty": "1 year",
                "in_stock": True
            },
        }
        return json.dumps(details.get(product_name, {"error": "Product not found"}))

    def calculate_shipping(destination: str, weight_kg: float) -> str:
        """Calculate shipping cost."""
        base_rates = {"US": 5, "EU": 8, "Asia": 12, "Other": 15}
        region = "US" if destination.lower() in ["usa", "us", "united states"] else "Other"
        cost = base_rates[region] + (weight_kg * 2)
        return json.dumps({"destination": destination, "cost": f"${cost:.2f}", "days": "3-5"})

    # Create tools
    tools = [
        Tool(
            name="search_products",
            description="Search for products in the store catalog",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "description": "Product category (optional)"}
                },
                "required": ["query"]
            },
            function=search_products,
        ),
        Tool(
            name="get_product_details",
            description="Get detailed information about a specific product",
            parameters={
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Name of the product"}
                },
                "required": ["product_name"]
            },
            function=get_product_details,
        ),
        Tool(
            name="calculate_shipping",
            description="Calculate shipping cost to a destination",
            parameters={
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Shipping destination country"},
                    "weight_kg": {"type": "number", "description": "Package weight in kg"}
                },
                "required": ["destination", "weight_kg"]
            },
            function=calculate_shipping,
        ),
    ]

    agent = MistralAgentSDK(
        agent_id=get_agent_id(),
        tools=tools,
        parallel_tool_calls=True
    )

    messages = [
        ChatMessage.from_user(
            "I'm looking for a laptop. Can you search for laptops and tell me about the MacBook Pro?"
        )
    ]
    print("User:", messages[0].text)

    result = agent.run(messages, tool_choice="auto")
    reply = result["replies"][0]

    if reply.tool_calls:
        print(f"\nAgent requested {len(reply.tool_calls)} tool call(s):")
        messages.append(reply)

        for tc in reply.tool_calls:
            print(f"  - {tc.tool_name}: {tc.arguments}")

            # Find and execute the tool
            tool_func = next(t.function for t in tools if t.name == tc.tool_name)
            tool_result = tool_func(**tc.arguments)
            print(f"    Result: {tool_result[:100]}...")

            messages.append(ChatMessage.from_tool(tool_result=tool_result, origin=tc))

        # Get final response
        result = agent.run(messages)
        print(f"\nAgent: {result['replies'][0].text}")
    else:
        print(f"Agent: {reply.text}")


# =============================================================================
# EXAMPLE 6: Toolset
# =============================================================================


def example_toolset():
    """
    Toolset example: Group related tools together.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Using Toolset")
    print("=" * 60)

    # Define math tools
    def add(a: float, b: float) -> str:
        return f"{a} + {b} = {a + b}"

    def subtract(a: float, b: float) -> str:
        return f"{a} - {b} = {a - b}"

    def multiply(a: float, b: float) -> str:
        return f"{a} × {b} = {a * b}"

    def divide(a: float, b: float) -> str:
        if b == 0:
            return "Error: Division by zero"
        return f"{a} ÷ {b} = {a / b}"

    # Create tools
    math_tools = [
        Tool(name="add", description="Add two numbers", parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"}, "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }, function=add),
        Tool(name="subtract", description="Subtract two numbers", parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"}, "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }, function=subtract),
        Tool(name="multiply", description="Multiply two numbers", parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"}, "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }, function=multiply),
        Tool(name="divide", description="Divide two numbers", parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"}, "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }, function=divide),
    ]

    # Create toolset
    math_toolset = Toolset(math_tools)

    agent = MistralAgentSDK(
        agent_id=get_agent_id(),
        tools=[math_toolset]
    )

    messages = [ChatMessage.from_user("What is 15 multiplied by 7?")]
    print("User: What is 15 multiplied by 7?")

    result = agent.run(messages, tool_choice="any")
    reply = result["replies"][0]

    if reply.tool_calls:
        tc = reply.tool_calls[0]
        print(f"Tool called: {tc.tool_name}({tc.arguments})")

        # Execute
        tool_func = next(t.function for t in math_tools if t.name == tc.tool_name)
        tool_result = tool_func(**tc.arguments)
        print(f"Result: {tool_result}")

        messages.append(reply)
        messages.append(ChatMessage.from_tool(tool_result=tool_result, origin=tc))
        result = agent.run(messages)
        print(f"Agent: {result['replies'][0].text}")
    else:
        print(f"Agent: {reply.text}")


# =============================================================================
# EXAMPLE 7: Structured Output (JSON)
# =============================================================================


def example_structured_output():
    """
    Structured output example: Force JSON responses with a schema.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Structured Output (JSON)")
    print("=" * 60)

    # Define JSON schema
    person_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "Person",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "occupation": {"type": "string"},
                    "hobbies": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "age", "occupation", "hobbies"],
                "additionalProperties": False
            }
        }
    }

    agent = MistralAgentSDK(
        agent_id=get_agent_id(),
        generation_kwargs={"response_format": person_schema}
    )

    messages = [
        ChatMessage.from_user(
            "Generate a fictional person profile for a 28-year-old software engineer named Alex "
            "who enjoys hiking and photography."
        )
    ]
    print("User: Generate a fictional person profile...")

    result = agent.run(messages)
    reply = result["replies"][0]

    print(f"\nRaw response: {reply.text}")

    # Parse and display structured data
    data = json.loads(reply.text)
    print(f"\nParsed data:")
    print(f"  Name: {data['name']}")
    print(f"  Age: {data['age']}")
    print(f"  Occupation: {data['occupation']}")
    print(f"  Hobbies: {', '.join(data['hobbies'])}")


# =============================================================================
# EXAMPLE 8: Pipeline Integration
# =============================================================================


def example_pipeline():
    """
    Pipeline example: Use MistralAgentSDK in a Haystack pipeline.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Pipeline Integration")
    print("=" * 60)

    # Define a simple tool
    def lookup_order(order_id: str) -> str:
        """Look up order status."""
        orders = {
            "ORD-123": {"status": "shipped", "eta": "2 days"},
            "ORD-456": {"status": "processing", "eta": "5 days"},
            "ORD-789": {"status": "delivered", "eta": "N/A"},
        }
        return json.dumps(orders.get(order_id, {"error": "Order not found"}))

    order_tool = Tool(
        name="lookup_order",
        description="Look up the status of an order by ID",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID (e.g., ORD-123)"}
            },
            "required": ["order_id"]
        },
        function=lookup_order,
    )

    # Build pipeline
    pipeline = Pipeline()
    pipeline.add_component(
        "agent",
        MistralAgentSDK(agent_id=get_agent_id(), tools=[order_tool])
    )
    pipeline.add_component(
        "tool_invoker",
        ToolInvoker(tools=[order_tool])
    )
    pipeline.connect("agent.replies", "tool_invoker.messages")

    print("Pipeline structure:")
    print("  [agent] --> [tool_invoker]")

    # Run pipeline
    result = pipeline.run({
        "agent": {
            "messages": [ChatMessage.from_user("Check the status of order ORD-123")],
            "tool_choice": "any"
        }
    })

    print(f"\nUser: Check the status of order ORD-123")

    if "tool_invoker" in result:
        tool_messages = result["tool_invoker"]["tool_messages"]
        if tool_messages:
            tool_result = tool_messages[0].tool_call_result.result
            print(f"Tool result: {tool_result}")


# =============================================================================
# EXAMPLE 9: Generation Parameters
# =============================================================================


def example_generation_params():
    """
    Generation parameters example: Control output behavior.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Generation Parameters")
    print("=" * 60)

    # Example with various generation parameters
    agent = MistralAgentSDK(
        agent_id=get_agent_id(),
        generation_kwargs={
            "max_tokens": 100,       # Limit response length
            "random_seed": 42,       # For reproducibility
            # "frequency_penalty": 0.5,  # Reduce repetition
            # "presence_penalty": 0.5,   # Encourage diversity
        }
    )

    messages = [
        ChatMessage.from_user("Write a creative tagline for a coffee shop.")
    ]
    print("User: Write a creative tagline for a coffee shop.")

    # Run multiple times to show determinism with seed
    print("\nRunning 3 times with same random_seed:")
    for i in range(3):
        result = agent.run(messages)
        print(f"  Run {i+1}: {result['replies'][0].text[:80]}...")


# =============================================================================
# EXAMPLE 10: Async Operations
# =============================================================================


async def example_async():
    """
    Async example: Non-blocking API calls.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Async Operations")
    print("=" * 60)

    agent = MistralAgentSDK(agent_id=get_agent_id())

    # Concurrent requests
    messages_list = [
        [ChatMessage.from_user("What is Python?")],
        [ChatMessage.from_user("What is JavaScript?")],
        [ChatMessage.from_user("What is Rust?")],
    ]

    print("Sending 3 concurrent requests...")

    # Run all requests concurrently
    tasks = [agent.run_async(msgs) for msgs in messages_list]
    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        reply = result["replies"][0]
        print(f"\nQuery {i+1}: {messages_list[i][0].text}")
        print(f"Response: {reply.text[:100]}...")


# =============================================================================
# EXAMPLE 11: Error Handling
# =============================================================================


def example_error_handling():
    """
    Error handling example: Gracefully handle API errors.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Error Handling")
    print("=" * 60)

    # Example 1: Invalid agent ID
    print("\n1. Testing with invalid agent ID:")
    try:
        agent = MistralAgentSDK(agent_id="invalid-agent-id-xyz")
        agent.run([ChatMessage.from_user("Hello")])
    except ValueError as e:
        print(f"   Caught ValueError: {str(e)[:80]}...")
    except Exception as e:
        print(f"   Caught {type(e).__name__}: {str(e)[:80]}...")

    # Example 2: Empty messages
    print("\n2. Testing with empty messages:")
    agent = MistralAgentSDK(agent_id=get_agent_id())
    result = agent.run([])
    print(f"   Result: {result}")  # Should return {"replies": []}

    # Example 3: Handling tool call errors gracefully
    print("\n3. Testing tool with potential errors:")

    def risky_tool(value: str) -> str:
        """A tool that might fail."""
        if value == "error":
            raise ValueError("Simulated error")
        return f"Processed: {value}"

    tool = Tool(
        name="risky_tool",
        description="A tool that processes values",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"]
        },
        function=risky_tool,
    )

    agent_with_tool = MistralAgentSDK(agent_id=get_agent_id(), tools=[tool])

    # This would need the agent to actually call the tool to demonstrate error handling
    print("   Tool registered successfully")


# =============================================================================
# EXAMPLE 12: Serialization
# =============================================================================


def example_serialization():
    """
    Serialization example: Save and load agent configuration.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Serialization")
    print("=" * 60)

    def dummy_callback(chunk):
        pass

    def dummy_tool_func(x: str) -> str:
        return x

    # Create agent with various configurations
    original_agent = MistralAgentSDK(
        agent_id=get_agent_id(),
        tools=[
            Tool(
                name="dummy",
                description="A dummy tool",
                parameters={"type": "object", "properties": {"x": {"type": "string"}}},
                function=dummy_tool_func,
            )
        ],
        tool_choice="auto",
        generation_kwargs={"max_tokens": 500, "random_seed": 123},
        timeout_ms=60000,
        max_retries=5,
    )

    # Serialize to dict
    serialized = original_agent.to_dict()
    print("Serialized configuration:")
    print(f"  Type: {serialized['type']}")
    print(f"  Agent ID: {serialized['init_parameters']['agent_id']}")
    print(f"  Tool choice: {serialized['init_parameters']['tool_choice']}")
    print(f"  Generation kwargs: {serialized['init_parameters']['generation_kwargs']}")

    # Deserialize
    restored_agent = MistralAgentSDK.from_dict(serialized)
    print("\nRestored agent:")
    print(f"  Agent ID: {restored_agent.agent_id}")
    print(f"  Tool choice: {restored_agent.tool_choice}")
    print(f"  Timeout: {restored_agent.timeout_ms}ms")

    # Pipeline serialization
    print("\nPipeline YAML serialization:")
    pipeline = Pipeline()
    pipeline.add_component("agent", MistralAgentSDK(agent_id=get_agent_id()))
    yaml_str = pipeline.dumps()
    print(f"  YAML length: {len(yaml_str)} characters")
    print(f"  First 200 chars: {yaml_str[:200]}...")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run all examples."""
    print("=" * 60)
    print("MistralAgentSDK - Comprehensive Examples")
    print("=" * 60)

    # Check environment
    if not os.environ.get("MISTRAL_API_KEY"):
        print("\nError: MISTRAL_API_KEY environment variable not set")
        print("Set it with: export MISTRAL_API_KEY='your-api-key'")
        sys.exit(1)

    if not os.environ.get("MISTRAL_AGENT_ID"):
        print("\nError: MISTRAL_AGENT_ID environment variable not set")
        print("Create an agent at https://console.mistral.ai/ and set:")
        print("  export MISTRAL_AGENT_ID='your-agent-id'")
        sys.exit(1)

    print(f"\nUsing Agent ID: {os.environ.get('MISTRAL_AGENT_ID')}")

    # Run examples
    try:
        example_basic_chat()
        example_multi_turn_conversation()
        example_streaming()
        example_tool_calling()
        example_multiple_tools()
        example_toolset()
        example_structured_output()
        example_pipeline()
        example_generation_params()

        # Run async example
        asyncio.run(example_async())

        example_error_handling()
        example_serialization()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        raise

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

