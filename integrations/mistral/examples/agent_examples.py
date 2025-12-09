from haystack.dataclasses import ChatMessage
from haystack_integrations.components.agents import MistralAgent

AGENT_ID="ag_019ad91a340872b7ba696e7ee24931f9"

generator = MistralAgent(agent_id=AGENT_ID)
messages = [ChatMessage.from_user("Hello! What can you help me with?")]
result = generator.run(messages)
print(result["replies"][0].text)




# Example of multi-turn conversation with MistralAgent

generator = MistralAgentGenerator(agent_id="your-agent-id")

# First turn
messages = [ChatMessage.from_user("What's the weather like in Paris?")]
result = generator.run(messages)
assistant_reply = result["replies"][0]
print(f"Agent: {assistant_reply.text}")

# Second turn - continue conversation
messages.append(assistant_reply)
messages.append(ChatMessage.from_user("And what about London?"))
result = generator.run(messages)
print(f"Agent: {result['replies'][0].text}")


# Example of streaming response with MistralAgent

from haystack.dataclasses import StreamingChunk

def my_callback(chunk: StreamingChunk) -> None:
    print(chunk.content, end="", flush=True)

generator = MistralAgent(
    agent_id=AGENT_ID,
    streaming_callback=my_callback
)

messages = [ChatMessage.from_user("Tell me a short story about a robot.")]
result = generator.run(messages)
print()  # New line after streaming



# Example with Tools (if supported by the agent)

from haystack.tools import Tool

def search_database(query: str) -> str:
    """Search internal database."""
    return f"Found 5 results for: {query}"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"


# Create tools
search_tool = Tool(
    name="search_database",
    description="Search the company database for information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    },
    function=search_database,
)

email_tool = Tool(
    name="send_email",
    description="Send an email to a recipient",
    parameters={
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Email address"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body"}
        },
        "required": ["to", "subject", "body"]
    },
    function=send_email,
)

# Agent with additional tools
generator = MistralAgent(
    agent_id=AGENT_ID,
    tools=[search_tool, email_tool],
    tool_choice="auto"  # Let agent decide when to use tools
)

messages = [ChatMessage.from_user("Search for Q3 sales reports")]
result = generator.run(messages)

# Check if agent wants to use a tool
reply = result["replies"][0]
if reply.tool_calls:
    for tool_call in reply.tool_calls:
        print(f"Tool: {tool_call.tool_name}")
        print(f"Args: {tool_call.arguments}")

        # Execute tool
        if tool_call.tool_name == "search_database":
            tool_result = search_database(**tool_call.arguments)

        # Continue conversation with tool result
        messages.append(reply)
        messages.append(ChatMessage.from_tool(tool_result=tool_result, origin=tool_call))
        final_result = generator.run(messages)
        print(f"Final: {final_result['replies'][0].text}")



## Example of customizing agent parameters

generator = MistralAgent(
    agent_id=AGENT_ID,
    generation_kwargs={
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "TaskList",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                                    "due_date": {"type": "string"}
                                },
                                "required": ["title", "priority"]
                            }
                        }
                    },
                    "required": ["tasks"],
                    "additionalProperties": False
                }
            }
        }
    }
)

messages = [ChatMessage.from_user("Create a task list for launching a new product")]
result = generator.run(messages)

import json
tasks = json.loads(result["replies"][0].text)
for task in tasks["tasks"]:
    print(f"- [{task['priority'].upper()}] {task['title']}")


## reasoning example with MistralAgent

generator = MistralAgentGenerator(
    agent_id="your-reasoning-agent-id",
    generation_kwargs={
        "prompt_mode": "reasoning",
        "max_tokens": 2000
    }
)

messages = [
    ChatMessage.from_user(
        "Solve this step by step: A store sells apples for $2 each. "
        "If I have $15 and want to buy as many apples as possible, "
        "how many can I buy and how much change will I have?"
    )
]

result = generator.run(messages)
print(result["replies"][0].text)