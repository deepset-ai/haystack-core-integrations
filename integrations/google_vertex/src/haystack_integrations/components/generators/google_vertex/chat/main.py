from vertexai.generative_models import Tool, FunctionDeclaration

get_current_weather_func = FunctionDeclaration(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
            "unit": {
                "type": "string",
                "enum": [
                    "celsius",
                    "fahrenheit",
                ],
            },
        },
        "required": ["location"],
    },
)
tool = Tool([get_current_weather_func])

def get_current_weather(location: str, unit: str = "celsius"):
  return {"weather": "sunny", "temperature": 21.8, "unit": unit}

from haystack_integrations.components.generators.google_vertex import VertexAIGeminiChatGenerator


gemini_chat = VertexAIGeminiChatGenerator(project_id="my-project-1487737228087", tools=[tool])
from haystack.dataclasses import ChatMessage


messages = [ChatMessage.from_user("What is the temperature in celsius in Berlin?")]
res = gemini_chat.run(messages=messages)
print ("RESPONSE")
print (res)
print(res["replies"][0].content)

weather = get_current_weather(**res["replies"][0].content)

messages += res["replies"] + [ChatMessage.from_function(content=weather, name="get_current_weather")]

res = gemini_chat.run(messages=messages)
print (res)
print(res["replies"][0].content)
