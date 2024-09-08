from haystack import Pipeline

from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockConverseGenerator
from haystack_integrations.components.generators.amazon_bedrock.converse.utils import ConverseMessage, ToolConfig


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a given location"""
    return f"The weather in {location} is 22 degrees {unit}."


def get_current_time(timezone: str) -> str:
    """Get the current time in a given timezone"""
    return f"The current time in {timezone} is 14:30."


generator = AmazonBedrockConverseGenerator(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    streaming_callback=print,
)

# Create ToolConfig from functions
tool_config = ToolConfig.from_functions([get_current_weather, get_current_time])

tool_config_dict = tool_config.to_dict()


pipeline = Pipeline()
pipeline.add_component("generator", generator)

result = pipeline.run(
    data={
        "generator": {
            "inference_config": {
                "temperature": 0.1,
                "maxTokens": 256,
                "topP": 0.1,
                "stopSequences": ["\\n"],
            },
            "messages": [
                ConverseMessage.from_user(["What's the weather like in Paris and what time is it in New York?"]),
            ],
            "tool_config": tool_config_dict,
        },
    },
)
print(result)
