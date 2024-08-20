import sys
from typing import Dict
from haystack.dataclasses import ChatMessage
from converse_generator import AmazonBedrockConverseGenerator
from utils import ConverseMessage, ToolConfig
from haystack.core.pipeline import Pipeline


def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a given location"""
    # This is a mock function, replace with actual API call
    return f"The weather in {location} is 22 degrees {unit}."


def get_current_time(timezone: str) -> str:
    """Get the current time in a given timezone"""
    # This is a mock function, replace with actual time lookup
    return f"The current time in {timezone} is 14:30."


def main():
    g = AmazonBedrockConverseGenerator(model="anthropic.claude-3-haiku-20240307-v1:0")

    # Create ToolConfig from functions
    tool_config = ToolConfig.from_functions([get_current_weather, get_current_time])

    # Convert ToolConfig to dict for use in the run method
    tool_config_dict = tool_config.to_dict()

    print("Tool Config:")
    print(tool_config_dict)

    p = Pipeline()
    p.add_component("generator", g)

    print("\nRunning pipeline with tools:")
    result = p.run(
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

    print("\nPipeline Result:")
    print(result)


if __name__ == '__main__':
    main()
