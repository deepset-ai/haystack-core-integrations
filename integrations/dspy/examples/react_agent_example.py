import dspy
from haystack import Pipeline
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.dspy import DSPySignatureChatGenerator


def get_weather(city: str) -> str:
    """Return the current weather for a city (stub)."""
    weather_data = {
        "paris": "15°C, partly cloudy",
        "tokyo": "22°C, sunny",
        "new york": "8°C, rainy",
    }
    return weather_data.get(city.lower(), f"No weather data available for {city}")


def get_population(city: str) -> str:
    """Return the population of a city (stub)."""
    population_data = {
        "paris": "2.1 million (city proper), 12.4 million (metro)",
        "tokyo": "13.9 million (city proper), 37.4 million (metro)",
        "new york": "8.3 million (city proper), 19.8 million (metro)",
    }
    return population_data.get(city.lower(), f"No population data available for {city}")


class CityInfoSignature(dspy.Signature):
    """Answer questions about cities using available tools."""

    question = dspy.InputField(desc="A question about a city")
    answer = dspy.OutputField(desc="A detailed answer based on tool results")


def react_agent_example():
    """Use ReAct to answer a question that requires tool calls."""

    generator = DSPySignatureChatGenerator(
        model="openai/gpt-5-mini",
        signature=CityInfoSignature,
        module_type="ReAct",
        output_field="answer",
        module_kwargs={"tools": [get_weather, get_population]},
    )

    pipeline = Pipeline()
    pipeline.add_component("agent", generator)

    messages = [ChatMessage.from_user("What is the weather and population of Tokyo?")]
    result = pipeline.run({"agent": {"messages": messages}})

    print(f"Question: {messages[0].text}")
    print(f"Answer  : {result['agent']['replies'][0].text}\n")


def react_string_signature_example():
    """ReAct with a string signature and tools."""

    generator = DSPySignatureChatGenerator(
        model="openai/gpt-5-mini",
        signature="question -> answer",
        module_type="ReAct",
        output_field="answer",
        module_kwargs={"tools": [get_weather]},
    )

    messages = [ChatMessage.from_user("What's the weather like in Paris?")]
    result = generator.run(messages=messages)

    print(f"Question: {messages[0].text}")
    print(f"Answer  : {result['replies'][0].text}\n")


if __name__ == "__main__":
    react_agent_example()
    react_string_signature_example()
