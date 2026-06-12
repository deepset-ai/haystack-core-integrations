import os
from haystack import Pipeline
from haystack_integrations.components.agents.ag2 import AG2Agent

pipeline = Pipeline()
pipeline.add_component("agent", AG2Agent(
    model="gpt-4o-mini",
    system_message="Answer questions clearly and concisely. Return TERMINATE when you have finished answering.",
))

result = pipeline.run({"agent": {"query": "Explain retrieval-augmented generation."}})
print(result["agent"]["reply"])
