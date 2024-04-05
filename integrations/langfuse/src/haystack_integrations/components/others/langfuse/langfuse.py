from haystack import tracing, component

from haystack_integrations.tracing.langfuse import LangfuseTracer

from langfuse import Langfuse

# from langfuse.openai import openai  # noqa


@component
class LangfuseComponent:
    def __init__(self, name: str):
        self.name = name
        tracing.enable_tracing(LangfuseTracer(Langfuse(), name))

    @component.output_types(name=str)
    def run(self):
        return {"name": self.name}
