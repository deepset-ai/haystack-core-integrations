from haystack import tracing, component

from haystack_integrations.tracing.langfuse import LangfuseTracer

from langfuse import Langfuse


@component
class LangfuseComponent:
    def __init__(self, name: str):
        self.name = name
        self.tracer = LangfuseTracer(Langfuse(), name)
        tracing.enable_tracing(self.tracer)

    @component.output_types(name=str, trace_url=str)
    def run(self):
        return {"name": self.name,
                "trace_url": self.tracer.get_trace_url()}
