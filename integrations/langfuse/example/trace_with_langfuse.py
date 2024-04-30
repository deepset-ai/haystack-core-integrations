import logging
import os
import time
from typing import Any, Dict, List

from haystack import Pipeline, component, tracing
from haystack.components.converters import OutputAdapter
from langfuse import Langfuse
from langfuse_haystack.tracing.langfuse_tracing import langfuse_session
from langfuse_haystack.tracing.tracer import LangfuseTracer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

langfuse = Langfuse(
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    host="https://cloud.langfuse.com",
)

haystack_tracer = LangfuseTracer(langfuse)
tracing.enable_tracing(haystack_tracer)
tracing.tracer.is_content_tracing_enabled = True


@component
class WaitingComponent:
    """
    A component that waits for a given number of seconds
    """

    def __init__(self, wait_time: int = 0.1):
        self.wait_time = wait_time

    @component.output_types(text=str)
    def run(self, text: str) -> str:
        time.sleep(self.wait_time)
        return {"text": text}


@component
class FailComponent:
    """
    A component that fails
    """

    error_details = "This component always fails"

    @component.output_types(text=str)
    def run(self, text_input: str, *, should_fail: bool = True) -> Dict[str, Any]:
        if should_fail:
            raise RuntimeError(self.error_details)
        return {"text": text_input}


@component
class MockedLLMGenerator:
    """
    A component that mocks the response of a language model generator
    """

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str):
        logger.info("Hello, world! %s", prompt)
        time.sleep(0.4)

        return {
            "replies": ["This is a mocked response with some text."],
            "meta": [
                {
                    "model": "gpt-3.5-turbo-0613",
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                }
            ],
        }


@component
class PipelineComponent:
    """
    A component that runs a pipeline
    """

    @component.output_types(text=str)
    def run(self, text: str) -> Dict[str, Any]:
        p = Pipeline()
        p.add_component("waiting_component1", WaitingComponent())
        p.add_component("output_adapter1", OutputAdapter(template="{{ replies[-1]}}", output_type=str))
        p.add_component(name="mocked_llm_generator1", instance=MockedLLMGenerator())

        p.connect("mocked_llm_generator1.replies", "output_adapter1.replies")
        p.connect("output_adapter1", "waiting_component1.text")

        retvalue = p.run(data={"mocked_llm_generator1": {"prompt": text}})

        return {"text": retvalue["waiting_component1"]}


pipeline = Pipeline()
pipeline.add_component("waiting_component", WaitingComponent())
pipeline.add_component("output_adapter", OutputAdapter(template="{{ replies[-1] }}", output_type=str))
pipeline.add_component(name="mocked_llm_generator", instance=MockedLLMGenerator())
pipeline.add_component(name="pipeline_component", instance=PipelineComponent())
pipeline.add_component(name="failing_component", instance=FailComponent())

pipeline.connect("mocked_llm_generator.replies", "output_adapter.replies")
pipeline.connect("output_adapter.output", "failing_component.text_input")
pipeline.connect("failing_component", "pipeline_component.text")
pipeline.connect("pipeline_component.text", "waiting_component.text")


if __name__ == "__main__":
    logger.info(pipeline)
    try:
        with langfuse_session():
            pipeline.run(
                data={"mocked_llm_generator": {"prompt": "Hello, world!"}, "failing_component": {"should_fail": True}}
            )
    except RuntimeError as e:
        logger.debug(e)

    pipeline.run(
        data={"mocked_llm_generator": {"prompt": "Hello, world!"}, "failing_component": {"should_fail": False}}
    )

    langfuse.flush()
