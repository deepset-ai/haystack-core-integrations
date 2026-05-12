import inspect
from asyncio import Semaphore, gather
from typing import Any, Union, cast, get_args, get_origin

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from pydantic import ValidationError

from haystack_integrations.components.evaluators.ragas.utils import _deserialize_metric, _serialize_metric
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import SimpleBaseMetric
from ragas.metrics.result import MetricResult


@component
class RagasEvaluator:
    """
    A component that uses the Ragas framework to evaluate inputs against specified Ragas metrics.

    See the [Ragas framework](https://docs.ragas.io/) for more details.

    This component supports the modern Ragas metrics API (`ragas.metrics.collections`).
    Each metric must be a `SimpleBaseMetric` instance with its LLM configured at construction time.

    Usage example:
    ```python
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness
    from haystack_integrations.components.evaluators.ragas import RagasEvaluator

    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)

    evaluator = RagasEvaluator(
        ragas_metrics=[Faithfulness(llm=llm)],
    )
    output = evaluator.run(
        query="Which is the most popular global sport?",
        documents=[
            "Football is undoubtedly the world's most popular sport with"
            " major events like the FIFA World Cup and sports personalities"
            " like Ronaldo and Messi, drawing a followership of more than 4"
            " billion people."
        ],
        reference="Football is the most popular sport with around 4 billion"
                  " followers worldwide",
    )

    output['result']
    ```
    """

    def __init__(self, ragas_metrics: list[SimpleBaseMetric], concurrency_limit: int = 4) -> None:
        """
        Constructs a new Ragas evaluator.

        :param ragas_metrics: A list of modern Ragas metrics from `ragas.metrics.collections`.
            Each metric must be fully configured (including its LLM) at construction time.
            Available metrics can be found in the
            [Ragas documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/).
        :param concurrency_limit: The maximum number of metric evaluations that should be allowed to run concurrently.
            This parameter is only used in the `run_async` method.
        """
        self._validate_inputs(ragas_metrics)
        self.metrics = ragas_metrics
        self.concurrency_limit = concurrency_limit

    @staticmethod
    def _validate_inputs(metrics: list[SimpleBaseMetric]) -> None:
        """
        Validate input parameters.

        :param metrics: List of Ragas metrics to validate.
        :return: None.
        """
        if not all(isinstance(metric, SimpleBaseMetric) for metric in metrics):
            error_message = "All items in ragas_metrics must be instances of SimpleBaseMetric."
            raise TypeError(error_message)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, ragas_metrics=[_serialize_metric(m) for m in self.metrics])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagasEvaluator":
        """
        Deserialize this component from a dictionary.

        Metrics are reconstructed from their stored class path and LLM/embedding
        configuration. Only the `openai` provider is supported for automatic
        deserialization; the API key is read from the `OPENAI_API_KEY` environment
        variable at load time.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        metrics_data = data.get("init_parameters", {}).get("ragas_metrics", [])
        data["init_parameters"]["ragas_metrics"] = [_deserialize_metric(m) for m in metrics_data]
        return default_from_dict(cls, data)

    @component.output_types(result=dict[str, dict[str, MetricResult]])
    def run(
        self,
        query: str | None = None,
        response: list[ChatMessage] | str | None = None,
        documents: list[Document | str] | None = None,
        reference_contexts: list[str] | None = None,
        multi_responses: list[str] | None = None,
        reference: str | None = None,
        rubrics: dict[str, str] | None = None,
    ) -> dict[str, dict[str, MetricResult]]:
        """
        Evaluates the provided inputs against each metric and returns the results.

        :param query: The input query from the user.
        :param response: A list of ChatMessage responses (typically from a language model or agent).
        :param documents: A list of Haystack Document or strings that were retrieved for the query.
        :param reference_contexts: A list of reference contexts that should have been retrieved for the query.
        :param multi_responses: List of multiple responses generated for the query.
        :param reference: A string reference answer for the query.
        :param rubrics: A dictionary of evaluation rubric, where keys represent the score
                        and the values represent the corresponding evaluation criteria.
        :return: A dictionary with key `result` mapping metric names to their `MetricResult`.
        """
        processed_docs = self._process_documents(documents)
        processed_response = self._process_response(response)

        try:
            sample = SingleTurnSample(
                user_input=query,
                retrieved_contexts=processed_docs,
                reference_contexts=reference_contexts,
                response=processed_response,
                multi_responses=multi_responses,
                reference=reference,
                rubrics=rubrics,
            )
        except ValidationError as e:
            self._handle_conversion_error(e)

        results: dict[str, MetricResult] = {}
        for metric in self.metrics:
            results[metric.name] = self._score_metric(metric, sample)

        return {"result": results}

    @component.output_types(result=dict[str, dict[str, MetricResult]])
    async def run_async(
        self,
        query: str | None = None,
        response: list[ChatMessage] | str | None = None,
        documents: list[Document | str] | None = None,
        reference_contexts: list[str] | None = None,
        multi_responses: list[str] | None = None,
        reference: str | None = None,
        rubrics: dict[str, str] | None = None,
    ) -> dict[str, dict[str, MetricResult]]:
        """
        Asynchronously evaluates the provided inputs against each metric and returns the results.

        :param query: The input query from the user.
        :param response: A list of ChatMessage responses (typically from a language model or agent).
        :param documents: A list of Haystack Document or strings that were retrieved for the query.
        :param reference_contexts: A list of reference contexts that should have been retrieved for the query.
        :param multi_responses: List of multiple responses generated for the query.
        :param reference: A string reference answer for the query.
        :param rubrics: A dictionary of evaluation rubric, where keys represent the score
                        and the values represent the corresponding evaluation criteria.
        :return: A dictionary with key `result` mapping metric names to their `MetricResult`.
        """
        processed_docs = self._process_documents(documents)
        processed_response = self._process_response(response)

        try:
            sample = SingleTurnSample(
                user_input=query,
                retrieved_contexts=processed_docs,
                reference_contexts=reference_contexts,
                response=processed_response,
                multi_responses=multi_responses,
                reference=reference,
                rubrics=rubrics,
            )
        except ValidationError as e:
            self._handle_conversion_error(e)

        sem = Semaphore(max(1, self.concurrency_limit))

        async def _runner(metric: SimpleBaseMetric) -> tuple[str, MetricResult]:
            async with sem:
                return metric.name, await self._score_metric_async(metric, sample)

        pairs = await gather(*[_runner(m) for m in self.metrics])
        results: dict[str, MetricResult] = dict(pairs)

        return {"result": results}

    def _score_metric(self, metric: SimpleBaseMetric, sample: SingleTurnSample) -> MetricResult:
        """
        Score a metric by inspecting its ascore() signature and passing only matching sample fields.

        :param metric: A SimpleBaseMetric instance to score.
        :param sample: The SingleTurnSample holding all available input fields.
        :return: MetricResult from the metric.
        """
        sig = inspect.signature(metric.ascore)
        excluded = {"self", "callbacks"}
        valid_params = {
            name
            for name, param in sig.parameters.items()
            if name not in excluded
            and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        sample_dict = sample.model_dump()
        kwargs = {k: v for k, v in sample_dict.items() if k in valid_params and v is not None}
        return metric.score(**kwargs)

    async def _score_metric_async(self, metric: SimpleBaseMetric, sample: SingleTurnSample) -> MetricResult:
        """
        Score a metric by inspecting its ascore() signature and passing only matching sample fields.

        :param metric: A SimpleBaseMetric instance to score.
        :param sample: The SingleTurnSample holding all available input fields.
        :return: MetricResult from the metric.
        """
        sig = inspect.signature(metric.ascore)
        excluded = {"self", "callbacks"}
        valid_params = {
            name
            for name, param in sig.parameters.items()
            if name not in excluded
            and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        sample_dict = sample.model_dump()
        kwargs = {k: v for k, v in sample_dict.items() if k in valid_params and v is not None}
        return await metric.ascore(**kwargs)

    def _process_documents(self, documents: list[Document | str] | None) -> list[str] | None:
        """
        Process and validate input documents.

        :param documents: List of Documents or strings to process.
        :return: List of document contents as strings or None.
        """
        if documents is None:
            return None

        if isinstance(documents, list) and all(isinstance(doc, str) for doc in documents):
            # we need to check types again in the list comprehension to make mypy happy
            return [doc for doc in documents if isinstance(doc, str)]

        if isinstance(documents, list) and all(isinstance(doc, Document) for doc in documents):
            # we need to check types again in the list comprehension to make mypy happy
            return [doc.content for doc in documents if isinstance(doc, Document) and doc.content]

        error_message = "'documents' must be a list of either Documents or strings."
        raise ValueError(error_message)

    def _process_response(self, response: list[ChatMessage] | str | None) -> str | None:
        """
        Process response into expected format.

        :param response: Response to process.
        :return: None or processed response string.
        """
        if isinstance(response, list):
            if all(isinstance(item, ChatMessage) and item.text for item in response):
                return response[0].text
            return None
        elif isinstance(response, str):
            return response
        return response

    def _handle_conversion_error(self, error: Exception) -> None:
        """
        Re-raise pydantic validation errors from SingleTurnSample with Haystack-friendly field names.

        :params error: Original error.
        """
        if isinstance(error, ValidationError):
            field_mapping = {"user_input": "query", "retrieved_contexts": "documents"}
            for err in error.errors():
                # loc is a tuple of strings and ints but according to pydantic docs, the first element is a string
                # https://docs.pydantic.dev/latest/errors/errors/
                field = cast(str, err["loc"][0])
                haystack_field = field_mapping.get(field, field)
                expected_type = self.run.__annotations__.get(haystack_field)
                type_desc = self._get_expected_type_description(expected_type)
                actual_type = type(err["input"]).__name__
                example = self._get_example_input(haystack_field)
                error_message = (
                    f"Validation error occurred while running RagasEvaluator Component:\n"
                    f"The '{haystack_field}' field expected '{type_desc}', "
                    f"but got '{actual_type}'.\n"
                    f"Hint: Provide {example}"
                )
                raise ValueError(error_message)

    def _get_expected_type_description(self, expected_type: Any) -> str:
        """Helper method to get a description of the expected type."""
        if get_origin(expected_type) is Union:
            expected_types = [getattr(t, "__name__", str(t)) for t in get_args(expected_type)]
            return f"one of {', '.join(expected_types)}"
        elif get_origin(expected_type) is list:
            expected_item_type = get_args(expected_type)[0]
            item_type_name = getattr(expected_item_type, "__name__", str(expected_item_type))
            return f"a list of {item_type_name}"
        elif get_origin(expected_type) is dict:
            key_type, value_type = get_args(expected_type)
            key_type_name = getattr(key_type, "__name__", str(key_type))
            value_type_name = getattr(value_type, "__name__", str(value_type))
            return f"a dictionary with keys of type {key_type_name} and values of type {value_type_name}"
        else:
            return getattr(expected_type, "__name__", str(expected_type))

    def _get_example_input(self, field: str) -> str:
        """
        Helper method to get an example input based on the field.

        :param field: Argument used to make SingleTurnSample.
        :returns: Example usage for the field.
        """
        examples = {
            "query": "A string query like 'Question?'",
            "documents": "[Document(content='Example content')]",
            "reference_contexts": "['Example string 1', 'Example string 2']",
            "response": "ChatMessage.from_assistant('Hi')",
            "multi_responses": "['Response 1', 'Response 2']",
            "reference": "'A reference string'",
            "rubrics": "{'score1': 'high_similarity'}",
        }
        return examples.get(field, "An appropriate value based on the field's type")
