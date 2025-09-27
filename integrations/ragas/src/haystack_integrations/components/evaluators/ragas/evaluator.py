import re
from typing import Any, Dict, List, Optional, Union, cast, get_args, get_origin

from haystack import Document, component
from haystack.dataclasses import ChatMessage
from pydantic import ValidationError

from ragas import evaluate
from ragas.dataset_schema import (
    EvaluationDataset,
    EvaluationResult,
    SingleTurnSample,
)
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.metrics import Metric


@component
class RagasEvaluator:
    """
    A component that uses the [Ragas framework](https://docs.ragas.io/) to evaluate
    inputs against specified Ragas metrics.

    Usage example:
    ```python
    from haystack.components.generators import OpenAIGenerator
    from haystack_integrations.components.evaluators.ragas import RagasEvaluator
    from ragas.metrics import ContextPrecision
    from ragas.llms import HaystackLLMWrapper

    llm = OpenAIGenerator(model="gpt-4o-mini")
    evaluator_llm = HaystackLLMWrapper(llm)

    evaluator = RagasEvaluator(
        ragas_metrics=[ContextPrecision()],
        evaluator_llm=evaluator_llm
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

    def __init__(
        self,
        ragas_metrics: List[Metric],
        evaluator_llm: Optional[BaseRagasLLM] = None,
        evaluator_embedding: Optional[BaseRagasEmbeddings] = None,
    ):
        """
        Constructs a new Ragas evaluator.

        :param ragas_metrics: A list of evaluation metrics from the [Ragas](https://docs.ragas.io/) library.
        :param evaluator_llm: A language model used by metrics that require LLMs for evaluation.
        :param evaluator_embedding: An embedding model used by metrics that require embeddings for evaluation.
        """
        self._validate_inputs(ragas_metrics, evaluator_llm, evaluator_embedding)
        self.metrics = ragas_metrics
        self.llm = evaluator_llm
        self.embedding = evaluator_embedding

    def _validate_inputs(
        self,
        metrics: List[Metric],
        llm: Optional[BaseRagasLLM],
        embedding: Optional[BaseRagasEmbeddings],
    ) -> None:
        """Validate input parameters.

        :param metrics: List of Ragas metrics to validate
        :param llm: Language model to validate
        :param embedding: Embedding model to validate

        :return: None.
        """
        if not all(isinstance(metric, Metric) for metric in metrics):
            error_message = "All items in ragas_metrics must be instances of Metric class."
            raise TypeError(error_message)

        if llm is not None and not isinstance(llm, BaseRagasLLM):
            error_message = f"Expected evaluator_llm to be BaseRagasLLM, got {type(llm).__name__}"
            raise TypeError(error_message)

        if embedding is not None and not isinstance(embedding, BaseRagasEmbeddings):
            error_message = f"Expected evaluator_embedding to be BaseRagasEmbeddings, got {type(embedding).__name__}"
            raise TypeError(error_message)

    @component.output_types(result=EvaluationResult)
    def run(
        self,
        query: Optional[str] = None,
        response: Optional[Union[List[ChatMessage], str]] = None,
        documents: Optional[List[Union[Document, str]]] = None,
        reference_contexts: Optional[List[str]] = None,
        multi_responses: Optional[List[str]] = None,
        reference: Optional[str] = None,
        rubrics: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the provided query against the documents and returns the evaluation result.

        :param query: The input query from the user.
        :param response: A list of ChatMessage responses (typically from a language model or agent).
        :param documents: A list of Haystack Document or strings that were retrieved for the query.
        :param reference_contexts: A list of reference contexts that should have been retrieved for the query.
        :param multi_responses: List of multiple responses generated for the query.
        :param reference: A string reference answer for the query.
        :param rubrics: A dictionary of evaluation rubric, where keys represent the score
                        and the values represent the corresponding evaluation criteria.
        :return: A dictionary containing the evaluation result.
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

        except (ValueError, ValidationError) as e:
            self._handle_conversion_error(e)

        dataset = EvaluationDataset([sample])

        try:
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embedding,
            )
        except (ValueError, ValidationError) as e:
            self._handle_evaluation_error(e)

        return {"result": result}

    def _process_documents(self, documents: Union[List[Union[Document, str]], None]) -> Union[List[str], None]:
        """Process and validate input documents.

        :param documents: List of Documents or strings to process
        :return: List of document contents as strings or None
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

    def _process_response(self, response: Optional[Union[List[ChatMessage], str]]) -> Union[str, None]:
        """Process response into expected format.

        :param response: Response to process
        :return: None or Processed response string
        """
        if isinstance(response, list):  # Check if response is a list
            if all(isinstance(item, ChatMessage) and item.text for item in response):
                return response[0].text
            return None
        elif isinstance(response, str):
            return response
        return response

    def _handle_conversion_error(self, error: Exception) -> None:
        """Handle evaluation errors with improved messages.

        :params error: Original error
        """
        if isinstance(error, ValidationError):
            field_mapping = {
                "user_input": "query",
                "retrieved_contexts": "documents",
            }
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
                    f"Validation error occured while running RagasEvaluator Component:\n"
                    f"The '{haystack_field}' field expected '{type_desc}', "
                    f"but got '{actual_type}'.\n"
                    f"Hint: Provide {example}"
                )
                raise ValueError(error_message)

    def _handle_evaluation_error(self, error: Exception) -> None:
        error_message = str(error)
        columns_match = re.search(r"additional columns \[(.*?)\]", error_message)
        field_mapping = {
            "user_input": "query",
            "retrieved_contexts": "documents",
        }
        if columns_match:
            columns_str = columns_match.group(1)
            columns = [col.strip().strip("'") for col in columns_str.split(",")]

            mapped_columns = [field_mapping.get(col, col) for col in columns]
            updated_columns_str = "[" + ", ".join(f"'{col}'" for col in mapped_columns) + "]"

            # Update the list of columns in the error message
            updated_error_message = error_message.replace(
                columns_match.group(0), f"additional columns {updated_columns_str}"
            )
            raise ValueError(updated_error_message)

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
            # Handle non-generic types or unknown types gracefully
            return getattr(expected_type, "__name__", str(expected_type))

    def _get_example_input(self, field: str) -> str:
        """
        Helper method to get an example input based on the field.

        :param field: Arguement used to make SingleTurnSample.
        :returns: Example usage for the field.
        """
        examples = {
            "query": "A string query like 'Question?'",
            "documents": "[Document(content='Example content')]",
            "reference_contexts": "['Example string 1', 'Example string 2']",
            "response": "ChatMessage(_content='Hi', _role='assistant')",
            "multi_responses": "['Response 1', 'Response 2']",
            "reference": "'A reference string'",
            "rubrics": "{'score1': 'high_similarity'}",
        }
        return examples.get(field, "An appropriate value based on the field's type")
