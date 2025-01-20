from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from langchain_core.embeddings import Embeddings as LangchainEmbeddings  # type: ignore
from langchain_core.language_models import BaseLanguageModel as LangchainLLM  # type: ignore

from ragas import evaluate  # type: ignore
from ragas.dataset_schema import EvaluationDataset, EvaluationResult, SingleTurnSample
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.metrics import Metric


@component
class RagasEvaluator:
    """
    A component that uses the [Ragas framework](https://docs.ragas.io/) to evaluate
    inputs against specified ragas metric.

    Usage example:
    ```python
    from haystack_integrations.components.evaluators.ragas import RagasEvaluator
    from ragas.metrics import ContextPrecision
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini")
    evaluator_llm = LangchainLLMWrapper(llm)

    evaluator = RagasEvaluator(
        ragas_metrics=[ContextPrecision()],
        evaluator_llm = evaluator_llm
    )
    output = evaluator.run(
        query="Which is the most popular global sport?",
        documents=
            [
                "Football is undoubtedly the world's most popular sport with"
                "major events like the FIFA World Cup and sports personalities"
                "like Ronaldo and Messi, drawing a followership of more than 4"
                "billion people."
            ]
        ,
        reference="Football is the most popular sport with around 4 billion" "followers worldwide",
    )

    output['result']
    ```
    """

    def __init__(
        self,
        ragas_metrics: List[Metric],
        evaluator_llm: Optional[BaseRagasLLM | LangchainLLM] = None,
        evaluator_embedding: Optional[BaseRagasEmbeddings | LangchainEmbeddings] = None,
    ):
        """
        Constructs a new Ragas evaluator.

        :param ragas_metrics:
            A list of evaluation metrics from the [Ragas](https://docs.ragas.io/) library.
            Each metric in this list must be a subclass of `BaseRagasMetric`.

        :param evaluator_llm:
            A language model used by metrics that require LLMs for evaluation.
            This can either be a Langchain LLM (`LangchainLLM`) or a Ragas LLM (`BaseRagasLLM`).

        :param evaluator_embedding:
            An embedding model used by metrics that require embeddings for evaluation.
            This can either be Langchain Embedding (`LangchainEmbeddings`) or Ragas Embedding (`BaseRagasEmbeddings`).

        :return:
            A Ragas evaluator instance.
        """
        self.llm = evaluator_llm
        self.metrics = ragas_metrics
        self.embedding = evaluator_embedding

    @component.output_types(result=EvaluationResult)
    def run(
        self,
        query: Optional[str] = None,
        response: Optional[List[ChatMessage]] = None,
        documents: Optional[List[Document | str]] = None,
        reference_contexts: Optional[List[str]] = None,
        multi_responses: Optional[List[str]] = None,
        reference: Optional[str] = None,
        rubrics: Optional[Dict[str, str]] = None,
    ):
        """
        Run method for evaluating.

        :param query:
            The input query from the user.

        :param response:
            A list of ChatMessage responses (typically from a language model or agent).

        :param documents:
            A list of Haystack Document or strings that were retrieved for the query.

        :param reference_contexts:
            A list of strings of refrence contexts or the contexts that should
            have been retrieved for the query.

        :param multi_responses:
            List of multiple responses generated for the query.

        :param reference:
            A string reference answer for the query.

        :param rubrics:
            A dictionary of evaluation rubric, where keys represent the score
            and the values represent the corresponding evaluation criteria.

        :return:
            Returns an `EvaluationResult` object from ragas library containing the outcomes of the evaluation process.
        """

        if documents:
            first_type = type(documents[0])
            if first_type is Document:
                # Ensure all elements are of type Document
                if any(not isinstance(doc, Document) for doc in documents):
                    error_message = "All elements in the documents list must be of type Document."
                    raise ValueError(error_message)
                documents = [doc.content for doc in documents]  # type: ignore[union-attr]
            elif first_type is str:
                # Ensure all elements are strings
                if any(not isinstance(doc, str) for doc in documents):
                    error_message = "All elements in the documents list must be strings."
                    raise ValueError(error_message)
            else:
                error_message = "Unsupported type in documents list."
                raise ValueError(error_message)

        single_turn_sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=documents,
            reference_contexts=reference_contexts,
            response=response[0]._content[0].text if response else None,
            multi_responses=multi_responses,
            reference=reference,
            rubrics=rubrics,
        )

        evaluation_dataset = EvaluationDataset([single_turn_sample])
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embedding,
        )
        return {"result": result}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        :raises DeserializationError:
            If the component cannot be serialized.
        """
        return default_to_dict(
            self, ragas_metrics=self.metrics, evaluator_llm=self.llm, evaluator_embedding=self.embedding
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RagasEvaluator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
