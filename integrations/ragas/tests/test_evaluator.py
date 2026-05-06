import inspect
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack import AsyncPipeline, Document, Pipeline
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from openai import AsyncOpenAI
from ragas.embeddings.base import BaseRagasEmbedding, BaseRagasEmbeddings, embedding_factory
from ragas.llms import InstructorBaseRagasLLM, llm_factory
from ragas.metrics.base import SimpleBaseMetric
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness
from ragas.metrics.result import MetricResult

from haystack_integrations.components.evaluators.ragas import RagasEvaluator


class ConcreteMetric(SimpleBaseMetric):
    """Minimal concrete SimpleBaseMetric for serialization tests."""

    def __init__(self, name: str = "concrete_metric", llm=None, embeddings=None):
        self.name = name
        self.llm = llm
        self.embeddings = embeddings

    async def ascore(self, user_input: str, response: str) -> MetricResult:
        return MetricResult(value=1.0, reason="test")

    def score(self, **kwargs) -> MetricResult:
        return MetricResult(value=1.0, reason="test")


def make_metric(name: str, score: float = 0.8, reason: str = "test reason") -> MagicMock:
    """Create a mock SimpleBaseMetric with a concrete ascore signature for inspect.signature."""
    metric = MagicMock(spec=SimpleBaseMetric)
    metric.name = name
    metric.score.return_value = MetricResult(value=score, reason=reason)

    async def ascore(user_input: str, response: str, retrieved_contexts: list) -> MetricResult:
        return MetricResult(value=score, reason=reason)

    metric.ascore = ascore
    return metric


def make_metric_async(name: str, score: float = 0.8, reason: str = "test reason") -> MagicMock:
    """Create a mock SimpleBaseMetric with a concrete ascore signature for inspect.signature."""
    metric = MagicMock(spec=SimpleBaseMetric)
    metric.name = name

    async def ascore(user_input: str, response: str, retrieved_contexts: list) -> MetricResult:
        return MetricResult(value=score, reason=reason)

    mock_ascore = AsyncMock(return_value=MetricResult(value=score, reason=reason))
    mock_ascore.__signature__ = inspect.signature(ascore)
    metric.ascore = mock_ascore
    return metric


class TestInit:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        metric = Faithfulness(llm=llm_factory("gpt-4o-mini", client=AsyncOpenAI()))
        evaluator = RagasEvaluator(ragas_metrics=[metric])
        assert evaluator.metrics == [metric]

    def test_init_with_multiple_metrics(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI())
        metrics = [
            Faithfulness(llm=llm),
            AnswerRelevancy(
                llm=llm, embeddings=embedding_factory("openai", model="text-embedding-3-small", client=AsyncOpenAI())
            ),
        ]
        evaluator = RagasEvaluator(ragas_metrics=metrics)
        assert len(evaluator.metrics) == 2

    def test_invalid_metrics_raises_type_error(self):
        with pytest.raises(TypeError, match=r"All items in ragas_metrics must be instances of SimpleBaseMetric."):
            RagasEvaluator(ragas_metrics=["not_a_metric"])


class TestRun:
    def test_run_returns_result_by_metric_name(self, monkeypatch):
        metric = make_metric("faithfulness", score=0.9)
        evaluator = RagasEvaluator(ragas_metrics=[metric])
        output = evaluator.run(
            query="Which is the most popular global sport?",
            response="Football is the most popular sport.",
            documents=["Football is undoubtedly the world's most popular sport."],
        )
        assert "result" in output
        assert "faithfulness" in output["result"]
        result = output["result"]["faithfulness"]
        assert isinstance(result, MetricResult)
        assert result.value == 0.9

    def test_run_scores_all_metrics(self):
        metrics = [make_metric("faithfulness", 0.9), make_metric("answer_relevancy", 0.7)]
        evaluator = RagasEvaluator(ragas_metrics=metrics)
        output = evaluator.run(query="test?", response="answer", documents=["doc"])
        assert set(output["result"].keys()) == {"faithfulness", "answer_relevancy"}
        assert output["result"]["faithfulness"].value == 0.9
        assert output["result"]["answer_relevancy"].value == 0.7

    def test_run_calls_score_on_each_metric(self):
        metric_a = make_metric("faithfulness")
        metric_b = make_metric("answer_relevancy")
        evaluator = RagasEvaluator(ragas_metrics=[metric_a, metric_b])
        evaluator.run(query="test?", response="answer", documents=["doc"])
        metric_a.score.assert_called_once()
        metric_b.score.assert_called_once()

    def test_score_metric_passes_only_matching_params(self):
        """Metric that only needs user_input + response should not receive retrieved_contexts."""
        metric = MagicMock(spec=SimpleBaseMetric)
        metric.name = "selective_metric"
        metric.score.return_value = MetricResult(value=0.5, reason="ok")

        async def ascore(user_input: str, response: str) -> MetricResult:
            return MetricResult(value=0.5, reason="ok")

        metric.ascore = ascore

        evaluator = RagasEvaluator(ragas_metrics=[metric])
        evaluator.run(query="test?", response="answer", documents=["doc"], reference="ref")
        metric.score.assert_called_once_with(user_input="test?", response="answer")

    def test_score_metric_omits_none_fields(self):
        metric = make_metric("faithfulness")
        evaluator = RagasEvaluator(ragas_metrics=[metric])
        evaluator.run(query="test?", response="answer")  # no documents → retrieved_contexts=None
        _, kwargs = metric.score.call_args
        assert "retrieved_contexts" not in kwargs

    def test_run_accepts_document_objects(self):
        metric = make_metric("faithfulness")
        evaluator = RagasEvaluator(ragas_metrics=[metric])
        evaluator.run(
            query="test?",
            response="answer",
            documents=[Document(content="some content"), Document(content="more content")],
        )
        _, kwargs = metric.score.call_args
        assert kwargs["retrieved_contexts"] == ["some content", "more content"]

    def test_run_accepts_string_documents(self):
        metric = make_metric("faithfulness")
        evaluator = RagasEvaluator(ragas_metrics=[metric])
        evaluator.run(query="test?", response="answer", documents=["doc one", "doc two"])
        _, kwargs = metric.score.call_args
        assert kwargs["retrieved_contexts"] == ["doc one", "doc two"]

    @pytest.mark.parametrize(
        "invalid_input,field_name,error_message",
        [
            (["Invalid query type"], "query", "'query' field expected"),
            ([123, ["Invalid document"]], "documents", "'documents' must be a list"),
            (["score_1"], "rubrics", "'rubrics' field expected"),
        ],
    )
    def test_run_raises_on_invalid_input_types(self, invalid_input, field_name, error_message):
        evaluator = RagasEvaluator(ragas_metrics=[make_metric("faithfulness")])
        query = "Which is the most popular global sport?"
        documents = ["Football is the most popular sport."]
        response = "Football is the most popular sport in the world"

        with pytest.raises(ValueError) as exc_info:
            if field_name == "query":
                evaluator.run(query=invalid_input, documents=documents, response=response)
            elif field_name == "documents":
                evaluator.run(query=query, documents=invalid_input, response=response)
            elif field_name == "rubrics":
                evaluator.run(query=query, rubrics=invalid_input, documents=documents, response=response)

        assert error_message in str(exc_info.value)


class TestRunAsync:
    @pytest.mark.asyncio
    async def test_run_async_returns_result_by_metric_name(self) -> None:
        metric = make_metric_async("faithfulness", score=0.9)
        evaluator = RagasEvaluator(ragas_metrics=[metric])
        output = await evaluator.run_async(
            query="Which is the most popular global sport?",
            response="Football is the most popular sport.",
            documents=["Football is undoubtedly the world's most popular sport."],
        )
        assert "result" in output
        assert "faithfulness" in output["result"]
        result = output["result"]["faithfulness"]
        assert isinstance(result, MetricResult)
        assert result.value == 0.9

    @pytest.mark.asyncio
    async def test_run_async_scores_all_metrics(self) -> None:
        metrics = [make_metric_async("faithfulness", 0.9), make_metric_async("answer_relevancy", 0.7)]
        evaluator = RagasEvaluator(ragas_metrics=metrics)
        output = await evaluator.run_async(query="test?", response="answer", documents=["doc"])
        assert set(output["result"].keys()) == {"faithfulness", "answer_relevancy"}
        assert output["result"]["faithfulness"].value == 0.9
        assert output["result"]["answer_relevancy"].value == 0.7

    @pytest.mark.asyncio
    async def test_run_async_calls_ascore_on_each_metric(self) -> None:
        metric_a = make_metric_async("faithfulness")
        metric_b = make_metric_async("answer_relevancy")
        evaluator = RagasEvaluator(ragas_metrics=[metric_a, metric_b])
        await evaluator.run_async(query="test?", response="answer", documents=["doc"])
        metric_a.ascore.assert_called_once()
        metric_b.ascore.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_metric_async_passes_only_matching_params(self) -> None:
        """Metric that only needs user_input + response should not receive retrieved_contexts."""
        metric = MagicMock(spec=SimpleBaseMetric)
        metric.name = "selective_metric"

        async def ascore(user_input: str, response: str) -> MetricResult:
            return MetricResult(value=0.5, reason="ok")

        metric.ascore = ascore

        evaluator = RagasEvaluator(ragas_metrics=[metric])
        await evaluator.run_async(query="test?", response="answer", documents=["doc"], reference="ref")
        # Only user_input and response should have been passed — not retrieved_contexts or reference
        # We wrap ascore to capture kwargs
        captured = {}

        async def capturing_ascore(user_input: str, response: str) -> MetricResult:
            captured.update({"user_input": user_input, "response": response})
            return MetricResult(value=0.5, reason="ok")

        metric.ascore = capturing_ascore
        await evaluator.run_async(query="test?", response="answer", documents=["doc"], reference="ref")
        assert set(captured.keys()) == {"user_input", "response"}

    @pytest.mark.asyncio
    async def test_score_metric_async_omits_none_fields(self) -> None:
        metric = make_metric_async("faithfulness")
        evaluator = RagasEvaluator(ragas_metrics=[metric])
        await evaluator.run_async(query="test?", response="answer")  # no documents → retrieved_contexts=None
        _, kwargs = metric.ascore.call_args
        assert "retrieved_contexts" not in kwargs

    @pytest.mark.asyncio
    async def test_run_async_accepts_document_objects(self) -> None:
        metric = make_metric_async("faithfulness")
        evaluator = RagasEvaluator(ragas_metrics=[metric])
        await evaluator.run_async(
            query="test?",
            response="answer",
            documents=[Document(content="some content"), Document(content="more content")],
        )
        _, kwargs = metric.ascore.call_args
        assert kwargs["retrieved_contexts"] == ["some content", "more content"]

    @pytest.mark.asyncio
    async def test_run_async_accepts_string_documents(self):
        metric = make_metric_async("faithfulness")
        evaluator = RagasEvaluator(ragas_metrics=[metric])
        await evaluator.run_async(query="test?", response="answer", documents=["doc one", "doc two"])
        _, kwargs = metric.ascore.call_args
        assert kwargs["retrieved_contexts"] == ["doc one", "doc two"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invalid_input,field_name,error_message",
        [
            (["Invalid query type"], "query", "'query' field expected"),
            ([123, ["Invalid document"]], "documents", "'documents' must be a list"),
            (["score_1"], "rubrics", "'rubrics' field expected"),
        ],
    )
    async def test_run_async_raises_on_invalid_input_types(self, invalid_input, field_name, error_message):
        evaluator = RagasEvaluator(ragas_metrics=[make_metric_async("faithfulness")])
        query = "Which is the most popular global sport?"
        documents = ["Football is the most popular sport."]
        response = "Football is the most popular sport in the world"

        with pytest.raises(ValueError) as exc_info:
            if field_name == "query":
                await evaluator.run_async(query=invalid_input, documents=documents, response=response)
            elif field_name == "documents":
                await evaluator.run_async(query=query, documents=invalid_input, response=response)
            elif field_name == "rubrics":
                await evaluator.run_async(query=query, rubrics=invalid_input, documents=documents, response=response)

        assert error_message in str(exc_info.value)


class TestSerialization:
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI())
        evaluator = RagasEvaluator(ragas_metrics=[ConcreteMetric(llm=llm), ConcreteMetric(name="another_metric")])
        data = evaluator.to_dict()
        assert data == {
            "type": "haystack_integrations.components.evaluators.ragas.evaluator.RagasEvaluator",
            "init_parameters": {
                "ragas_metrics": [
                    {
                        "type": "tests.test_evaluator.ConcreteMetric",
                        "name": "concrete_metric",
                        "llm": {"model": "gpt-4o-mini", "provider": "openai"},
                    },
                    {"type": "tests.test_evaluator.ConcreteMetric", "name": "another_metric"},
                ]
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        data = {
            "type": "haystack_integrations.components.evaluators.ragas.evaluator.RagasEvaluator",
            "init_parameters": {
                "ragas_metrics": [
                    {
                        "type": "tests.test_evaluator.ConcreteMetric",
                        "name": "concrete_metric",
                        "llm": {"model": "gpt-4o-mini", "provider": "openai"},
                    },
                ],
            },
        }
        reconstructed = RagasEvaluator.from_dict(data)
        assert len(reconstructed.metrics) == 1
        assert reconstructed.metrics[0].name == "concrete_metric"

    def test_from_dict_raises_for_unsupported_provider(self):
        data = {
            "type": "haystack_integrations.components.evaluators.ragas.evaluator.RagasEvaluator",
            "init_parameters": {
                "ragas_metrics": [
                    {
                        "type": "tests.test_evaluator.ConcreteMetric",
                        "name": "some_metric",
                        "llm": {"model": "gemini-pro", "provider": "google"},
                    }
                ]
            },
        }

        with pytest.raises(ValueError, match="only supports the 'openai' provider"):
            RagasEvaluator.from_dict(data)


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="Set OPENAI_API_KEY to run integration tests.")
@pytest.mark.integration
class TestStandaloneEvaluationIntegration:
    def make_llm(self):
        return llm_factory("gpt-4o-mini", client=AsyncOpenAI())

    def make_embeddings(self):
        return embedding_factory("openai", model="text-embedding-3-small", client=AsyncOpenAI())

    def test_faithfulness_returns_valid_score(self):
        evaluator = RagasEvaluator(ragas_metrics=[Faithfulness(llm=self.make_llm())])

        output = evaluator.run(
            query="What makes Meta AI's LLaMA models stand out?",
            response="Meta AI's LLaMA models stand out for being open-source.",
            documents=[
                "Meta AI is best known for its LLaMA series, which has been made open-source "
                "for researchers and developers. LLaMA models are praised for their ability to "
                "support innovation and experimentation due to their accessibility."
            ],
        )

        result = output["result"]["faithfulness"]
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.value <= 1.0

    def test_answer_relevancy_uses_only_query_and_response(self):
        """AnswerRelevancy only declares user_input + response in ascore — documents should not be forwarded."""
        evaluator = RagasEvaluator(
            ragas_metrics=[AnswerRelevancy(llm=self.make_llm(), embeddings=self.make_embeddings())]
        )

        output = evaluator.run(
            query="What makes Meta AI's LLaMA models stand out?",
            response="They are open-source and freely available to researchers.",
            documents=["Meta AI released LLaMA as an open-source model."],
        )

        result = output["result"]["answer_relevancy"]
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.value <= 1.0

    def test_multiple_metrics_all_return_results(self):
        llm = self.make_llm()
        embeddings = self.make_embeddings()
        evaluator = RagasEvaluator(
            ragas_metrics=[
                Faithfulness(llm=llm),
                AnswerRelevancy(llm=llm, embeddings=embeddings),
                ContextPrecision(llm=llm),
            ]
        )

        output = evaluator.run(
            query="What makes Meta AI's LLaMA models stand out?",
            response=(
                "Meta AI's LLaMA models stand out for being open-source, supporting "
                "innovation and experimentation due to their accessibility and strong performance."
            ),
            documents=[
                "Meta AI is best known for its LLaMA series, which has been made open-source.",
                "Meta AI with its LLaMA models aims to democratize AI development by making "
                "high-quality models available for free, fostering collaboration across industries.",
            ],
            reference=(
                "Meta AI's LLaMA models stand out for being open-source, supporting innovation "
                "and experimentation due to their accessibility and strong performance."
            ),
        )

        assert set(output["result"].keys()) == {"faithfulness", "answer_relevancy", "context_precision"}
        for metric_result in output["result"].values():
            assert isinstance(metric_result, MetricResult)
            assert 0.0 <= metric_result.value <= 1.0


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="Set OPENAI_API_KEY to run integration tests.")
@pytest.mark.integration
class TestStandaloneEvaluationIntegrationAsync:
    def make_llm(self) -> InstructorBaseRagasLLM:
        return llm_factory("gpt-4o-mini", client=AsyncOpenAI())

    def make_embeddings(self) -> BaseRagasEmbedding | BaseRagasEmbeddings:
        return embedding_factory("openai", model="text-embedding-3-small", client=AsyncOpenAI())

    @pytest.mark.asyncio
    async def test_faithfulness_returns_valid_score(self) -> None:
        evaluator = RagasEvaluator(ragas_metrics=[Faithfulness(llm=self.make_llm())])

        output = await evaluator.run_async(
            query="What makes Meta AI's LLaMA models stand out?",
            response="Meta AI's LLaMA models stand out for being open-source.",
            documents=[
                "Meta AI is best known for its LLaMA series, which has been made open-source "
                "for researchers and developers. LLaMA models are praised for their ability to "
                "support innovation and experimentation due to their accessibility."
            ],
        )

        result = output["result"]["faithfulness"]
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.value <= 1.0

    @pytest.mark.asyncio
    async def test_answer_relevancy_uses_only_query_and_response(self) -> None:
        """AnswerRelevancy only declares user_input + response in ascore — documents should not be forwarded."""
        evaluator = RagasEvaluator(
            ragas_metrics=[AnswerRelevancy(llm=self.make_llm(), embeddings=self.make_embeddings())]
        )

        output = await evaluator.run_async(
            query="What makes Meta AI's LLaMA models stand out?",
            response="They are open-source and freely available to researchers.",
            documents=["Meta AI released LLaMA as an open-source model."],
        )

        result = output["result"]["answer_relevancy"]
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.value <= 1.0

    @pytest.mark.asyncio
    async def test_multiple_metrics_all_return_results(self) -> None:
        llm = self.make_llm()
        embeddings = self.make_embeddings()
        evaluator = RagasEvaluator(
            ragas_metrics=[
                Faithfulness(llm=llm),
                AnswerRelevancy(llm=llm, embeddings=embeddings),
                ContextPrecision(llm=llm),
            ]
        )

        output = await evaluator.run_async(
            query="What makes Meta AI's LLaMA models stand out?",
            response=(
                "Meta AI's LLaMA models stand out for being open-source, supporting "
                "innovation and experimentation due to their accessibility and strong performance."
            ),
            documents=[
                "Meta AI is best known for its LLaMA series, which has been made open-source.",
                "Meta AI with its LLaMA models aims to democratize AI development by making "
                "high-quality models available for free, fostering collaboration across industries.",
            ],
            reference=(
                "Meta AI's LLaMA models stand out for being open-source, supporting innovation "
                "and experimentation due to their accessibility and strong performance."
            ),
        )

        assert set(output["result"].keys()) == {"faithfulness", "answer_relevancy", "context_precision"}
        for metric_result in output["result"].values():
            assert isinstance(metric_result, MetricResult)
            assert 0.0 <= metric_result.value <= 1.0


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="Set OPENAI_API_KEY to run integration tests.")
@pytest.mark.integration
class TestPipelineIntegration:
    def test_ragas_evaluator_in_rag_pipeline(self):
        dataset = [
            "Meta AI is best known for its LLaMA series, which has been made open-source "
            "for researchers and developers.",
            "LLaMA models are praised for their ability to support innovation and "
            "experimentation due to their accessibility and strong performance.",
            "Meta AI with its LLaMA models aims to democratize AI development by making "
            "high-quality models available for free.",
        ]

        document_store = InMemoryDocumentStore()
        docs = [Document(content=text) for text in dataset]
        document_embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
        document_store.write_documents(document_embedder.run(docs)["documents"])

        ragas_evaluator = RagasEvaluator(
            ragas_metrics=[Faithfulness(llm=llm_factory("gpt-4o-mini", client=AsyncOpenAI()))]
        )

        template = [
            ChatMessage.from_user(
                "Answer the question based on the context.\n\n"
                "Context:\n{% for document in documents %}{{ document.content }}\n{% endfor %}\n\n"
                "Question: {{question}}\nAnswer:"
            )
        ]

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", OpenAITextEmbedder(model="text-embedding-3-small"))
        pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=2))
        pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template, required_variables="*"))
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
        pipeline.add_component("answer_builder", AnswerBuilder())
        pipeline.add_component("ragas_evaluator", ragas_evaluator)

        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever", "prompt_builder")
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        pipeline.connect("llm.replies", "answer_builder.replies")
        pipeline.connect("retriever", "answer_builder.documents")
        pipeline.connect("retriever", "ragas_evaluator.documents")
        pipeline.connect("llm.replies", "ragas_evaluator.response")

        question = "What makes Meta AI's LLaMA models stand out?"
        result = pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
                "ragas_evaluator": {"query": question},
            }
        )

        assert "ragas_evaluator" in result
        faithfulness_result = result["ragas_evaluator"]["result"]["faithfulness"]
        assert isinstance(faithfulness_result, MetricResult)
        assert 0.0 <= faithfulness_result.value <= 1.0


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="Set OPENAI_API_KEY to run integration tests.")
@pytest.mark.integration
class TestPipelineIntegrationAsync:
    @pytest.mark.asyncio
    async def test_ragas_evaluator_in_rag_pipeline(self) -> None:
        dataset = [
            "Meta AI is best known for its LLaMA series, which has been made open-source "
            "for researchers and developers.",
            "LLaMA models are praised for their ability to support innovation and "
            "experimentation due to their accessibility and strong performance.",
            "Meta AI with its LLaMA models aims to democratize AI development by making "
            "high-quality models available for free.",
        ]

        document_store = InMemoryDocumentStore()
        docs = [Document(content=text) for text in dataset]
        document_embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
        document_store.write_documents(document_embedder.run(docs)["documents"])

        ragas_evaluator = RagasEvaluator(
            ragas_metrics=[Faithfulness(llm=llm_factory("gpt-4o-mini", client=AsyncOpenAI()))]
        )

        template = [
            ChatMessage.from_user(
                "Answer the question based on the context.\n\n"
                "Context:\n{% for document in documents %}{{ document.content }}\n{% endfor %}\n\n"
                "Question: {{question}}\nAnswer:"
            )
        ]

        pipeline = AsyncPipeline()
        pipeline.add_component("text_embedder", OpenAITextEmbedder(model="text-embedding-3-small"))
        pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store, top_k=2))
        pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template, required_variables="*"))
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
        pipeline.add_component("answer_builder", AnswerBuilder())
        pipeline.add_component("ragas_evaluator", ragas_evaluator)

        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever", "prompt_builder")
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        pipeline.connect("llm.replies", "answer_builder.replies")
        pipeline.connect("retriever", "answer_builder.documents")
        pipeline.connect("retriever", "ragas_evaluator.documents")
        pipeline.connect("llm.replies", "ragas_evaluator.response")

        question = "What makes Meta AI's LLaMA models stand out?"
        result = await pipeline.run_async(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
                "ragas_evaluator": {"query": question},
            }
        )

        assert "ragas_evaluator" in result
        faithfulness_result = result["ragas_evaluator"]["result"]["faithfulness"]
        assert isinstance(faithfulness_result, MetricResult)
        assert 0.0 <= faithfulness_result.value <= 1.0
