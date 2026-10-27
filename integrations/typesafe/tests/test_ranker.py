# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document, Pipeline
from haystack.core.serialization import component_from_dict
from haystack.utils import Secret
from typesafe_sdk import SystemOneResponse, TypeSafeAPIConnectionError

from haystack_integrations.components.rankers.typesafe import TypeSafeRanker

COMPONENT_TYPE = "haystack_integrations.components.rankers.typesafe.ranker.TypeSafeRanker"
BOOM = "boom"
CRITERIA = ["unrelated", "background", "answers"]


def _response(score: float, input_tokens: int = 10, model: str = "jev-test") -> SystemOneResponse:
    # The SDK decodes responses from JSON, which is where score level keys become integers
    return SystemOneResponse.model_validate_json(
        json.dumps(
            {
                "model": model,
                "answers": {
                    "relevance": {
                        "type": "score",
                        "score": score,
                        "confidence": 0.9,
                        "legend": {"0": "unrelated", "1": "background", "2": "answers"},
                        "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
                    }
                },
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            }
        )
    )


def _by_content(state: dict, questions: dict) -> SystemOneResponse:  # noqa: ARG001
    scores = {"low": 0.2, "mid": 1.0, "high": 1.8}
    return _response(scores[state["passage"]["text"]])


DOCUMENTS = [Document(content="mid"), Document(content="high"), Document(content="low")]


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")


@pytest.mark.usefixtures("api_key")
class TestInit:
    def test_init_default(self):
        ranker = TypeSafeRanker()
        assert ranker.model == "jev-latest"
        assert ranker.api_key == Secret.from_env_var("TYPESAFE_API_KEY")
        assert ranker.api_base_url is None
        assert len(ranker.criteria) == 5
        assert ranker.top_k == 10
        assert ranker.max_workers == 3
        assert ranker._client is None

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"criteria": ["only one"]}, "at least two"),
            ({"top_k": 0}, "`top_k` must be at least 1"),
            ({"max_chars": 0}, "`max_chars` must be at least 1"),
            ({"max_workers": 0}, "`max_workers` must be at least 1"),
        ],
    )
    def test_init_invalid(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TypeSafeRanker(**kwargs)


@pytest.mark.usefixtures("api_key")
class TestSerialization:
    def test_to_dict_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("OLLAYA_KEY", "local")
        ranker = TypeSafeRanker(
            model="laya",
            api_key=Secret.from_env_var("OLLAYA_KEY"),
            api_base_url="http://localhost:11435",
            instructions="Does `passage` answer `query`?",
            criteria=CRITERIA,
            meta_fields_to_include=["title"],
            max_chars=500,
            metadata_field="ranking",
            top_k=4,
            timeout=30.0,
            max_retries=5,
            max_workers=8,
            raise_on_failure=True,
        )
        data = ranker.to_dict()
        assert data == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "model": "laya",
                "api_key": {"env_vars": ["OLLAYA_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": "http://localhost:11435",
                "instructions": "Does `passage` answer `query`?",
                "criteria": CRITERIA,
                "meta_fields_to_include": ["title"],
                "max_chars": 500,
                "metadata_field": "ranking",
                "top_k": 4,
                "timeout": 30.0,
                "max_retries": 5,
                "max_workers": 8,
                "raise_on_failure": True,
            },
        }
        restored = component_from_dict(TypeSafeRanker, data, name="ranker")
        assert restored.to_dict() == data

    def test_pipeline_round_trip(self):
        pipeline = Pipeline()
        pipeline.add_component("ranker", TypeSafeRanker(criteria=CRITERIA))
        restored = Pipeline.loads(pipeline.dumps())
        assert restored.get_component("ranker").to_dict() == pipeline.get_component("ranker").to_dict()


@pytest.mark.usefixtures("api_key")
class TestWarmUp:
    @patch("haystack_integrations.components.rankers.typesafe.ranker.TypeSafeClient")
    def test_warm_up(self, mock_client):
        ranker = TypeSafeRanker(model="laya", api_base_url="http://localhost:11435", timeout=30.0, max_retries=5)
        ranker.warm_up()
        ranker.warm_up()
        mock_client.assert_called_once()
        kwargs = mock_client.call_args.kwargs
        assert kwargs["api_key"] == "test-key"
        assert kwargs["base_url"] == "http://localhost:11435"
        assert kwargs["model"] == "laya"
        assert kwargs["timeout"] == 30.0
        assert kwargs["retry"].max_retries == 5
        assert kwargs["retry"].timeout is None

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.rankers.typesafe.ranker.AsyncTypeSafeClient")
    async def test_warm_up_async(self, mock_client):
        ranker = TypeSafeRanker()
        await ranker.warm_up_async()
        await ranker.warm_up_async()
        mock_client.assert_called_once()
        assert ranker._async_client is mock_client.return_value

    def test_close(self):
        ranker = TypeSafeRanker()
        client = MagicMock()
        ranker._client = client
        ranker.close()
        ranker.close()
        client.close.assert_called_once()


@pytest.mark.usefixtures("api_key")
class TestRun:
    @patch("haystack_integrations.components.rankers.typesafe.ranker.TypeSafeClient")
    def test_run_ranks_and_reports_usage(self, mock_client):
        mock_client.return_value.system_one.side_effect = _by_content
        ranker = TypeSafeRanker(criteria=CRITERIA, top_k=2)
        result = ranker.run(query="q", documents=DOCUMENTS)

        assert [d.content for d in result["documents"]] == ["high", "mid"]
        assert [d.score for d in result["documents"]] == pytest.approx([0.9, 0.5])
        assert result["documents"][0].meta["typesafe"] == {"relevance": {"score": 1.8, "confidence": 0.9}}
        assert result["meta"] == {"model": "jev-test", "usage": {"input_tokens": 30, "output_tokens": 0}}

    @patch("haystack_integrations.components.rankers.typesafe.ranker.TypeSafeClient")
    def test_run_keeps_classifier_answers(self, mock_client):
        # TypeSafeDocumentClassifier writes its answers to the same default metadata field
        mock_client.return_value.system_one.side_effect = _by_content
        classifier_answers = {"department": {"type": "choice", "choice": "billing", "confidence": 0.95}}
        document = Document(content="high", meta={"typesafe": classifier_answers})

        result = TypeSafeRanker(criteria=CRITERIA).run(query="q", documents=[document])

        assert result["documents"][0].meta["typesafe"] == {
            **classifier_answers,
            "relevance": {"score": 1.8, "confidence": 0.9},
        }

    @patch("haystack_integrations.components.rankers.typesafe.ranker.TypeSafeClient")
    def test_run_top_k_override_and_input_untouched(self, mock_client):
        mock_client.return_value.system_one.side_effect = _by_content
        ranker = TypeSafeRanker(criteria=CRITERIA, top_k=2)
        result = ranker.run(query="q", documents=DOCUMENTS, top_k=3)
        assert len(result["documents"]) == 3
        assert DOCUMENTS[0].score is None
        assert "typesafe" not in DOCUMENTS[0].meta

    @patch("haystack_integrations.components.rankers.typesafe.ranker.TypeSafeClient")
    def test_run_sends_query_and_passage_as_state(self, mock_client):
        mock_client.return_value.system_one.return_value = _response(1.0)
        ranker = TypeSafeRanker(
            criteria=CRITERIA,
            instructions="Does `passage` answer `query`?",
            meta_fields_to_include=["title", "missing"],
            max_chars=3,
        )
        ranker.run(query="the query", documents=[Document(content="abcdef", meta={"title": "T", "other": 1})])

        kwargs = mock_client.return_value.system_one.call_args.kwargs
        assert kwargs["state"] == {"query": "the query", "passage": {"title": "T", "text": "abc"}}
        question = kwargs["questions"]["relevance"]
        assert question.instructions == "Does `passage` answer `query`?"
        assert question.criteria == CRITERIA

    @patch("haystack_integrations.components.rankers.typesafe.ranker.TypeSafeClient")
    def test_run_failed_documents_rank_last(self, mock_client):
        def system_one(state, questions):
            if state["passage"]["text"] == "mid":
                raise TypeSafeAPIConnectionError(BOOM)
            return _by_content(state, questions)

        mock_client.return_value.system_one.side_effect = system_one
        ranker = TypeSafeRanker(criteria=CRITERIA, top_k=3)
        result = ranker.run(query="q", documents=DOCUMENTS)

        assert [d.content for d in result["documents"]] == ["high", "low", "mid"]
        assert "boom" in result["documents"][2].meta["ranking_error"]
        assert result["documents"][2].score is None
        # Only the two successful requests are counted
        assert result["meta"]["usage"]["input_tokens"] == 20

    @patch("haystack_integrations.components.rankers.typesafe.ranker.TypeSafeClient")
    def test_run_raise_on_failure(self, mock_client):
        mock_client.return_value.system_one.side_effect = TypeSafeAPIConnectionError(BOOM)
        ranker = TypeSafeRanker(criteria=CRITERIA, raise_on_failure=True)
        with pytest.raises(TypeSafeAPIConnectionError):
            ranker.run(query="q", documents=DOCUMENTS)

    def test_run_no_documents(self):
        result = TypeSafeRanker(model="laya").run(query="q", documents=[])
        assert result == {"documents": [], "meta": {"model": "laya", "usage": {"input_tokens": 0, "output_tokens": 0}}}

    def test_run_invalid_top_k(self):
        with pytest.raises(ValueError, match="`top_k` must be at least 1"):
            TypeSafeRanker().run(query="q", documents=DOCUMENTS, top_k=0)

    @patch("haystack_integrations.components.rankers.typesafe.ranker.TypeSafeClient")
    def test_run_in_pipeline(self, mock_client):
        mock_client.return_value.system_one.side_effect = _by_content
        pipeline = Pipeline()
        pipeline.add_component("ranker", TypeSafeRanker(criteria=CRITERIA, top_k=1))
        result = pipeline.run({"ranker": {"query": "q", "documents": DOCUMENTS}})
        assert [d.content for d in result["ranker"]["documents"]] == ["high"]


@pytest.mark.usefixtures("api_key")
class TestRunAsync:
    @pytest.mark.asyncio
    @patch("haystack_integrations.components.rankers.typesafe.ranker.AsyncTypeSafeClient")
    async def test_run_async(self, mock_client):
        mock_client.return_value.system_one = AsyncMock(side_effect=_by_content)
        ranker = TypeSafeRanker(criteria=CRITERIA, top_k=2)
        result = await ranker.run_async(query="q", documents=DOCUMENTS)

        assert [d.content for d in result["documents"]] == ["high", "mid"]
        assert result["meta"] == {"model": "jev-test", "usage": {"input_tokens": 30, "output_tokens": 0}}

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.rankers.typesafe.ranker.AsyncTypeSafeClient")
    async def test_run_async_failed_documents_rank_last(self, mock_client):
        async def system_one(state, questions):
            if state["passage"]["text"] == "high":
                raise TypeSafeAPIConnectionError(BOOM)
            return _by_content(state, questions)

        mock_client.return_value.system_one = AsyncMock(side_effect=system_one)
        result = await TypeSafeRanker(criteria=CRITERIA, top_k=3).run_async(query="q", documents=DOCUMENTS)
        assert [d.content for d in result["documents"]] == ["mid", "low", "high"]

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.rankers.typesafe.ranker.AsyncTypeSafeClient")
    async def test_run_async_raise_on_failure(self, mock_client):
        mock_client.return_value.system_one = AsyncMock(side_effect=TypeSafeAPIConnectionError(BOOM))
        ranker = TypeSafeRanker(criteria=CRITERIA, raise_on_failure=True)
        with pytest.raises(TypeSafeAPIConnectionError):
            await ranker.run_async(query="q", documents=DOCUMENTS)

    @pytest.mark.asyncio
    async def test_run_async_no_documents(self):
        result = await TypeSafeRanker().run_async(query="q", documents=[])
        assert result["documents"] == []


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("TYPESAFE_BASE_URL"),
    reason="Set TYPESAFE_BASE_URL to a TypeSafe-compatible server, such as Ollaya, to run the integration tests.",
)
class TestTypeSafeRankerInference:
    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.asyncio
    async def test_run(self, use_async):
        ranker = TypeSafeRanker(model="laya:en", timeout=60.0)
        query = "How do I reset my password?"
        documents = [
            Document(content="Our offices are closed on public holidays."),
            Document(content="To reset your password, click 'Forgot password' on the login page."),
        ]
        result = (
            await ranker.run_async(query=query, documents=documents)
            if use_async
            else ranker.run(query=query, documents=documents)
        )
        ranked = result["documents"]
        assert ranked[0].content == documents[1].content
        assert ranked[0].score > ranked[1].score
        assert all(0 <= document.score <= 1 for document in ranked)
        assert result["meta"]["usage"]["input_tokens"] > 0
