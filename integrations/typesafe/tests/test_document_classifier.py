# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document, Pipeline
from haystack.utils import Secret
from typesafe_sdk import SystemOneResponse

from haystack_integrations.components.classifiers.typesafe import TypeSafeDocumentClassifier

COMPONENT_TYPE = "haystack_integrations.components.classifiers.typesafe.document_classifier.TypeSafeDocumentClassifier"

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this ticket?",
        "criteria": {"billing": None, "technical": None},
    },
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["can wait", "today"]},
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
}


def _response_for(text: str, _questions: dict) -> SystemOneResponse:
    return _response("billing" if "charged" in text else "technical")


def _response(choice: str) -> SystemOneResponse:
    # The SDK decodes responses from JSON, which is where score level keys become integers
    return SystemOneResponse.model_validate_json(
        json.dumps(
            {
                "model": "laya:en",
                "answers": {
                    "department": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": 0.9,
                        "probabilities": {"billing": 0.95, "technical": 0.05},
                    },
                    "urgency": {
                        "type": "score",
                        "score": 0.7,
                        "confidence": 0.6,
                        "legend": {"0": "can wait", "1": "today"},
                        "probabilities": {"0": 0.3, "1": 0.7},
                    },
                    "refund": {"type": "noul", "noul": 0.8},
                },
                "usage": {"input_tokens": 10, "output_tokens": 0},
            }
        )
    )


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")


@pytest.mark.usefixtures("api_key")
class TestInit:
    def test_init_default(self):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS)
        assert component.questions == QUESTIONS
        assert component.model == "jev-latest"
        assert component.api_key == Secret.from_env_var("TYPESAFE_API_KEY")
        assert component.api_base_url is None
        assert component.metadata_field == "typesafe"
        assert component._client is None
        assert component._async_client is None

    @pytest.mark.parametrize(
        ("questions", "match"),
        [
            ({}, "at least one question"),
            ({"q": {"type": "yesno", "instructions": "?"}}, "has type 'yesno'"),
            ({"q": {"type": "choice", "instructions": "?"}}, "needs `criteria`"),
            ({"q": {"type": "score", "instructions": "?", "criteria": []}}, "needs `criteria`"),
        ],
    )
    def test_init_invalid_questions(self, questions, match):
        with pytest.raises(ValueError, match=match):
            TypeSafeDocumentClassifier(questions=questions)


@pytest.mark.usefixtures("api_key")
class TestSerialization:
    def test_to_dict_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("OLLAYA_KEY", "local")
        component = TypeSafeDocumentClassifier(
            questions=QUESTIONS,
            model="laya",
            api_key=Secret.from_env_var("OLLAYA_KEY"),
            api_base_url="http://localhost:11435",
            classification_field="summary",
            metadata_field="decisions",
            timeout=30.0,
            max_retries=5,
            max_workers=8,
        )
        data = component.to_dict()
        assert data == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "questions": QUESTIONS,
                "model": "laya",
                "api_key": {"env_vars": ["OLLAYA_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": "http://localhost:11435",
                "classification_field": "summary",
                "metadata_field": "decisions",
                "timeout": 30.0,
                "max_retries": 5,
                "max_workers": 8,
            },
        }
        restored = TypeSafeDocumentClassifier.from_dict(data)
        assert restored.questions == QUESTIONS
        assert restored.model == "laya"
        assert restored.api_key.resolve_value() == "local"
        assert restored.api_base_url == "http://localhost:11435"
        assert restored.classification_field == "summary"
        assert restored.metadata_field == "decisions"
        assert restored.timeout == 30.0
        assert restored.max_retries == 5
        assert restored.max_workers == 8

    def test_pipeline_round_trip(self):
        pipeline = Pipeline()
        pipeline.add_component("classifier", TypeSafeDocumentClassifier(questions=QUESTIONS))
        restored = Pipeline.loads(pipeline.dumps())
        assert restored.get_component("classifier").to_dict() == pipeline.get_component("classifier").to_dict()


@pytest.mark.usefixtures("api_key")
class TestWarmUp:
    @patch("haystack_integrations.components.classifiers.typesafe.document_classifier.TypeSafeClient")
    def test_warm_up(self, mock_client):
        component = TypeSafeDocumentClassifier(
            questions=QUESTIONS, model="laya", api_base_url="http://localhost:11435", timeout=30.0, max_retries=5
        )
        component.warm_up()
        component.warm_up()
        mock_client.assert_called_once()
        kwargs = mock_client.call_args.kwargs
        assert kwargs["api_key"] == "test-key"
        assert kwargs["base_url"] == "http://localhost:11435"
        assert kwargs["model"] == "laya"
        assert kwargs["timeout"] == 30.0
        assert kwargs["retry"].max_retries == 5
        assert component._client is mock_client.return_value

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.classifiers.typesafe.document_classifier.AsyncTypeSafeClient")
    async def test_warm_up_async(self, mock_client):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS)
        await component.warm_up_async()
        await component.warm_up_async()
        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["retry"] is None
        assert component._async_client is mock_client.return_value

    def test_close(self):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS)
        client = MagicMock()
        component._client = client
        component.close()
        component.close()
        client.close.assert_called_once()
        assert component._client is None

    @pytest.mark.asyncio
    async def test_close_async(self):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS)
        client = MagicMock(aclose=AsyncMock())
        component._async_client = client
        await component.close_async()
        await component.close_async()
        client.aclose.assert_awaited_once()
        assert component._async_client is None


@pytest.mark.usefixtures("api_key")
class TestRun:
    def test_run(self):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS)
        component._client = MagicMock()
        component._client.system_one.side_effect = _response_for
        documents = [
            Document(content="I was charged twice.", meta={"source": "email"}),
            Document(content="The app crashes on startup."),
        ]
        result = component.run(documents=documents)
        assert sorted(call.args[0] for call in component._client.system_one.call_args_list) == [
            "I was charged twice.",
            "The app crashes on startup.",
        ]
        classified = result["documents"]
        assert classified[0].meta == {
            "source": "email",
            "typesafe": {
                "department": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.9,
                    "probabilities": {"billing": 0.95, "technical": 0.05},
                },
                "urgency": {
                    "type": "score",
                    "score": 0.7,
                    "confidence": 0.6,
                    "legend": {"0": "can wait", "1": "today"},
                    "probabilities": {"0": 0.3, "1": 0.7},
                },
                "refund": {"type": "noul", "noul": 0.8},
            },
        }
        assert classified[1].meta["typesafe"]["department"]["choice"] == "technical"
        # Input documents are not mutated
        assert "typesafe" not in documents[0].meta

    def test_run_classification_field_and_skipped_documents(self, caplog):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS, classification_field="summary")
        component._client = MagicMock()
        component._client.system_one.return_value = _response("billing")
        documents = [
            Document(content="ignored", meta={"summary": "Double charge"}),
            Document(content="no summary"),
        ]
        result = component.run(documents=documents)
        component._client.system_one.assert_called_once_with("Double charge", QUESTIONS)
        assert result["documents"][0].meta["typesafe"]["department"]["choice"] == "billing"
        assert result["documents"][1] is documents[1]
        assert "has no text to classify" in caplog.text

    @pytest.mark.asyncio
    async def test_run_async(self):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS)
        component._async_client = MagicMock()
        component._async_client.system_one = AsyncMock(side_effect=_response_for)
        documents = [Document(content="I was charged twice."), Document(content=""), Document(content="It crashes.")]
        result = await component.run_async(documents=documents)
        assert component._async_client.system_one.await_count == 2
        classified = result["documents"]
        assert classified[0].meta["typesafe"]["department"]["choice"] == "billing"
        assert classified[1] is documents[1]
        assert classified[2].meta["typesafe"]["department"]["choice"] == "technical"

    @pytest.mark.asyncio
    async def test_run_async_limits_concurrent_requests(self):
        in_flight = 0
        max_in_flight = 0

        async def system_one(text, questions):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return _response_for(text, questions)

        component = TypeSafeDocumentClassifier(questions=QUESTIONS, max_workers=2)
        component._async_client = MagicMock(system_one=system_one)
        documents = [Document(content=f"Ticket {i}") for i in range(6)]
        result = await component.run_async(documents=documents)
        assert max_in_flight == 2
        assert all("typesafe" in document.meta for document in result["documents"])


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("TYPESAFE_BASE_URL"),
    reason="Set TYPESAFE_BASE_URL to a TypeSafe-compatible server, such as Ollaya, to run the integration tests.",
)
class TestTypeSafeDocumentClassifierInference:
    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.asyncio
    async def test_run(self, use_async):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS, model="laya:en", timeout=60.0)
        documents = [
            Document(content="I was charged twice for my subscription, please refund the second payment."),
            Document(content="The app crashes every time I open the settings page."),
        ]
        result = await component.run_async(documents=documents) if use_async else component.run(documents=documents)
        billing, technical = (document.meta["typesafe"] for document in result["documents"])
        assert billing["department"]["choice"] == "billing"
        assert technical["department"]["choice"] == "technical"
        assert set(billing["department"]["probabilities"]) == {"billing", "technical"}
        assert billing["refund"]["noul"] > technical["refund"]["noul"]
