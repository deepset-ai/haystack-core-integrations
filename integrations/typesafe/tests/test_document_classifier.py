# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document, Pipeline
from haystack.core.serialization import component_from_dict
from haystack.utils import Secret
from typesafe_sdk import Choice, Noul, SystemOneResponse, TypeSafeAPIConnectionError

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


def _response_for(state: str, questions: dict) -> SystemOneResponse:  # noqa: ARG001
    return _response(choice="billing" if "charged" in state else "technical")


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
        ("kwargs", "match"),
        [
            ({"questions": {}}, "at least one question"),
            ({"questions": {"q": {"type": "yesno", "instructions": "?"}}}, "has type 'yesno'"),
            ({"questions": {"q": {"type": "choice", "instructions": "?"}}}, "needs `criteria`"),
            ({"questions": {"q": {"type": "score", "instructions": "?", "criteria": []}}}, "needs `criteria`"),
            ({"questions": QUESTIONS, "max_workers": 0}, "at least 1"),
        ],
    )
    def test_init_invalid(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TypeSafeDocumentClassifier(**kwargs)


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
            raise_on_failure=True,
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
                "raise_on_failure": True,
            },
        }
        restored = component_from_dict(TypeSafeDocumentClassifier, data, name="classifier")
        assert restored.questions == QUESTIONS
        assert restored.model == "laya"
        assert restored.api_key.resolve_value() == "local"
        assert restored.api_base_url == "http://localhost:11435"
        assert restored.classification_field == "summary"
        assert restored.metadata_field == "decisions"
        assert restored.timeout == 30.0
        assert restored.max_retries == 5
        assert restored.max_workers == 8
        assert restored.raise_on_failure is True

    def test_to_dict_with_question_objects(self):
        component = TypeSafeDocumentClassifier(
            questions={
                "department": Choice(instructions="Which department?", criteria={"billing": None, "technical": None}),
                "refund": Noul(instructions="Is this a refund request?"),
            }
        )
        assert component.to_dict()["init_parameters"]["questions"] == {
            "department": {
                "type": "choice",
                "instructions": "Which department?",
                "criteria": {"billing": None, "technical": None},
            },
            "refund": {"type": "noul", "instructions": "Is this a refund request?"},
        }

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
        # The SDK's total retry budget is disabled so a long attempt can still be retried
        assert kwargs["retry"].timeout is None
        assert component._client is mock_client.return_value

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.classifiers.typesafe.document_classifier.AsyncTypeSafeClient")
    async def test_warm_up_async(self, mock_client):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS)
        await component.warm_up_async()
        await component.warm_up_async()
        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["retry"].max_retries == 2
        assert mock_client.call_args.kwargs["retry"].timeout is None
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
        assert sorted(call.kwargs["state"] for call in component._client.system_one.call_args_list) == [
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
        assert result["failed_documents"] == []
        # Input documents are not mutated
        assert "typesafe" not in documents[0].meta

    def test_run_classification_field_and_documents_without_text(self):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS, classification_field="summary")
        component._client = MagicMock()
        component._client.system_one.return_value = _response("billing")
        documents = [
            Document(content="ignored", meta={"summary": "Double charge"}),
            Document(content="no summary"),
        ]
        result = component.run(documents=documents)
        component._client.system_one.assert_called_once_with(state="Double charge", questions=QUESTIONS)
        assert [document.meta["typesafe"]["department"]["choice"] for document in result["documents"]] == ["billing"]
        assert [document.meta for document in result["failed_documents"]] == [
            {"classification_error": "Document has no text to classify."}
        ]

    def test_run_records_failed_requests(self):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS)
        component._client = MagicMock()
        component._client.system_one.side_effect = [TypeSafeAPIConnectionError("connection refused")]
        result = component.run(documents=[Document(content="I was charged twice.")])
        assert result["documents"] == []
        assert result["failed_documents"][0].meta["classification_error"] == "connection refused"

    def test_run_clears_previous_error(self):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS)
        component._client = MagicMock()
        component._client.system_one.side_effect = _response_for
        document = Document(content="I was charged twice.", meta={"classification_error": "connection refused"})
        result = component.run(documents=[document])
        assert "classification_error" not in result["documents"][0].meta

    def test_run_raise_on_failure(self):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS, raise_on_failure=True)
        component._client = MagicMock()
        component._client.system_one.side_effect = TypeSafeAPIConnectionError("connection refused")
        with pytest.raises(TypeSafeAPIConnectionError, match="connection refused"):
            component.run(documents=[Document(content="I was charged twice.")])

    @pytest.mark.asyncio
    async def test_run_async(self):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS)
        component._async_client = MagicMock()
        component._async_client.system_one = AsyncMock(side_effect=_response_for)
        documents = [Document(content="I was charged twice."), Document(content=""), Document(content="It crashes.")]
        result = await component.run_async(documents=documents)
        assert component._async_client.system_one.await_count == 2
        choices = [document.meta["typesafe"]["department"]["choice"] for document in result["documents"]]
        assert choices == ["billing", "technical"]
        assert result["failed_documents"][0].meta == {"classification_error": "Document has no text to classify."}

    @pytest.mark.asyncio
    async def test_run_async_records_failed_requests(self):
        component = TypeSafeDocumentClassifier(questions=QUESTIONS)
        component._async_client = MagicMock()
        component._async_client.system_one = AsyncMock(
            side_effect=[_response("billing"), TypeSafeAPIConnectionError("connection refused")]
        )
        documents = [Document(content="I was charged twice."), Document(content="It crashes.")]
        result = await component.run_async(documents=documents)
        assert len(result["documents"]) == 1
        assert result["failed_documents"][0].meta["classification_error"] == "connection refused"

    @pytest.mark.asyncio
    async def test_run_async_raise_on_failure_cancels_remaining_requests(self):
        cancelled = asyncio.Event()

        async def system_one(state, questions):  # noqa: ARG001
            if state == "fails":
                msg = "connection refused"
                raise TypeSafeAPIConnectionError(msg)
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        component = TypeSafeDocumentClassifier(questions=QUESTIONS, raise_on_failure=True)
        component._async_client = MagicMock(system_one=system_one)
        with pytest.raises(TypeSafeAPIConnectionError, match="connection refused"):
            await component.run_async(documents=[Document(content="slow"), Document(content="fails")])
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_run_async_limits_concurrent_requests(self):
        in_flight = 0
        max_in_flight = 0

        async def system_one(state, questions):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return _response_for(state=state, questions=questions)

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
