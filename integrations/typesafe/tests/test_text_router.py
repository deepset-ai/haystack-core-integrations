# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Pipeline
from haystack.core.serialization import component_from_dict
from haystack.utils import Secret
from typesafe_sdk import Choice, SystemOneResponse

from haystack_integrations.components.routers.typesafe import TypeSafeTextRouter

COMPONENT_TYPE = "haystack_integrations.components.routers.typesafe.text_router.TypeSafeTextRouter"

LABELS = {"billing": "Payments, invoices and refunds", "technical": "Bugs and errors in the product"}


def _response(choice: str, confidence: float = 0.9) -> SystemOneResponse:
    return SystemOneResponse.model_validate_json(
        json.dumps(
            {
                "model": "laya:en",
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": confidence,
                        "probabilities": {"billing": 0.5, "technical": 0.5},
                    }
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
        router = TypeSafeTextRouter(labels=["billing", "technical"])
        assert router.labels == ["billing", "technical"]
        assert router.instructions == "Which category does this text belong to?"
        assert router.model == "jev-latest"
        assert router.api_key == Secret.from_env_var("TYPESAFE_API_KEY")
        assert router.min_confidence is None
        assert router._question == Choice(
            instructions="Which category does this text belong to?", criteria={"billing": None, "technical": None}
        )
        assert set(router.__haystack_output__._sockets_dict) == {"billing", "technical"}

    def test_init_min_confidence_adds_low_confidence_output(self):
        router = TypeSafeTextRouter(labels=LABELS, min_confidence=0.6)
        assert router._question.criteria == LABELS
        assert set(router.__haystack_output__._sockets_dict) == {"billing", "technical", "low_confidence"}

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"labels": []}, "at least one label"),
            ({"labels": ["billing", "low_confidence"]}, "reserved"),
            ({"labels": ["billing"], "min_confidence": 1.5}, "between 0 and 1"),
        ],
    )
    def test_init_invalid(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TypeSafeTextRouter(**kwargs)


@pytest.mark.usefixtures("api_key")
class TestSerialization:
    def test_to_dict_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("OLLAYA_KEY", "local")
        router = TypeSafeTextRouter(
            labels=LABELS,
            instructions="Which team should handle this?",
            model="laya",
            api_key=Secret.from_env_var("OLLAYA_KEY"),
            api_base_url="http://localhost:11435",
            min_confidence=0.6,
            timeout=30.0,
            max_retries=5,
        )
        data = router.to_dict()
        assert data == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "labels": LABELS,
                "instructions": "Which team should handle this?",
                "model": "laya",
                "api_key": {"env_vars": ["OLLAYA_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": "http://localhost:11435",
                "min_confidence": 0.6,
                "timeout": 30.0,
                "max_retries": 5,
            },
        }
        restored = component_from_dict(TypeSafeTextRouter, data, name="router")
        assert restored.labels == LABELS
        assert restored.instructions == "Which team should handle this?"
        assert restored.model == "laya"
        assert restored.api_key.resolve_value() == "local"
        assert restored.api_base_url == "http://localhost:11435"
        assert restored.min_confidence == 0.6
        assert restored.timeout == 30.0
        assert restored.max_retries == 5

    def test_pipeline_round_trip(self):
        pipeline = Pipeline()
        pipeline.add_component("router", TypeSafeTextRouter(labels=LABELS, min_confidence=0.6))
        restored = Pipeline.loads(pipeline.dumps())
        assert restored.get_component("router").to_dict() == pipeline.get_component("router").to_dict()


@pytest.mark.usefixtures("api_key")
class TestWarmUp:
    @patch("haystack_integrations.components.routers.typesafe.text_router.TypeSafeClient")
    def test_warm_up(self, mock_client):
        router = TypeSafeTextRouter(labels=LABELS, model="laya", api_base_url="http://localhost:11435", max_retries=5)
        router.warm_up()
        router.warm_up()
        mock_client.assert_called_once()
        kwargs = mock_client.call_args.kwargs
        assert kwargs["api_key"] == "test-key"
        assert kwargs["base_url"] == "http://localhost:11435"
        assert kwargs["model"] == "laya"
        assert kwargs["timeout"] is None
        assert kwargs["retry"].max_retries == 5
        assert kwargs["retry"].timeout is None
        assert router._client is mock_client.return_value

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.routers.typesafe.text_router.AsyncTypeSafeClient")
    async def test_warm_up_async(self, mock_client):
        router = TypeSafeTextRouter(labels=LABELS)
        await router.warm_up_async()
        await router.warm_up_async()
        mock_client.assert_called_once()
        assert router._async_client is mock_client.return_value

    def test_close(self):
        router = TypeSafeTextRouter(labels=LABELS)
        client = MagicMock()
        router._client = client
        router.close()
        router.close()
        client.close.assert_called_once()
        assert router._client is None

    @pytest.mark.asyncio
    async def test_close_async(self):
        router = TypeSafeTextRouter(labels=LABELS)
        client = MagicMock(aclose=AsyncMock())
        router._async_client = client
        await router.close_async()
        await router.close_async()
        client.aclose.assert_awaited_once()
        assert router._async_client is None


@pytest.mark.usefixtures("api_key")
class TestRun:
    @pytest.mark.parametrize(
        ("confidence", "expected_output"),
        [(0.9, "billing"), (0.3, "low_confidence")],
    )
    def test_run(self, confidence, expected_output):
        router = TypeSafeTextRouter(labels=LABELS, instructions="Which team should handle this?", min_confidence=0.6)
        router._client = MagicMock()
        router._client.system_one.return_value = _response("billing", confidence=confidence)
        result = router.run(text="I was charged twice.")
        router._client.system_one.assert_called_once_with(
            state="I was charged twice.",
            questions={"route": Choice(instructions="Which team should handle this?", criteria=LABELS)},
        )
        assert result == {expected_output: "I was charged twice."}

    def test_run_without_min_confidence_ignores_confidence(self):
        router = TypeSafeTextRouter(labels=LABELS)
        router._client = MagicMock()
        router._client.system_one.return_value = _response("technical", confidence=0.01)
        assert router.run(text="Hello?") == {"technical": "Hello?"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("confidence", "expected_output"),
        [(0.9, "billing"), (0.3, "low_confidence")],
    )
    async def test_run_async(self, confidence, expected_output):
        router = TypeSafeTextRouter(labels=LABELS, min_confidence=0.6)
        router._async_client = MagicMock()
        router._async_client.system_one = AsyncMock(return_value=_response("billing", confidence=confidence))
        assert await router.run_async(text="I was charged twice.") == {expected_output: "I was charged twice."}

    @pytest.mark.asyncio
    async def test_run_wrong_input_type(self):
        router = TypeSafeTextRouter(labels=LABELS)
        with pytest.raises(TypeError, match="expects a str"):
            router.run(text=["not", "a", "string"])
        with pytest.raises(TypeError, match="expects a str"):
            await router.run_async(text=["not", "a", "string"])


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("TYPESAFE_BASE_URL"),
    reason="Set TYPESAFE_BASE_URL to a TypeSafe-compatible server, such as Ollaya, to run the integration tests.",
)
class TestTypeSafeTextRouterInference:
    @pytest.mark.parametrize("use_async", [False, True])
    @pytest.mark.asyncio
    async def test_run_in_pipeline(self, use_async):
        pipeline = Pipeline()
        pipeline.add_component(
            "router",
            TypeSafeTextRouter(
                labels=LABELS,
                instructions="Which team should handle this support request?",
                model="laya:en",
                timeout=60.0,
            ),
        )
        texts = {
            "billing": "I was charged twice, please refund the second payment.",
            "technical": "The app crashes every time I open the settings page.",
        }
        for label, text in texts.items():
            data = {"router": {"text": text}}
            result = await pipeline.run_async(data) if use_async else pipeline.run(data)
            assert result == {"router": {label: text}}
