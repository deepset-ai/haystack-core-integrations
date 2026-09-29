# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack import Pipeline
from haystack.utils import ComponentDevice, Secret

from haystack_integrations.components.routers.laya import LayaTextRouter

COMPONENT_TYPE = "haystack_integrations.components.routers.laya.text_router.LayaTextRouter"

LABELS = {"billing": "Payments, invoices and refunds", "technical": "Bugs and errors in the product"}


def _result(choice: str, *, low_confidence: bool = False) -> dict:
    answer = {"type": "choice", "choice": choice, "probabilities": {}, "confidence": 0.9}
    if low_confidence:
        answer["low_confidence"] = True
    return {"model": "laya-rl-agent", "answers": {"route": answer}, "usage": {"input_tokens": 10, "output_tokens": 0}}


class TestInit:
    def test_init_default(self):
        router = LayaTextRouter(labels=["billing", "technical"])
        assert router.labels == ["billing", "technical"]
        assert router.instructions == "Which category does this text belong to?"
        assert router.model == "convaiinnovations/laya"
        assert router.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert router.min_confidence is None
        assert router._agent is None
        assert set(router.__haystack_output__._sockets_dict) == {"billing", "technical"}

    def test_init_min_confidence_adds_low_confidence_output(self):
        router = LayaTextRouter(labels=LABELS, min_confidence=0.6)
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
            LayaTextRouter(**kwargs)


class TestSerialization:
    def test_to_dict_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("LAYA_HF_TOKEN", "test-token")
        router = LayaTextRouter(
            labels=LABELS,
            instructions="Which team should handle this?",
            model="my-org/laya-finetune",
            subfolder="multilingual",
            revision="abc123",
            device=ComponentDevice.from_str("cpu"),
            token=Secret.from_env_var("LAYA_HF_TOKEN"),
            min_confidence=0.6,
        )

        data = router.to_dict()
        assert data == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "labels": LABELS,
                "instructions": "Which team should handle this?",
                "model": "my-org/laya-finetune",
                "subfolder": "multilingual",
                "revision": "abc123",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "token": {"env_vars": ["LAYA_HF_TOKEN"], "strict": True, "type": "env_var"},
                "min_confidence": 0.6,
            },
        }

        restored = LayaTextRouter.from_dict(data)
        assert restored.labels == LABELS
        assert restored.instructions == "Which team should handle this?"
        assert restored.model == "my-org/laya-finetune"
        assert restored.subfolder == "multilingual"
        assert restored.revision == "abc123"
        assert restored.device == ComponentDevice.from_str("cpu")
        assert restored.token.resolve_value() == "test-token"
        assert restored.min_confidence == 0.6

    def test_pipeline_round_trip(self):
        pipeline = Pipeline()
        pipeline.add_component("router", LayaTextRouter(labels=LABELS, min_confidence=0.6))
        restored = Pipeline.loads(pipeline.dumps())
        assert restored.get_component("router").to_dict() == pipeline.get_component("router").to_dict()


class TestWarmUp:
    @patch("haystack_integrations.components.routers.laya.text_router.load")
    def test_warm_up(self, mock_load, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")
        router = LayaTextRouter(
            labels=LABELS, subfolder="multilingual", revision="abc123", device=ComponentDevice.from_str("cpu")
        )
        router.warm_up()
        router.warm_up()

        mock_load.assert_called_once_with(
            "convaiinnovations/laya", device="cpu", token="test-token", subfolder="multilingual", revision="abc123"
        )
        assert router._agent is mock_load.return_value


class TestRun:
    def test_run(self):
        router = LayaTextRouter(labels=LABELS, instructions="Which team should handle this?", min_confidence=0.6)
        router._agent = MagicMock()
        router._agent.predict.return_value = _result("billing")

        result = router.run(text="I was charged twice.")

        router._agent.predict.assert_called_once_with(
            "I was charged twice.",
            {"route": {"type": "choice", "instructions": "Which team should handle this?", "criteria": LABELS}},
            min_confidence=0.6,
        )
        assert result == {"billing": "I was charged twice."}

    def test_run_low_confidence(self):
        router = LayaTextRouter(labels=LABELS, min_confidence=0.6)
        router._agent = MagicMock()
        router._agent.predict.return_value = _result("billing", low_confidence=True)

        assert router.run(text="Hello?") == {"low_confidence": "Hello?"}

    def test_run_wrong_input_type(self):
        router = LayaTextRouter(labels=LABELS)
        with pytest.raises(TypeError, match="expects a str"):
            router.run(text=["not", "a", "string"])

    @patch("haystack_integrations.components.routers.laya.text_router.load")
    def test_run_warms_up(self, mock_load):
        mock_load.return_value.predict.return_value = _result("technical")
        router = LayaTextRouter(labels=LABELS)

        router.run(text="The app crashes.")

        mock_load.assert_called_once()


@pytest.mark.integration
class TestLayaTextRouterInference:
    def test_run_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component(
            "router",
            LayaTextRouter(
                labels=LABELS,
                instructions="Which team should handle this support request?",
                device=ComponentDevice.from_str("cpu"),
            ),
        )

        billing = pipeline.run({"router": {"text": "I was charged twice, please refund the second payment."}})
        technical = pipeline.run({"router": {"text": "The app crashes every time I open the settings page."}})

        assert billing == {"router": {"billing": "I was charged twice, please refund the second payment."}}
        assert technical == {"router": {"technical": "The app crashes every time I open the settings page."}}
