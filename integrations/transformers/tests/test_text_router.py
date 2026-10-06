# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.utils import ComponentDevice, Secret

from haystack_integrations.components.routers.transformers import TransformersTextRouter

COMPONENT_TYPE = "haystack_integrations.components.routers.transformers.text_router.TransformersTextRouter"


class TestSerialization:
    @patch("haystack_integrations.components.routers.transformers.text_router.AutoConfig.from_pretrained")
    def test_to_dict(self, mock_auto_config_from_pretrained):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection")
        router_dict = router.to_dict()
        assert router_dict == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "labels": ["en", "de"],
                "model": "papluca/xlm-roberta-base-language-detection",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "papluca/xlm-roberta-base-language-detection",
                    "device": ComponentDevice.resolve_device(None).to_hf(),
                    "task": "text-classification",
                },
            },
        }

    @patch("haystack_integrations.components.routers.transformers.text_router.AutoConfig.from_pretrained")
    def test_to_dict_with_cpu_device(self, mock_auto_config_from_pretrained):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        router = TransformersTextRouter(
            model="papluca/xlm-roberta-base-language-detection", device=ComponentDevice.from_str("cpu")
        )
        router_dict = router.to_dict()
        assert router_dict == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "labels": ["en", "de"],
                "model": "papluca/xlm-roberta-base-language-detection",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "papluca/xlm-roberta-base-language-detection",
                    "device": ComponentDevice.from_str("cpu").to_hf(),
                    "task": "text-classification",
                },
            },
        }

    @patch("haystack_integrations.components.routers.transformers.text_router.AutoConfig.from_pretrained")
    def test_from_dict(self, mock_auto_config_from_pretrained):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        data = {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "model": "papluca/xlm-roberta-base-language-detection",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "papluca/xlm-roberta-base-language-detection",
                    "device": ComponentDevice.resolve_device(None).to_hf(),
                    "task": "zero-shot-classification",
                },
            },
        }

        component = TransformersTextRouter.from_dict(data)
        assert component.labels == ["en", "de"]
        assert component.pipeline is None
        assert component.token == Secret.from_dict(
            {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        )
        assert component.huggingface_pipeline_kwargs == {
            "model": "papluca/xlm-roberta-base-language-detection",
            "device": ComponentDevice.resolve_device(None).to_hf(),
            "task": "text-classification",
        }

    @patch("haystack_integrations.components.routers.transformers.text_router.AutoConfig.from_pretrained")
    def test_from_dict_no_default_parameters(self, mock_auto_config_from_pretrained):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        data = {
            "type": COMPONENT_TYPE,
            "init_parameters": {"model": "papluca/xlm-roberta-base-language-detection"},
        }
        component = TransformersTextRouter.from_dict(data)
        assert component.labels == ["en", "de"]
        assert component.pipeline is None
        assert component.token == Secret.from_dict(
            {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        )
        assert component.huggingface_pipeline_kwargs == {
            "model": "papluca/xlm-roberta-base-language-detection",
            "device": ComponentDevice.resolve_device(None).to_hf(),
            "task": "text-classification",
        }

    @patch("haystack_integrations.components.routers.transformers.text_router.AutoConfig.from_pretrained")
    def test_from_dict_with_cpu_device(self, mock_auto_config_from_pretrained):
        mock_auto_config_from_pretrained.return_value = MagicMock(label2id={"en": 0, "de": 1})
        data = {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "model": "papluca/xlm-roberta-base-language-detection",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "papluca/xlm-roberta-base-language-detection",
                    "device": ComponentDevice.from_str("cpu").to_hf(),
                    "task": "zero-shot-classification",
                },
            },
        }

        component = TransformersTextRouter.from_dict(data)
        assert component.labels == ["en", "de"]
        assert component.pipeline is None
        assert component.token == Secret.from_dict(
            {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        )
        assert component.huggingface_pipeline_kwargs == {
            "model": "papluca/xlm-roberta-base-language-detection",
            "device": ComponentDevice.from_str("cpu").to_hf(),
            "task": "text-classification",
        }


class TestComponentLifecycle:
    def test_key_resolved_at_init_when_labels_are_inferred(self, monkeypatch):
        monkeypatch.delenv("MISSING_HF_TOKEN", raising=False)

        with pytest.raises(ValueError, match="MISSING_HF_TOKEN"):
            TransformersTextRouter(model="model", token=Secret.from_env_var("MISSING_HF_TOKEN"))

    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_HF_TOKEN", raising=False)
        router = TransformersTextRouter(model="model", labels=["label"], token=Secret.from_env_var("MISSING_HF_TOKEN"))

        with pytest.raises(ValueError, match="MISSING_HF_TOKEN"):
            router.warm_up()

    @patch("haystack_integrations.components.routers.transformers.text_router.pipeline")
    def test_warm_up_is_idempotent(self, pipeline_mock):
        pipeline_mock.return_value = MagicMock(model=MagicMock(config=MagicMock(label2id={"label": 0})))
        router = TransformersTextRouter(model="model", labels=["label"], token=None)

        router.warm_up()
        router.warm_up()

        pipeline_mock.assert_called_once()


class TestRun:
    @patch("haystack_integrations.components.routers.transformers.text_router.pipeline")
    def test_warm_up(self, hf_pipeline_mock):
        hf_pipeline_mock.return_value = MagicMock(model=MagicMock(config=MagicMock(label2id={"en": 0, "de": 1})))
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection", labels=["en", "de"])
        router.warm_up()
        assert router.pipeline is not None

    @patch("haystack_integrations.components.routers.transformers.text_router.pipeline")
    @patch.object(TransformersTextRouter, "warm_up")
    def test_run_calls_warm_up(self, warm_up_mock, hf_pipeline_mock):
        hf_pipeline_mock.return_value = [{"label": "en", "score": 0.9}]
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection", labels=["en", "de"])
        warm_up_mock.side_effect = lambda: setattr(router, "pipeline", hf_pipeline_mock)
        router.run(text="test")
        warm_up_mock.assert_called_once()

    @patch("haystack_integrations.components.routers.transformers.text_router.pipeline")
    def test_run_fails_with_non_string_input(self, hf_pipeline_mock):
        hf_pipeline_mock.return_value = MagicMock(model=MagicMock(config=MagicMock(label2id={"en": 0, "de": 1})))
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection", labels=["en", "de"])
        with pytest.raises(TypeError):
            router.run(text=["wrong_input"])

    @patch("haystack_integrations.components.routers.transformers.text_router.pipeline")
    def test_run_unit(self, hf_pipeline_mock):
        hf_pipeline_mock.return_value = [{"label": "en", "score": 0.9}]
        router = TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection", labels=["en", "de"])
        router.pipeline = hf_pipeline_mock
        out = router.run("What is the color of the sky?")
        assert router.pipeline is not None
        assert out == {"en": "What is the color of the sky?"}


class TestIntegration:
    @pytest.mark.integration
    def test_run(self, del_hf_env_vars_if_empty):
        router = TransformersTextRouter(
            model="papluca/xlm-roberta-base-language-detection", device=ComponentDevice.from_str("cpu")
        )
        out = router.run("What is the color of the sky?")
        assert set(router.labels) == {
            "ar",
            "bg",
            "de",
            "el",
            "en",
            "es",
            "fr",
            "hi",
            "it",
            "ja",
            "nl",
            "pl",
            "pt",
            "ru",
            "sw",
            "th",
            "tr",
            "ur",
            "vi",
            "zh",
        }
        assert router.pipeline is not None
        assert out == {"en": "What is the color of the sky?"}

    @pytest.mark.integration
    def test_wrong_labels(self, del_hf_env_vars_if_empty):
        router = TransformersTextRouter(
            model="papluca/xlm-roberta-base-language-detection",
            labels=["en", "de"],
            device=ComponentDevice.from_str("cpu"),
        )
        with pytest.raises(ValueError):
            router.warm_up()
