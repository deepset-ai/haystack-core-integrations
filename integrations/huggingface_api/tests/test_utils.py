# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from haystack_integrations.common.huggingface_api.utils import (
    HFEmbeddingAPIType,
    HFGenerationAPIType,
    HFModelType,
    _check_valid_model,
    _check_valid_model_async,
)


class TestAPITypes:
    def test_from_str(self):
        assert HFGenerationAPIType.from_str("serverless_inference_api") == HFGenerationAPIType.SERVERLESS_INFERENCE_API
        assert HFEmbeddingAPIType.from_str("text_embeddings_inference") == HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE
        assert str(HFGenerationAPIType.TEXT_GENERATION_INFERENCE) == "text_generation_inference"
        assert str(HFEmbeddingAPIType.INFERENCE_ENDPOINTS) == "inference_endpoints"

    def test_from_str_unknown_type(self):
        with pytest.raises(ValueError):
            HFGenerationAPIType.from_str("unknown_api_type")
        with pytest.raises(ValueError):
            HFEmbeddingAPIType.from_str("unknown_api_type")


class TestCheckValidModel:
    @patch("haystack_integrations.common.huggingface_api.utils.HfApi")
    def test_valid_model(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = MagicMock(pipeline_tag="feature-extraction")
        _check_valid_model("BAAI/bge-small-en-v1.5", HFModelType.EMBEDDING, token=None)
        mock_hf_api.return_value.model_info.assert_called_once_with("BAAI/bge-small-en-v1.5", token=None)

    @patch("haystack_integrations.common.huggingface_api.utils.HfApi")
    def test_wrong_model_type(self, mock_hf_api):
        mock_hf_api.return_value.model_info.return_value = MagicMock(pipeline_tag="text-generation")
        with pytest.raises(ValueError, match="not a embedding model"):
            _check_valid_model("microsoft/phi-2", HFModelType.EMBEDDING, token=None)

    @patch("haystack_integrations.common.huggingface_api.utils.HfApi")
    def test_model_not_found(self, mock_hf_api):
        mock_hf_api.return_value.model_info.side_effect = RepositoryNotFoundError("not found", response=MagicMock())
        with pytest.raises(ValueError, match="not found on HuggingFace Hub"):
            _check_valid_model("invalid/model-id", HFModelType.GENERATION, token=None)


class TestCheckValidModelAsync:
    @pytest.mark.asyncio
    @patch("haystack_integrations.common.huggingface_api.utils.hf_raise_for_status")
    @patch("haystack_integrations.common.huggingface_api.utils.get_async_session")
    async def test_valid_model(self, mock_get_async_session, mock_raise_for_status):
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        response = MagicMock()
        response.json.return_value = {"pipeline_tag": "feature-extraction"}
        client.get = AsyncMock(return_value=response)
        mock_get_async_session.return_value = client

        await _check_valid_model_async("BAAI/bge-small-en-v1.5", HFModelType.EMBEDDING, token=None)

        client.get.assert_awaited_once()
        mock_raise_for_status.assert_called_once_with(response)

    @pytest.mark.asyncio
    @patch("haystack_integrations.common.huggingface_api.utils.hf_raise_for_status")
    @patch("haystack_integrations.common.huggingface_api.utils.get_async_session")
    async def test_model_not_found(self, mock_get_async_session, mock_raise_for_status):
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get = AsyncMock(return_value=MagicMock())
        mock_get_async_session.return_value = client
        mock_raise_for_status.side_effect = RepositoryNotFoundError("not found", response=MagicMock())

        with pytest.raises(ValueError, match="not found on HuggingFace Hub"):
            await _check_valid_model_async("invalid/model-id", HFModelType.GENERATION, token=None)
