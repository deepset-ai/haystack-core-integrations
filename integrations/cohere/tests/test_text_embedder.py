# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack.utils import Secret

from haystack_integrations.components.embedders.cohere import CohereTextEmbedder
from haystack_integrations.components.embedders.cohere.embedding_types import EmbeddingTypes

COHERE_API_URL = "https://api.cohere.com"


class TestCohereTextEmbedder:
    def test_init_default(self, monkeypatch):
        """
        Test default initialization parameters for CohereTextEmbedder.
        """
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder = CohereTextEmbedder()

        assert embedder.api_key == Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"])
        assert embedder.model == "embed-english-v2.0"
        assert embedder.input_type == "search_query"
        assert embedder.api_base_url == COHERE_API_URL
        assert embedder.truncate == "END"
        assert embedder.timeout == 120

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for CohereTextEmbedder.
        """
        embedder = CohereTextEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="embed-multilingual-v2.0",
            input_type="classification",
            api_base_url="https://custom-api-base-url.com",
            truncate="START",
            timeout=60,
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.model == "embed-multilingual-v2.0"
        assert embedder.input_type == "classification"
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.truncate == "START"
        assert embedder.timeout == 60
        assert embedder.embedding_type == EmbeddingTypes.FLOAT

    def test_to_dict(self, monkeypatch):
        """
        Test serialization of this component to a dictionary, using default initialization parameters.
        """
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder_component = CohereTextEmbedder()
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.cohere.text_embedder.CohereTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "model": "embed-english-v2.0",
                "input_type": "search_query",
                "api_base_url": COHERE_API_URL,
                "truncate": "END",
                "timeout": 120,
                "embedding_type": "float",
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        """
        Test serialization of this component to a dictionary, using custom initialization parameters.
        """
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        embedder_component = CohereTextEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="embed-multilingual-v2.0",
            input_type="classification",
            api_base_url="https://custom-api-base-url.com",
            truncate="START",
            timeout=60,
            embedding_type=EmbeddingTypes.INT8,
        )
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.cohere.text_embedder.CohereTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "embed-multilingual-v2.0",
                "input_type": "classification",
                "api_base_url": "https://custom-api-base-url.com",
                "truncate": "START",
                "timeout": 60,
                "embedding_type": "int8",
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        component_dict = {
            "type": "haystack_integrations.components.embedders.cohere.text_embedder.CohereTextEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "model": "embed-english-v2.0",
                "input_type": "search_query",
                "api_base_url": COHERE_API_URL,
                "truncate": "END",
                "timeout": 120,
                "embedding_type": "float",
                "use_async_client": False,  # legacy parameter
            },
        }

        embedder = CohereTextEmbedder.from_dict(component_dict)
        assert embedder.api_key == Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"])
        assert embedder.model == "embed-english-v2.0"
        assert embedder.input_type == "search_query"
        assert embedder.api_base_url == COHERE_API_URL
        assert embedder.truncate == "END"
        assert embedder.timeout == 120
        assert embedder.embedding_type == EmbeddingTypes.FLOAT
        assert not hasattr(embedder, "use_async_client")

    def test_run_wrong_input_format(self):
        """
        Test for checking incorrect input when creating embedding.
        """
        embedder = CohereTextEmbedder(api_key=Secret.from_token("test-api-key"))
        list_integers_input = ["text_snippet_1", "text_snippet_2"]

        with pytest.raises(TypeError):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        embedder = CohereTextEmbedder()
        text = "The food was delicious"
        result = embedder.run(text=text)

        assert len(result["embedding"]) == 4096
        assert all(isinstance(x, float) for x in result["embedding"])

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    async def test_run_async(self):
        embedder = CohereTextEmbedder()
        text = "The food was delicious"
        result = await embedder.run_async(text=text)

        assert len(result["embedding"]) == 4096
        assert all(isinstance(x, float) for x in result["embedding"])
