# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from cohere import COHERE_API_URL

from cohere_haystack.embedders.text_embedder import CohereTextEmbedder

pytestmark = pytest.mark.embedders


class TestCohereTextEmbedder:
    def test_init_default(self):
        """
        Test default initialization parameters for CohereTextEmbedder.
        """
        embedder = CohereTextEmbedder(api_key="test-api-key")

        assert embedder.api_key == "test-api-key"
        assert embedder.model_name == "embed-english-v2.0"
        assert embedder.input_type == "search_document"
        assert embedder.api_base_url == COHERE_API_URL
        assert embedder.truncate == "END"
        assert embedder.use_async_client is False
        assert embedder.max_retries == 3
        assert embedder.timeout == 120

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for CohereTextEmbedder.
        """
        embedder = CohereTextEmbedder(
            api_key="test-api-key",
            model_name="embed-multilingual-v2.0",
            input_type="search_query",
            api_base_url="https://custom-api-base-url.com",
            truncate="START",
            use_async_client=True,
            max_retries=5,
            timeout=60,
        )
        assert embedder.api_key == "test-api-key"
        assert embedder.model_name == "embed-multilingual-v2.0"
        assert embedder.input_type == "search_query"
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.truncate == "START"
        assert embedder.use_async_client is True
        assert embedder.max_retries == 5
        assert embedder.timeout == 60

    def test_to_dict(self):
        """
        Test serialization of this component to a dictionary, using default initialization parameters.
        """
        embedder_component = CohereTextEmbedder(api_key="test-api-key")
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "cohere_haystack.embedders.text_embedder.CohereTextEmbedder",
            "init_parameters": {
                "model_name": "embed-english-v2.0",
                "input_type": "search_document",
                "api_base_url": COHERE_API_URL,
                "truncate": "END",
                "use_async_client": False,
                "max_retries": 3,
                "timeout": 120,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        """
        Test serialization of this component to a dictionary, using custom initialization parameters.
        """
        embedder_component = CohereTextEmbedder(
            api_key="test-api-key",
            model_name="embed-multilingual-v2.0",
            input_type="search_query",
            api_base_url="https://custom-api-base-url.com",
            truncate="START",
            use_async_client=True,
            max_retries=5,
            timeout=60,
        )
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "cohere_haystack.embedders.text_embedder.CohereTextEmbedder",
            "init_parameters": {
                "model_name": "embed-multilingual-v2.0",
                "input_type": "search_query",
                "api_base_url": "https://custom-api-base-url.com",
                "truncate": "START",
                "use_async_client": True,
                "max_retries": 5,
                "timeout": 60,
            },
        }

    def test_run_wrong_input_format(self):
        """
        Test for checking incorrect input when creating embedding.
        """
        embedder = CohereTextEmbedder(api_key="test-api-key")
        list_integers_input = ["text_snippet_1", "text_snippet_2"]

        with pytest.raises(TypeError, match="CohereTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        embedder = CohereTextEmbedder()
        text = "The food was delicious"
        result = embedder.run(text)
        assert all(isinstance(x, float) for x in result["embedding"])
