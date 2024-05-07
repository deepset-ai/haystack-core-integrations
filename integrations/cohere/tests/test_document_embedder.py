# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack import Document
from haystack.utils import Secret
from haystack_integrations.components.embedders.cohere import CohereDocumentEmbedder

pytestmark = pytest.mark.embedders
COHERE_API_URL = "https://api.cohere.com"


class TestCohereDocumentEmbedder:
    def test_init_default(self):
        embedder = CohereDocumentEmbedder()
        assert embedder.api_key == Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"])
        assert embedder.model == "embed-english-v2.0"
        assert embedder.input_type == "search_document"
        assert embedder.api_base_url == COHERE_API_URL
        assert embedder.truncate == "END"
        assert embedder.use_async_client is False
        assert embedder.timeout == 120
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        embedder = CohereDocumentEmbedder(
            api_key=Secret.from_token("test-api-key"),
            model="embed-multilingual-v2.0",
            input_type="search_query",
            api_base_url="https://custom-api-base-url.com",
            truncate="START",
            use_async_client=True,
            timeout=60,
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator="-",
        )
        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.model == "embed-multilingual-v2.0"
        assert embedder.input_type == "search_query"
        assert embedder.api_base_url == "https://custom-api-base-url.com"
        assert embedder.truncate == "START"
        assert embedder.use_async_client is True
        assert embedder.timeout == 60
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == "-"

    def test_to_dict(self):
        embedder_component = CohereDocumentEmbedder()
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.cohere.document_embedder.CohereDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "model": "embed-english-v2.0",
                "input_type": "search_document",
                "api_base_url": COHERE_API_URL,
                "truncate": "END",
                "use_async_client": False,
                "timeout": 120,
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        embedder_component = CohereDocumentEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="embed-multilingual-v2.0",
            input_type="search_query",
            api_base_url="https://custom-api-base-url.com",
            truncate="START",
            use_async_client=True,
            timeout=60,
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["text_field"],
            embedding_separator="-",
        )
        component_dict = embedder_component.to_dict()
        assert component_dict == {
            "type": "haystack_integrations.components.embedders.cohere.document_embedder.CohereDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "embed-multilingual-v2.0",
                "input_type": "search_query",
                "api_base_url": "https://custom-api-base-url.com",
                "truncate": "START",
                "use_async_client": True,
                "timeout": 60,
                "batch_size": 64,
                "progress_bar": False,
                "meta_fields_to_embed": ["text_field"],
                "embedding_separator": "-",
            },
        }

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_run(self):
        embedder = CohereDocumentEmbedder()

        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        result = embedder.run(docs)
        docs_with_embeddings = result["documents"]

        assert isinstance(docs_with_embeddings, list)
        assert len(docs_with_embeddings) == len(docs)
        for doc in docs_with_embeddings:
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)

    def test_run_wrong_input_format(self):
        embedder = CohereDocumentEmbedder(api_key=Secret.from_token("test-api-key"))

        with pytest.raises(TypeError):
            embedder.run(documents="text")
        with pytest.raises(TypeError):
            embedder.run(documents=[1, 2, 3])

        assert embedder.run(documents=[]) == {"documents": [], "meta": {}}
