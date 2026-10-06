# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch

import pytest
from haystack import Document
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.watsonx.document_embedder import WatsonxDocumentEmbedder


@pytest.fixture
def mock_embeddings():
    """Fixture for setting up common mocks"""
    with patch("haystack_integrations.components.embedders.watsonx.document_embedder.Embeddings") as embeddings:
        yield embeddings.return_value


class TestInitializationAndSerialization:
    def test_init_default(self):
        embedder = WatsonxDocumentEmbedder(project_id=Secret.from_token("fake-project-id"))

        assert embedder.embedder is None
        assert embedder.model == "ibm/slate-30m-english-rtrvr-v2"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 1000
        assert embedder.concurrency_limit == 5
        assert isinstance(embedder.project_id, Secret)
        assert embedder.project_id.resolve_value() == "fake-project-id"

    def test_init_with_parameters(self):
        embedder = WatsonxDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            api_base_url="https://custom-url.ibm.com",
            project_id=Secret.from_token("custom-project-id"),
            truncate_input_tokens=128,
            prefix="prefix ",
            suffix=" suffix",
            batch_size=500,
            concurrency_limit=3,
            timeout=30.0,
            max_retries=5,
        )

        assert embedder.embedder is None
        assert isinstance(embedder.project_id, Secret)
        assert embedder.project_id.resolve_value() == "custom-project-id"

    def test_to_dict(self):
        component = WatsonxDocumentEmbedder(project_id=Secret.from_env_var("WATSONX_PROJECT_ID"))
        data = component.to_dict()

        assert data == {
            "type": "haystack_integrations.components.embedders.watsonx.document_embedder.WatsonxDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/slate-30m-english-rtrvr-v2",
                "api_base_url": "https://us-south.ml.cloud.ibm.com",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "truncate_input_tokens": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 1000,
                "concurrency_limit": 5,
                "timeout": None,
                "max_retries": None,
                "embedding_separator": "\n",
                "meta_fields_to_embed": [],
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.embedders.watsonx.document_embedder.WatsonxDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["WATSONX_API_KEY"], "strict": True, "type": "env_var"},
                "model": "ibm/slate-125m-english-rtrvr",
                "api_base_url": "https://custom-url.ibm.com",
                "project_id": {"env_vars": ["WATSONX_PROJECT_ID"], "strict": True, "type": "env_var"},
                "prefix": "prefix ",
                "suffix": " suffix",
                "batch_size": 500,
                "concurrency_limit": 3,
            },
        }

        component = WatsonxDocumentEmbedder.from_dict(data)

        assert component.model == "ibm/slate-125m-english-rtrvr"
        assert component.api_base_url == "https://custom-url.ibm.com"
        assert isinstance(component.project_id, Secret)
        assert component.project_id == Secret.from_env_var("WATSONX_PROJECT_ID")
        assert component.prefix == "prefix "
        assert component.suffix == " suffix"
        assert component.batch_size == 500
        assert component.concurrency_limit == 3


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        embedder = WatsonxDocumentEmbedder(project_id=Secret.from_token("fake-project-id"))

        assert embedder.embedder is None
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            embedder.warm_up()

    def test_project_id_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
        embedder = WatsonxDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))

        assert embedder.embedder is None
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            embedder.warm_up()

    def test_warm_up_is_idempotent(self):
        with (
            patch("haystack_integrations.components.embedders.watsonx.document_embedder.Credentials") as credentials,
            patch("haystack_integrations.components.embedders.watsonx.document_embedder.Embeddings") as embeddings,
        ):
            embedder = WatsonxDocumentEmbedder(
                api_key=Secret.from_token("fake-api-key"),
                api_base_url="https://custom-url.ibm.com",
                project_id=Secret.from_token("fake-project-id"),
                truncate_input_tokens=128,
                batch_size=500,
                concurrency_limit=3,
                max_retries=5,
            )
            embedder.warm_up()
            client = embedder.embedder
            embedder.warm_up()

        credentials.assert_called_once_with(api_key="fake-api-key", url="https://custom-url.ibm.com")
        embeddings.assert_called_once_with(
            model_id="ibm/slate-30m-english-rtrvr-v2",
            credentials=credentials.return_value,
            project_id="fake-project-id",
            params={"truncate_input_tokens": 128},
            batch_size=500,
            concurrency_limit=3,
            max_retries=5,
        )
        assert embedder.embedder is client


class TestRun:
    def test_prepare_texts_to_embed(self):
        embedder = WatsonxDocumentEmbedder(
            project_id=Secret.from_token("fake-project-id"),
            prefix="prefix ",
            suffix=" suffix",
            meta_fields_to_embed=["source"],
        )
        prepared_text = embedder._prepare_texts_to_embed(
            [Document(content="The food was delicious", meta={"source": "test"})]
        )
        assert prepared_text == ["prefix test\nThe food was delicious suffix"]

    def test_run_wrong_input_format(self):
        embedder = WatsonxDocumentEmbedder(project_id=Secret.from_token("fake-project-id"))
        with pytest.raises(TypeError, match=r"WatsonxDocumentEmbedder expects a list of Documents as input\."):
            embedder.run(documents="not a list")  # type: ignore

    def test_run_empty_documents(self, mock_embeddings):
        embedder = WatsonxDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"), project_id=Secret.from_token("fake-project-id")
        )
        result = embedder.run(documents=[])
        assert result == {
            "documents": [],
            "meta": {"model": "ibm/slate-30m-english-rtrvr-v2", "truncate_input_tokens": None, "batch_size": 1000},
        }

    def test_run_does_not_modify_original_documents(self, mock_embeddings):
        """Test that original documents are not modified during embedding"""
        embedder = WatsonxDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"), project_id=Secret.from_token("fake-project-id")
        )
        original_docs = [
            Document(content="I love cheese"),
            Document(content="A transformer is a deep learning architecture"),
        ]

        # Mock the embedder to return embeddings
        mock_embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        result = embedder.run(documents=original_docs)

        # Check that original documents are not modified
        for doc in original_docs:
            assert doc.embedding is None

        # Check that returned documents have embeddings
        for doc_with_embedding in result["documents"]:
            assert doc_with_embedding.embedding is not None


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("WATSONX_API_KEY") or not os.environ.get("WATSONX_PROJECT_ID"),
    reason="WATSONX_API_KEY or WATSONX_PROJECT_ID not set",
)
class TestIntegration:
    """Integration tests for WatsonxDocumentEmbedder (requires real credentials)"""

    @pytest.fixture
    def test_documents(self):
        return [
            Document(content="The quick brown fox jumps over the lazy dog"),
            Document(content="Artificial intelligence is transforming industries"),
            Document(content="Haystack is an open-source framework for building search systems"),
        ]

    def test_run(self, test_documents):
        """Test real API call with documents"""
        embedder = WatsonxDocumentEmbedder(
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
            truncate_input_tokens=128,
        )
        result = embedder.run(test_documents)

        assert len(result["documents"]) == 3
        for doc in result["documents"]:
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) > 0
            assert all(isinstance(x, float) for x in doc.embedding)

        assert result["meta"]["model"] == "ibm/slate-30m-english-rtrvr-v2"

    def test_batch_processing(self, test_documents):
        """Test that batch processing works"""
        embedder = WatsonxDocumentEmbedder(
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
            batch_size=2,
            truncate_input_tokens=128,
        )

        result = embedder.run(test_documents)
        assert len(result["documents"]) == 3
        assert all(doc.embedding is not None for doc in result["documents"])

    def test_text_truncation(self):
        """Test that truncation works with long documents"""
        long_content = "This is a very long document. " * 10
        long_document = Document(content=long_content)

        embedder = WatsonxDocumentEmbedder(
            api_key=Secret.from_env_var("WATSONX_API_KEY"),
            project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
            truncate_input_tokens=4,
        )

        result = embedder.run([long_document])
        assert len(result["documents"][0].embedding) > 0
        assert result["meta"]["truncate_input_tokens"] == 4
