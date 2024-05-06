import os
from unittest.mock import Mock, patch

import pytest
from haystack import Document
from haystack.utils import Secret
from haystack_integrations.components.embedders.nvidia import NvidiaDocumentEmbedder


class TestNvidiaDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        embedder = NvidiaDocumentEmbedder("nvolveqa_40k")

        assert embedder.api_key == Secret.from_env_var("NVIDIA_API_KEY")
        assert embedder.model == "nvolveqa_40k"
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_with_parameters(self):
        embedder = NvidiaDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model="nvolveqa_40k",
            prefix="prefix",
            suffix="suffix",
            batch_size=30,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )

        assert embedder.api_key == Secret.from_token("fake-api-key")
        assert embedder.model == "nvolveqa_40k"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 30
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        embedder = NvidiaDocumentEmbedder("nvolveqa_40k")
        with pytest.raises(ValueError):
            embedder.warm_up()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        component = NvidiaDocumentEmbedder("playground_nvolveqa_40k")
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "api_url": None,
                "model": "playground_nvolveqa_40k",
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "truncate": "NONE",
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        component = NvidiaDocumentEmbedder(
            model="playground_nvolveqa_40k",
            api_url="https://example.com",
            prefix="prefix",
            suffix="suffix",
            batch_size=10,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
            truncate="END",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.nvidia.document_embedder.NvidiaDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "api_url": "https://example.com",
                "model": "playground_nvolveqa_40k",
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 10,
                "progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
                "truncate": "END",
            },
        }

    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(content=f"document number {i}:\ncontent", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = NvidiaDocumentEmbedder(
            "playground_nvolveqa_40k",
            api_key=Secret.from_token("fake-api-key"),
            meta_fields_to_embed=["meta_field"],
            embedding_separator=" | ",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        # note that newline is replaced by space
        assert prepared_texts == [
            "meta_value 0 | document number 0:\ncontent",
            "meta_value 1 | document number 1:\ncontent",
            "meta_value 2 | document number 2:\ncontent",
            "meta_value 3 | document number 3:\ncontent",
            "meta_value 4 | document number 4:\ncontent",
        ]

    def test_prepare_texts_to_embed_w_suffix(self):
        documents = [Document(content=f"document number {i}") for i in range(5)]

        embedder = NvidiaDocumentEmbedder(
            "playground_nvolveqa_40k",
            api_key=Secret.from_token("fake-api-key"),
            prefix="my_prefix ",
            suffix=" my_suffix",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "my_prefix document number 0 my_suffix",
            "my_prefix document number 1 my_suffix",
            "my_prefix document number 2 my_suffix",
            "my_prefix document number 3 my_suffix",
            "my_prefix document number 4 my_suffix",
        ]

    @patch("haystack_integrations.components.embedders.nvidia._nvcf_backend.NvidiaCloudFunctionsClient")
    def test_embed_batch(self, mock_client_class):
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        embedder = NvidiaDocumentEmbedder(
            "playground_nvolveqa_40k",
            api_key=Secret.from_token("fake-api-key"),
        )

        def mock_query_function(_, payload):
            inputs = payload["input"]
            data = [{"index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(len(inputs))]
            return {"data": data, "usage": {"total_tokens": 4, "prompt_tokens": 4}}

        mock_client = Mock(
            get_model_nvcf_id=Mock(return_value="some_id"),
            query_function=mock_query_function,
        )
        mock_client_class.return_value = mock_client
        embedder.warm_up()

        embeddings, metadata = embedder._embed_batch(texts_to_embed=texts, batch_size=2)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 3
            assert all(isinstance(x, float) for x in embedding)

        assert metadata == {"usage": {"prompt_tokens": 3 * 4, "total_tokens": 3 * 4}}

    @patch("haystack_integrations.components.embedders.nvidia._nvcf_backend.NvidiaCloudFunctionsClient")
    def test_run(self, mock_client_class):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        model = "playground_nvolveqa_40k"
        embedder = NvidiaDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model=model,
            prefix="prefix ",
            suffix=" suffix",
            meta_fields_to_embed=["topic"],
            embedding_separator=" | ",
        )

        def mock_query_function(_, payload):
            inputs = payload["input"]
            data = [{"index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(len(inputs))]
            return {"data": data, "usage": {"total_tokens": 4, "prompt_tokens": 4}}

        mock_client = Mock(
            get_model_nvcf_id=Mock(return_value="some_id"),
            query_function=mock_query_function,
        )
        mock_client_class.return_value = mock_client
        embedder.warm_up()

        result = embedder.run(documents=docs)

        documents_with_embeddings = result["documents"]
        metadata = result["meta"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 3
            assert all(isinstance(x, float) for x in doc.embedding)
        assert metadata == {"usage": {"prompt_tokens": 4, "total_tokens": 4}}

    @patch("haystack_integrations.components.embedders.nvidia._nvcf_backend.NvidiaCloudFunctionsClient")
    def test_run_custom_batch_size(self, mock_client_class):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]
        model = "playground_nvolveqa_40k"
        embedder = NvidiaDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model=model,
            prefix="prefix ",
            suffix=" suffix",
            meta_fields_to_embed=["topic"],
            embedding_separator=" | ",
            batch_size=1,
        )

        def mock_query_function(_, payload):
            inputs = payload["input"]
            data = [{"index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(len(inputs))]
            return {"data": data, "usage": {"total_tokens": 4, "prompt_tokens": 4}}

        mock_client = Mock(
            get_model_nvcf_id=Mock(return_value="some_id"),
            query_function=mock_query_function,
        )
        mock_client_class.return_value = mock_client
        embedder.warm_up()

        result = embedder.run(documents=docs)

        documents_with_embeddings = result["documents"]
        metadata = result["meta"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 3
            assert all(isinstance(x, float) for x in doc.embedding)

        assert metadata == {"usage": {"prompt_tokens": 2 * 4, "total_tokens": 2 * 4}}

    @patch("haystack_integrations.components.embedders.nvidia._nvcf_backend.NvidiaCloudFunctionsClient")
    def test_run_wrong_input_format(self, mock_client_class):
        embedder = NvidiaDocumentEmbedder("playground_nvolveqa_40k", api_key=Secret.from_token("fake-api-key"))

        def mock_query_function(_, payload):
            inputs = payload["input"]
            data = [{"index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(len(inputs))]
            return {"data": data, "usage": {"total_tokens": 4, "prompt_tokens": 4}}

        mock_client = Mock(
            get_model_nvcf_id=Mock(return_value="some_id"),
            query_function=mock_query_function,
        )
        mock_client_class.return_value = mock_client
        embedder.warm_up()

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="NvidiaDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="NvidiaDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    @patch("haystack_integrations.components.embedders.nvidia._nvcf_backend.NvidiaCloudFunctionsClient")
    def test_run_on_empty_list(self, mock_client_class):
        embedder = NvidiaDocumentEmbedder("playground_nvolveqa_40k", api_key=Secret.from_token("fake-api-key"))

        def mock_query_function(_, payload):
            inputs = payload["input"]
            data = [{"index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(len(inputs))]
            return {"data": data, "usage": {"total_tokens": 4, "prompt_tokens": 4}}

        mock_client = Mock(
            get_model_nvcf_id=Mock(return_value="some_id"),
            query_function=mock_query_function,
        )
        mock_client_class.return_value = mock_client
        embedder.warm_up()

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the Nvidia API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        embedder = NvidiaDocumentEmbedder("playground_nvolveqa_40k")
        embedder.warm_up()

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

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_NIM_EMBEDDER_MODEL", None) or not os.environ.get("NVIDIA_NIM_ENDPOINT_URL", None),
        reason="Export an env var called NVIDIA_NIM_EMBEDDER_MODEL containing the hosted model name and "
        "NVIDIA_NIM_ENDPOINT_URL containing the local URL to call.",
    )
    @pytest.mark.integration
    def test_run_integration_with_nim_backend(self):
        model = os.environ["NVIDIA_NIM_EMBEDDER_MODEL"]
        url = os.environ["NVIDIA_NIM_ENDPOINT_URL"]
        embedder = NvidiaDocumentEmbedder(
            model=model,
            api_url=url,
            api_key=None,
        )
        embedder.warm_up()
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

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_CATALOG_API_KEY", None),
        reason="Export an env var called NVIDIA_CATALOG_API_KEY containing the Nvidia API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration_with_api_catalog(self):
        embedder = NvidiaDocumentEmbedder(
            model="NV-Embed-QA",
            api_url="https://ai.api.nvidia.com/v1/retrieval/nvidia",
            api_key=Secret.from_env_var("NVIDIA_CATALOG_API_KEY"),
        )
        embedder.warm_up()

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
