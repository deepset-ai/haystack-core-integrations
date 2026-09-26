# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, call, patch

import pytest
import torch
from haystack import Document, Pipeline
from haystack.utils import ComponentDevice
from haystack.utils.auth import Secret

from haystack_integrations.components.rankers.sentence_transformers import SentenceTransformersDiversityRanker
from haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity import (
    DiversityRankingSimilarity,
)


def mock_encode_response(texts, **kwargs):
    if texts == ["city"]:
        return torch.tensor([[1.0, 1.0]])
    if texts == ["Eiffel Tower", "Berlin", "Bananas"]:
        return torch.tensor([[1.0, 0.0], [0.8, 0.8], [0.0, 1.0]])
    return torch.tensor([[0.0, 1.0]] * len(texts))


class TestSentenceTransformersDiversityRanker:
    def test_init(self):
        component = SentenceTransformersDiversityRanker()
        assert component.model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2"
        assert component.top_k == 10
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.similarity == DiversityRankingSimilarity.COSINE
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.query_prefix == ""
        assert component.document_prefix == ""
        assert component.query_suffix == ""
        assert component.document_suffix == ""
        assert component.meta_fields_to_embed == []
        assert component.embedding_separator == "\n"
        assert component.lambda_threshold == 0.5

    def test_init_with_custom_parameters(self):
        component = SentenceTransformersDiversityRanker(
            model="sentence-transformers/msmarco-distilbert-base-v4",
            top_k=5,
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_token("fake-api-token"),
            similarity="dot_product",
            query_prefix="query:",
            document_prefix="document:",
            query_suffix="query suffix",
            document_suffix="document suffix",
            meta_fields_to_embed=["meta_field"],
            embedding_separator="--",
            lambda_threshold=0,
        )
        assert component.model_name_or_path == "sentence-transformers/msmarco-distilbert-base-v4"
        assert component.top_k == 5
        assert component.device == ComponentDevice.from_str("cuda:0")
        assert component.similarity == DiversityRankingSimilarity.DOT_PRODUCT
        assert component.token == Secret.from_token("fake-api-token")
        assert component.query_prefix == "query:"
        assert component.document_prefix == "document:"
        assert component.query_suffix == "query suffix"
        assert component.document_suffix == "document suffix"
        assert component.meta_fields_to_embed == ["meta_field"]
        assert component.embedding_separator == "--"
        assert component.lambda_threshold == 0

    def test_to_dict(self):
        component = SentenceTransformersDiversityRanker()
        data = component.to_dict()
        assert (
            data["type"]
            == "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity.SentenceTransformersDiversityRanker"
        )
        assert data["init_parameters"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert data["init_parameters"]["top_k"] == 10
        assert data["init_parameters"]["device"] == ComponentDevice.resolve_device(None).to_dict()
        assert data["init_parameters"]["similarity"] == "cosine"
        assert data["init_parameters"]["token"] == {
            "env_vars": ["HF_API_TOKEN", "HF_TOKEN"],
            "strict": False,
            "type": "env_var",
        }
        assert data["init_parameters"]["query_prefix"] == ""
        assert data["init_parameters"]["document_prefix"] == ""
        assert data["init_parameters"]["query_suffix"] == ""
        assert data["init_parameters"]["document_suffix"] == ""
        assert data["init_parameters"]["meta_fields_to_embed"] == []
        assert data["init_parameters"]["embedding_separator"] == "\n"
        assert data["init_parameters"]["strategy"] == "greedy_diversity_order"

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity.SentenceTransformersDiversityRanker",
            "init_parameters": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "top_k": 10,
                "device": ComponentDevice.resolve_device(None).to_dict(),
                "similarity": "cosine",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "query_prefix": "",
                "document_prefix": "",
                "query_suffix": "",
                "document_suffix": "",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }
        ranker = SentenceTransformersDiversityRanker.from_dict(data)

        assert ranker.model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2"
        assert ranker.top_k == 10
        assert ranker.device == ComponentDevice.resolve_device(None)
        assert ranker.similarity == DiversityRankingSimilarity.COSINE
        assert ranker.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert ranker.query_prefix == ""
        assert ranker.document_prefix == ""
        assert ranker.query_suffix == ""
        assert ranker.document_suffix == ""
        assert ranker.meta_fields_to_embed == []
        assert ranker.embedding_separator == "\n"

    def test_from_dict_none_device(self):
        data = {
            "type": "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity.SentenceTransformersDiversityRanker",
            "init_parameters": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "top_k": 10,
                "device": None,
                "similarity": "cosine",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "query_prefix": "",
                "document_prefix": "",
                "query_suffix": "",
                "document_suffix": "",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }
        ranker = SentenceTransformersDiversityRanker.from_dict(data)

        assert ranker.model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2"
        assert ranker.top_k == 10
        assert ranker.device == ComponentDevice.resolve_device(None)
        assert ranker.similarity == DiversityRankingSimilarity.COSINE
        assert ranker.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert ranker.query_prefix == ""
        assert ranker.document_prefix == ""
        assert ranker.query_suffix == ""
        assert ranker.document_suffix == ""
        assert ranker.meta_fields_to_embed == []
        assert ranker.embedding_separator == "\n"

    def test_from_dict_no_default_parameters(self):
        data = {
            "type": "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity.SentenceTransformersDiversityRanker",
            "init_parameters": {},
        }
        ranker = SentenceTransformersDiversityRanker.from_dict(data)

        assert ranker.model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2"
        assert ranker.top_k == 10
        assert ranker.device == ComponentDevice.resolve_device(None)
        assert ranker.similarity == DiversityRankingSimilarity.COSINE
        assert ranker.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert ranker.query_prefix == ""
        assert ranker.document_prefix == ""
        assert ranker.query_suffix == ""
        assert ranker.document_suffix == ""
        assert ranker.meta_fields_to_embed == []
        assert ranker.embedding_separator == "\n"

    def test_to_dict_with_custom_parameters(self):
        component = SentenceTransformersDiversityRanker(
            model="sentence-transformers/msmarco-distilbert-base-v4",
            top_k=5,
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_env_var("ENV_VAR", strict=False),
            similarity="dot_product",
            query_prefix="query:",
            document_prefix="document:",
            query_suffix="query suffix",
            document_suffix="document suffix",
            meta_fields_to_embed=["meta_field"],
            embedding_separator="--",
        )
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity.SentenceTransformersDiversityRanker"
        )
        assert data["init_parameters"]["model"] == "sentence-transformers/msmarco-distilbert-base-v4"
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["device"] == ComponentDevice.from_str("cuda:0").to_dict()
        assert data["init_parameters"]["token"] == {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"}
        assert data["init_parameters"]["similarity"] == "dot_product"
        assert data["init_parameters"]["query_prefix"] == "query:"
        assert data["init_parameters"]["document_prefix"] == "document:"
        assert data["init_parameters"]["query_suffix"] == "query suffix"
        assert data["init_parameters"]["document_suffix"] == "document suffix"
        assert data["init_parameters"]["meta_fields_to_embed"] == ["meta_field"]
        assert data["init_parameters"]["embedding_separator"] == "--"
        assert data["init_parameters"]["strategy"] == "greedy_diversity_order"

    def test_from_dict_with_custom_init_parameters(self):
        data = {
            "type": "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity.SentenceTransformersDiversityRanker",
            "init_parameters": {
                "model": "sentence-transformers/msmarco-distilbert-base-v4",
                "top_k": 5,
                "device": ComponentDevice.from_str("cuda:0").to_dict(),
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "similarity": "dot_product",
                "query_prefix": "query:",
                "document_prefix": "document:",
                "query_suffix": "query suffix",
                "document_suffix": "document suffix",
                "meta_fields_to_embed": ["meta_field"],
                "embedding_separator": "--",
            },
        }
        ranker = SentenceTransformersDiversityRanker.from_dict(data)

        assert ranker.model_name_or_path == "sentence-transformers/msmarco-distilbert-base-v4"
        assert ranker.top_k == 5
        assert ranker.device == ComponentDevice.from_str("cuda:0")
        assert ranker.similarity == DiversityRankingSimilarity.DOT_PRODUCT
        assert ranker.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert ranker.query_prefix == "query:"
        assert ranker.document_prefix == "document:"
        assert ranker.query_suffix == "query suffix"
        assert ranker.document_suffix == "document suffix"
        assert ranker.meta_fields_to_embed == ["meta_field"]
        assert ranker.embedding_separator == "--"

    def test_run_invalid_similarity(self):
        """
        Tests that run method raises ValueError if similarity is incorrect
        """
        similarity = "incorrect"
        with pytest.raises(ValueError, match="Unknown similarity metric"):
            SentenceTransformersDiversityRanker(model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity)

    def test_run_invalid_strategy(self):
        """
        Tests that run method raises ValueError if strategy is incorrect
        """
        strategy = "incorrect"
        with pytest.raises(ValueError, match="Unknown strategy"):
            SentenceTransformersDiversityRanker(
                model="sentence-transformers/all-MiniLM-L6-v2", similarity="cosine", strategy=strategy
            )

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_warm_up(self, similarity, del_hf_env_vars_if_empty):
        """
        Test that ranker loads the SentenceTransformer model correctly during warm up.
        """
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_class.return_value = mock_model_instance

        with patch(
            "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity.SentenceTransformer",
            new=mock_model_class,
        ):
            ranker = SentenceTransformersDiversityRanker(model="mock_model_name", similarity=similarity, token=None)

            assert ranker.model is None

            ranker.warm_up()

            mock_model_class.assert_called_once_with(
                model_name_or_path="mock_model_name",
                device=ComponentDevice.resolve_device(None).to_torch_str(),
                token=None,
                model_kwargs=None,
                processor_kwargs=None,
                config_kwargs=None,
                backend="torch",
            )
            assert ranker.model == mock_model_instance

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_empty_query(self, similarity):
        """
        Test that ranker can be run with an empty query.
        """
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2", top_k=3, similarity=similarity
        )
        ranker.model = MagicMock()
        ranker.model.encode = MagicMock(side_effect=mock_encode_response)
        documents = [Document(content="doc1"), Document(content="doc2")]

        result = ranker.run(query="", documents=documents)
        ranked_docs = result["documents"]

        assert isinstance(ranked_docs, list)
        assert len(ranked_docs) == 2
        assert all(isinstance(doc, Document) for doc in ranked_docs)

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_top_k(self, similarity):
        """
        Test that run method returns the correct number of documents for different top_k values passed at
        initialization and runtime.
        """
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity, top_k=3
        )
        ranker.model = MagicMock()
        ranker.model.encode = MagicMock(side_effect=mock_encode_response)
        query = "test query"
        documents = [
            Document(content="doc1"),
            Document(content="doc2"),
            Document(content="doc3"),
            Document(content="doc4"),
        ]

        result = ranker.run(query=query, documents=documents)
        ranked_docs = result["documents"]

        assert isinstance(ranked_docs, list)
        assert len(ranked_docs) == 3
        assert all(isinstance(doc, Document) for doc in ranked_docs)

        # Passing a different top_k at runtime
        result = ranker.run(query=query, documents=documents, top_k=2)
        ranked_docs = result["documents"]

        assert isinstance(ranked_docs, list)
        assert len(ranked_docs) == 2
        assert all(isinstance(doc, Document) for doc in ranked_docs)

    def test_run_deduplicates_documents(self):
        ranker = SentenceTransformersDiversityRanker()
        ranker.model = MagicMock()
        ranker.model.encode = MagicMock(side_effect=mock_encode_response)
        documents = [
            Document(id="duplicate", content="keep me", score=0.9),
            Document(id="duplicate", content="drop me", score=0.1),
            Document(id="unique", content="unique"),
        ]

        result = ranker.run(query="test", documents=documents)
        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "keep me"
        assert result["documents"][1].content == "unique"

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_negative_top_k_at_init(self, similarity):
        """
        Tests that run method raises an error for negative top-k set at init.
        """
        with pytest.raises(ValueError, match="top_k must be > 0, but got"):
            SentenceTransformersDiversityRanker(
                model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity, top_k=-5
            )

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_top_k_is_none_at_init(self, similarity):
        """
        Tests that run method raises an error for top-k set to None at init.
        """
        with pytest.raises(ValueError, match="top_k must be > 0, but got"):
            SentenceTransformersDiversityRanker(
                model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity, top_k=None
            )

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_negative_top_k(self, similarity):
        """
        Tests that run method raises an error for negative top-k set at runtime.
        """
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity, top_k=10
        )
        ranker.model = MagicMock()
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]

        with pytest.raises(ValueError, match="top_k must be > 0"):
            ranker.run(query=query, documents=documents, top_k=-5)

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_top_k_is_none(self, similarity):
        """
        Tests that run method returns the correct order of documents for top-k set to None.
        """
        # Setting top_k to None is ignored during runtime, it should use top_k set at init.
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity, top_k=2
        )
        ranker.model = MagicMock()
        ranker.model.encode = MagicMock(side_effect=mock_encode_response)
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
        result = ranker.run(query=query, documents=documents, top_k=None)

        assert len(result["documents"]) == 2

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_no_documents_provided(self, similarity):
        """
        Test that run method returns an empty list if no documents are supplied.
        """
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity
        )
        ranker.model = MagicMock()
        query = "test query"
        documents = []
        results = ranker.run(query=query, documents=documents)

        assert len(results["documents"]) == 0

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_with_less_documents_than_top_k(self, similarity):
        """
        Tests that run method returns the correct number of documents for top_k values greater than number of documents.
        """
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity, top_k=5
        )
        ranker.model = MagicMock()
        ranker.model.encode = MagicMock(side_effect=mock_encode_response)
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
        result = ranker.run(query=query, documents=documents)

        assert len(result["documents"]) == 3

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_single_document_corner_case(self, similarity):
        """
        Tests that run method returns the correct number of documents for a single document
        """
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity
        )
        ranker.model = MagicMock()
        ranker.model.encode = MagicMock(side_effect=mock_encode_response)
        query = "test"
        documents = [Document(content="doc1")]
        result = ranker.run(query=query, documents=documents)

        assert len(result["documents"]) == 1

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_prepare_texts_to_embed(self, similarity):
        """
        Test creation of texts to embed from documents with meta fields, document prefix and suffix.
        """
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2",
            similarity=similarity,
            document_prefix="test doc: ",
            document_suffix=" end doc.",
            meta_fields_to_embed=["meta_field"],
            embedding_separator="\n",
        )
        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]
        texts = ranker._prepare_texts_to_embed(documents=documents)

        assert texts == [
            "test doc: meta_value 0\ndocument number 0 end doc.",
            "test doc: meta_value 1\ndocument number 1 end doc.",
            "test doc: meta_value 2\ndocument number 2 end doc.",
            "test doc: meta_value 3\ndocument number 3 end doc.",
            "test doc: meta_value 4\ndocument number 4 end doc.",
        ]

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_prepare_texts_to_embed_preserves_falsy_meta_values(self, similarity):
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2",
            similarity=similarity,
            document_prefix="test doc: ",
            document_suffix=" end doc.",
            meta_fields_to_embed=["rating", "is_available", "missing", "none_value"],
            embedding_separator="\n",
        )
        documents = [Document(content="document", meta={"rating": 0, "is_available": False, "none_value": None})]

        texts = ranker._prepare_texts_to_embed(documents=documents)

        assert texts == ["test doc: 0\nFalse\ndocument end doc."]

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_encode_text(self, similarity):
        """
        Test addition of suffix and prefix to the query and documents when creating embeddings.
        """
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2",
            similarity=similarity,
            query_prefix="test query: ",
            query_suffix=" end query.",
            document_prefix="test doc: ",
            document_suffix=" end doc.",
            meta_fields_to_embed=["meta_field"],
            embedding_separator="\n",
        )
        query = "query"
        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]
        ranker.model = MagicMock()
        ranker.model.encode = MagicMock(side_effect=mock_encode_response)
        ranker.run(query=query, documents=documents)

        assert ranker.model.encode.call_count == 2
        ranker.model.assert_has_calls(
            [
                call.encode(
                    [
                        "test doc: meta_value 0\ndocument number 0 end doc.",
                        "test doc: meta_value 1\ndocument number 1 end doc.",
                        "test doc: meta_value 2\ndocument number 2 end doc.",
                        "test doc: meta_value 3\ndocument number 3 end doc.",
                        "test doc: meta_value 4\ndocument number 4 end doc.",
                    ],
                    convert_to_tensor=True,
                ),
                call.encode(["test query: query end query."], convert_to_tensor=True),
            ]
        )

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_greedy_diversity_order(self, similarity):
        """
        Tests that the given list of documents is ordered to maximize diversity.
        """
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity
        )
        query = "city"
        documents = [Document(content="Eiffel Tower"), Document(content="Berlin"), Document(content="Bananas")]
        ranker.model = MagicMock()
        ranker.model.encode = MagicMock(side_effect=mock_encode_response)

        ranked_docs = ranker._greedy_diversity_order(query=query, documents=documents)
        ranked_text = " ".join([doc.content for doc in ranked_docs])

        assert ranked_text == "Berlin Eiffel Tower Bananas"

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_maximum_margin_relevance(self, similarity):
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity
        )
        ranker.model = MagicMock()
        ranker.model.encode = MagicMock(side_effect=mock_encode_response)

        query = "city"
        documents = [Document(content="Eiffel Tower"), Document(content="Berlin"), Document(content="Bananas")]
        ranker.model = MagicMock()
        ranker.model.encode = MagicMock(side_effect=mock_encode_response)

        ranked_docs = ranker._maximum_margin_relevance(query=query, documents=documents, lambda_threshold=0, top_k=3)
        ranked_text = " ".join([doc.content for doc in ranked_docs])

        assert ranked_text == "Berlin Eiffel Tower Bananas"

    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_maximum_margin_relevance_with_given_lambda_threshold(self, similarity):
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2", similarity=similarity
        )
        ranker.model = MagicMock()
        ranker.model.encode = MagicMock(side_effect=mock_encode_response)

        query = "city"
        documents = [Document(content="Eiffel Tower"), Document(content="Berlin"), Document(content="Bananas")]
        ranker.model = MagicMock()
        ranker.model.encode = MagicMock(side_effect=mock_encode_response)

        ranked_docs = ranker._maximum_margin_relevance(query=query, documents=documents, lambda_threshold=1, top_k=3)
        ranked_text = " ".join([doc.content for doc in ranked_docs])

        assert ranked_text == "Berlin Eiffel Tower Bananas"

    def test_pipeline_serialise_deserialise(self):
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2", similarity="cosine", top_k=5
        )

        pipe = Pipeline()
        pipe.add_component("ranker", ranker)
        pipe_serialized = pipe.dumps()
        assert Pipeline.loads(pipe_serialized) == pipe

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_real_world_use_case(self, similarity, del_hf_env_vars_if_empty):
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers-testing/stsb-bert-tiny-safetensors",
            similarity=similarity,
            device=ComponentDevice.from_str("cpu"),
        )
        query = "What are the reasons for long-standing animosities between Russia and Poland?"

        doc1 = Document(
            "One of the earliest known events in Russian-Polish history dates back to 981, when the Grand Prince of "
            "Kiev , Vladimir Svyatoslavich , seized the Cherven Cities from the Duchy of Poland . The relationship "
            "between two by that time was mostly close and cordial, as there had been no serious wars between both. "
            "In 966, Poland accepted Christianity from Rome while Kievan Rus' —the ancestor of Russia, Ukraine and "
            "Belarus—was Christianized by Constantinople. In 1054, the internal Christian divide formally split the "
            "Church into the Catholic and Orthodox branches separating the Poles from the Eastern Slavs."
        )
        doc2 = Document(
            "Since the fall of the Soviet Union , with Lithuania , Ukraine and Belarus regaining independence, the "
            "Polish Russian border has mostly been replaced by borders with the respective countries, but there still "
            "is a 210 km long border between Poland and the Kaliningrad Oblast"
        )
        doc3 = Document(
            "As part of Poland's plans to become fully energy independent from Russia within the next years, Piotr "
            "Wozniak, president of state-controlled oil and gas company PGNiG , stated in February 2019: 'The strategy "
            "of the company is just to forget about Eastern suppliers and especially about Gazprom .'[53] In 2020, the "
            "Stockholm Arbitrary Tribunal ruled that PGNiG's long-term contract gas price with Gazprom linked to oil "
            "prices should be changed to approximate the Western European gas market price, backdated to 1 November "
            "2014 when PGNiG requested a price review under the contract. Gazprom had to refund about $1.5 billion to "
            "PGNiG."
        )
        doc4 = Document(
            "Both Poland and Russia had accused each other for their historical revisionism . Russia has repeatedly "
            "accused Poland for not honoring Soviet Red Army soldiers fallen in World War II for Poland, notably in "
            "2017, in which Poland was thought on 'attempting to impose its own version of history' after Moscow was "
            "not allowed to join an international effort to renovate a World War II museum at Sobibór , site of a "
            "notorious Sobibor extermination camp."
        )
        doc5 = Document(
            "President of Russia Vladimir Putin and Prime Minister of Poland Leszek Miller in 2002 Modern Polish "
            "Russian relations begin with the fall of communism in1989 in Poland ( Solidarity and the Polish Round "
            "Table Agreement ) and 1991 in Russia ( dissolution of the Soviet Union ). With a new democratic "
            "government after the 1989 elections , Poland regained full sovereignty, [2] and what was the Soviet "
            "Union, became 15 newly independent states , including the Russian Federation . Relations between modern "
            "Poland and Russia suffer from constant ups and downs."
        )
        doc6 = Document(
            "Soviet influence in Poland finally ended with the Round Table Agreement of 1989 guaranteeing free "
            "elections in Poland, the Revolutions of 1989 against Soviet-sponsored Communist governments in the "
            "Eastern Block , and finally the formal dissolution of the Warsaw Pact."
        )
        doc7 = Document(
            "Dmitry Medvedev and then Polish Prime Minister Donald Tusk , 6 December 2010 BBC News reported that one "
            "of the main effects of the 2010 Polish Air Force Tu-154 crash would be the impact it has on "
            "Russian-Polish relations. [38] It was thought if the inquiry into the crash were not transparent, it "
            "would increase suspicions toward Russia in Poland."
        )
        doc8 = Document(
            "Soviet control over the Polish People's Republic lessened after Stalin's death and Gomulka's Thaw , and "
            "ceased completely after the fall of the communist government in Poland in late 1989, although the "
            "Soviet-Russian Northern Group of Forces did not leave Polish soil until 1993. The continuing Soviet "
            "military presence allowed the Soviet Union to heavily influence Polish politics."
        )

        documents = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8]
        result = ranker.run(query=query, documents=documents)
        expected_order = [doc5, doc7, doc3, doc1, doc4, doc2, doc6, doc8]
        expected_content = " ".join([doc.content or "" for doc in expected_order])
        result_content = " ".join([doc.content or "" for doc in result["documents"]])

        # Check the order of ranked documents by comparing the content of the ranked documents
        assert result_content == expected_content

    @pytest.mark.integration
    @pytest.mark.parametrize("similarity", ["dot_product", "cosine"])
    def test_run_with_maximum_margin_relevance_strategy(self, similarity, del_hf_env_vars_if_empty):
        query = "renewable energy sources"
        docs = [
            Document(content="18th-century French literature"),
            Document(content="Solar power generation"),
            Document(content="Ancient Egyptian hieroglyphics"),
            Document(content="Wind turbine technology"),
            Document(content="Baking sourdough bread"),
            Document(content="Hydroelectric dam systems"),
            Document(content="Geothermal energy extraction"),
            Document(content="Biomass fuel production"),
        ]

        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers-testing/stsb-bert-tiny-safetensors",
            similarity=similarity,
            strategy="maximum_margin_relevance",
            device=ComponentDevice.from_str("cpu"),
        )

        # lambda_threshold=1, the most relevant document should be returned first
        results = ranker.run(query=query, documents=docs, lambda_threshold=1, top_k=len(docs))
        expected = [
            "Geothermal energy extraction",
            "Wind turbine technology",
            "Solar power generation",
            "Biomass fuel production",
            "Hydroelectric dam systems",
            "Baking sourdough bread",
            "Ancient Egyptian hieroglyphics",
            "18th-century French literature",
        ]
        assert [doc.content for doc in results["documents"]] == expected

        # lambda_threshold=0, after the most relevant one, diverse documents should be returned
        results = ranker.run(query=query, documents=docs, lambda_threshold=0, top_k=len(docs))
        expected = [
            "Geothermal energy extraction",
            "18th-century French literature",
            "Ancient Egyptian hieroglyphics",
            "Baking sourdough bread",
            "Solar power generation",
            "Biomass fuel production",
            "Hydroelectric dam systems",
            "Wind turbine technology",
        ]
        assert [doc.content for doc in results["documents"]] == expected

    @patch(
        "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity.SentenceTransformer"
    )
    def test_model_onnx_backend(self, mocked_sentence_transformer):
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            model_kwargs={"file_name": "onnx/model.onnx"},
            backend="onnx",
        )
        ranker.warm_up()

        mocked_sentence_transformer.assert_called_once_with(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            token=None,
            model_kwargs={"file_name": "onnx/model.onnx"},
            processor_kwargs=None,
            config_kwargs=None,
            backend="onnx",
        )

    @patch(
        "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity.SentenceTransformer"
    )
    def test_model_openvino_backend(self, mocked_sentence_transformer):
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            model_kwargs={"file_name": "openvino/openvino_model.xml"},
            backend="openvino",
        )
        ranker.warm_up()

        mocked_sentence_transformer.assert_called_once_with(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            token=None,
            model_kwargs={"file_name": "openvino/openvino_model.xml"},
            processor_kwargs=None,
            config_kwargs=None,
            backend="openvino",
        )

    @patch(
        "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity.SentenceTransformer"
    )
    @pytest.mark.parametrize("model_kwargs", [{"torch_dtype": "float16"}, {"torch_dtype": "bfloat16"}])
    def test_dtype_on_gpu(self, mocked_sentence_transformer, model_kwargs):
        ranker = SentenceTransformersDiversityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cuda:0"),
            model_kwargs=model_kwargs,
        )
        ranker.warm_up()

        mocked_sentence_transformer.assert_called_once_with(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            device="cuda:0",
            token=None,
            model_kwargs=model_kwargs,
            processor_kwargs=None,
            config_kwargs=None,
            backend="torch",
        )


_PROCESSOR_UTIL_LOGGER = "haystack_integrations.utils.sentence_transformers"
_PROCESSOR_DEPRECATED_MESSAGE = "`tokenizer_kwargs` is deprecated. Use `processor_kwargs` instead."
_PROCESSOR_CONFLICT_SUFFIX = " Both were provided; `processor_kwargs` takes precedence."
_PROCESSOR_TYPE_NAME = "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity.SentenceTransformersDiversityRanker"
_PROCESSOR_PATCH_TARGET = (
    "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_diversity.SentenceTransformer"
)


def _processor_util_warnings(caplog):
    return [
        record for record in caplog.records if record.name == _PROCESSOR_UTIL_LOGGER and record.levelname == "WARNING"
    ]


class TestSentenceTransformersDiversityRankerProcessorKwargs:
    @pytest.mark.parametrize(
        ("legacy", "canonical", "expect_warn", "expect_conflict"),
        [
            pytest.param(None, None, False, False, id="both-none"),
            pytest.param({"model_max_length": 512}, None, True, False, id="legacy-only"),
            pytest.param(None, {"model_max_length": 128}, False, False, id="canonical-only"),
            pytest.param({}, None, True, False, id="legacy-empty"),
            pytest.param(None, {}, False, False, id="canonical-empty"),
            pytest.param({"model_max_length": 512}, {}, True, True, id="legacy-plus-canonical-empty"),
            pytest.param({"model_max_length": 512}, {"model_max_length": 128}, True, True, id="both-nonempty"),
            pytest.param({"model_max_length": 512}, None, True, False, id="legacy-plus-explicit-canonical-none"),
        ],
    )
    def test_constructor_matrix(self, caplog, legacy, canonical, expect_warn, expect_conflict):
        legacy_snapshot = None if legacy is None else dict(legacy)
        canonical_snapshot = None if canonical is None else dict(canonical)
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersDiversityRanker(
                model="model",
                token=None,
                device=ComponentDevice.from_str("cpu"),
                tokenizer_kwargs=legacy,
                processor_kwargs=canonical,
            )
        expected = canonical if canonical is not None else legacy
        assert component.tokenizer_kwargs == expected
        if expected is None:
            assert component.tokenizer_kwargs is None
        elif canonical is not None:
            assert component.tokenizer_kwargs is canonical
        else:
            assert component.tokenizer_kwargs is legacy
        assert not hasattr(component, "processor_kwargs")
        assert legacy == legacy_snapshot
        assert canonical == canonical_snapshot
        records = _processor_util_warnings(caplog)
        if expect_warn:
            assert len(records) == 1
            expected_message = _PROCESSOR_DEPRECATED_MESSAGE
            if expect_conflict:
                expected_message += _PROCESSOR_CONFLICT_SUFFIX
            assert records[0].getMessage() == expected_message
        else:
            assert records == []

    def test_to_dict_emits_legacy_key_only(self):
        component = SentenceTransformersDiversityRanker(model="model", processor_kwargs={"model_max_length": 128})
        data = component.to_dict()
        assert data["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 128}
        assert "processor_kwargs" not in data["init_parameters"]
        legacy = SentenceTransformersDiversityRanker(model="model", tokenizer_kwargs={"model_max_length": 128})
        assert set(data["init_parameters"]) == set(legacy.to_dict()["init_parameters"])

    def test_from_dict_accepts_canonical_key_quietly(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {"model": "model", "processor_kwargs": {"model_max_length": 128}},
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersDiversityRanker.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 128}
        assert _processor_util_warnings(caplog) == []

    def test_from_dict_legacy_only_is_quiet(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {"model": "model", "tokenizer_kwargs": {"model_max_length": 512}},
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersDiversityRanker.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert _processor_util_warnings(caplog) == []

    def test_from_dict_both_keys_canonical_wins_with_warning(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {
                "model": "model",
                "tokenizer_kwargs": {"model_max_length": 512},
                "processor_kwargs": {"model_max_length": 128},
            },
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersDiversityRanker.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 128}
        records = _processor_util_warnings(caplog)
        assert len(records) == 1
        assert records[0].getMessage() == _PROCESSOR_DEPRECATED_MESSAGE + _PROCESSOR_CONFLICT_SUFFIX
        assert data["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 512}
        assert data["init_parameters"]["processor_kwargs"] == {"model_max_length": 128}
        reserialized = component.to_dict()
        assert reserialized["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 128}
        assert "processor_kwargs" not in reserialized["init_parameters"]

    def test_from_dict_explicit_canonical_none_falls_back_quietly(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {
                "model": "model",
                "tokenizer_kwargs": {"model_max_length": 512},
                "processor_kwargs": None,
            },
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersDiversityRanker.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert _processor_util_warnings(caplog) == []

    def test_from_dict_canonical_empty_overrides_legacy(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {
                "model": "model",
                "tokenizer_kwargs": {"model_max_length": 512},
                "processor_kwargs": {},
            },
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersDiversityRanker.from_dict(data)
        assert component.tokenizer_kwargs == {}
        assert len(_processor_util_warnings(caplog)) == 1

    def test_canonical_round_trip_is_quiet(self, caplog):
        component = SentenceTransformersDiversityRanker(model="model", processor_kwargs={"model_max_length": 128})
        data = component.to_dict()
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            restored = SentenceTransformersDiversityRanker.from_dict(data)
        assert restored.tokenizer_kwargs == {"model_max_length": 128}
        assert _processor_util_warnings(caplog) == []

    @pytest.mark.parametrize(
        ("legacy", "canonical", "expected"),
        [
            pytest.param(None, None, None, id="both-none"),
            pytest.param({"model_max_length": 512}, None, {"model_max_length": 512}, id="legacy-only"),
            pytest.param(None, {"model_max_length": 128}, {"model_max_length": 128}, id="canonical-only"),
            pytest.param(
                {"model_max_length": 512}, {"model_max_length": 128}, {"model_max_length": 128}, id="both-nonempty"
            ),
            pytest.param({"model_max_length": 512}, {}, {}, id="canonical-empty-wins"),
        ],
    )
    @patch(_PROCESSOR_PATCH_TARGET)
    def test_warmup_forwards_effective_as_canonical(self, mocked_sdk, legacy, canonical, expected):
        ranker = SentenceTransformersDiversityRanker(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            tokenizer_kwargs=legacy,
            processor_kwargs=canonical,
        )
        ranker.warm_up()
        mocked_sdk.assert_called_once_with(
            model_name_or_path="model",
            device="cpu",
            token=None,
            model_kwargs=None,
            processor_kwargs=expected,
            config_kwargs=None,
            backend="torch",
        )
        assert "tokenizer_kwargs" not in mocked_sdk.call_args.kwargs

    @patch(_PROCESSOR_PATCH_TARGET)
    def test_warmup_repeated_does_not_reconstruct_or_warn(self, mocked_sdk, caplog):
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            ranker = SentenceTransformersDiversityRanker(model="model", tokenizer_kwargs={"model_max_length": 512})
            ranker.warm_up()
            ranker.warm_up()
        mocked_sdk.assert_called_once()
        assert len(_processor_util_warnings(caplog)) == 1

    @patch(_PROCESSOR_PATCH_TARGET)
    def test_legacy_reassignment_before_warmup_is_effective(self, mocked_sdk):
        ranker = SentenceTransformersDiversityRanker(model="model", token=None, device=ComponentDevice.from_str("cpu"))
        ranker.tokenizer_kwargs = {"model_max_length": 128}
        ranker.warm_up()
        assert mocked_sdk.call_args.kwargs["processor_kwargs"] == {"model_max_length": 128}
