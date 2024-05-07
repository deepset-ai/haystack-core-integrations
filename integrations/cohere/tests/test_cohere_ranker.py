import os
from unittest.mock import Mock, patch

import pytest
from haystack import Document
from haystack.utils.auth import Secret
from haystack_integrations.components.rankers.cohere import CohereRanker

pytestmark = pytest.mark.ranker
COHERE_API_URL = "https://api.cohere.com"


@pytest.fixture
def mock_ranker_response():
    """
    Mock the Cohere ranker API response and reuse it for tests
    The `response` is an object of <class 'cohere.responses.rerank.Reranking'>
    and `response.results` is list : [RerankResult<document['text']: "", index: 2, relevance_score: 0.98>,
                                      RerankResult<document['text']: "", index: 0, relevance_score: 0.98>,
                                      RerankResult<document['text']: "", index: 1, relevance_score: 0.04>]
    """
    with patch("cohere.Client.rerank", autospec=True) as mock_ranker_response:

        mock_response = Mock()

        mock_ranker_res_obj1 = Mock()
        mock_ranker_res_obj1.index = 2
        mock_ranker_res_obj1.relevance_score = 0.98

        mock_ranker_res_obj2 = Mock()
        mock_ranker_res_obj2.index = 1
        mock_ranker_res_obj2.relevance_score = 0.95

        mock_response.results = [mock_ranker_res_obj1, mock_ranker_res_obj2]
        mock_ranker_response.return_value = mock_response
        yield mock_ranker_response


class TestCohereRanker:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        component = CohereRanker()
        assert component.model_name == "rerank-english-v2.0"
        assert component.top_k == 10
        assert component.api_key == Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"])
        assert component.api_base_url == COHERE_API_URL
        assert component.max_chunks_per_doc is None
        assert component.meta_fields_to_embed == []
        assert component.meta_data_separator == "\n"

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("CO_API_KEY", raising=False)
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the following authentication environment variables are set: *"):
            CohereRanker()

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        component = CohereRanker(
            model="rerank-multilingual-v2.0",
            top_k=5,
            api_key=Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
            api_base_url="test-base-url",
            max_chunks_per_doc=40,
            meta_fields_to_embed=["meta_field_1", "meta_field_2"],
            meta_data_separator=",",
        )
        assert component.model_name == "rerank-multilingual-v2.0"
        assert component.top_k == 5
        assert component.api_key == Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"])
        assert component.api_base_url == "test-base-url"
        assert component.max_chunks_per_doc == 40
        assert component.meta_fields_to_embed == ["meta_field_1", "meta_field_2"]
        assert component.meta_data_separator == ","

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        component = CohereRanker()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.cohere.ranker.CohereRanker",
            "init_parameters": {
                "model": "rerank-english-v2.0",
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": COHERE_API_URL,
                "top_k": 10,
                "max_chunks_per_doc": None,
                "meta_fields_to_embed": [],
                "meta_data_separator": "\n",
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        component = CohereRanker(
            model="rerank-multilingual-v2.0",
            top_k=2,
            api_key=Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
            api_base_url="test-base-url",
            max_chunks_per_doc=50,
            meta_fields_to_embed=["meta_field_1", "meta_field_2"],
            meta_data_separator=",",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.cohere.ranker.CohereRanker",
            "init_parameters": {
                "model": "rerank-multilingual-v2.0",
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": "test-base-url",
                "top_k": 2,
                "max_chunks_per_doc": 50,
                "meta_fields_to_embed": ["meta_field_1", "meta_field_2"],
                "meta_data_separator": ",",
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        data = {
            "type": "haystack_integrations.components.rankers.cohere.ranker.CohereRanker",
            "init_parameters": {
                "model": "rerank-multilingual-v2.0",
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": "test-base-url",
                "top_k": 2,
                "max_chunks_per_doc": 50,
                "meta_fields_to_embed": ["meta_field_1", "meta_field_2"],
                "meta_data_separator": ",",
            },
        }
        component = CohereRanker.from_dict(data)
        assert component.model_name == "rerank-multilingual-v2.0"
        assert component.top_k == 2
        assert component.api_key == Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"])
        assert component.api_base_url == "test-base-url"
        assert component.max_chunks_per_doc == 50
        assert component.meta_fields_to_embed == ["meta_field_1", "meta_field_2"]
        assert component.meta_data_separator == ","

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("CO_API_KEY", raising=False)
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.rankers.cohere.ranker.CohereRanker",
            "init_parameters": {
                "model": "rerank-multilingual-v2.0",
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "top_k": 2,
                "max_chunks_per_doc": 50,
            },
        }
        with pytest.raises(ValueError, match="None of the following authentication environment variables are set: *"):
            CohereRanker.from_dict(data)

    def test_prepare_cohere_input_docs_default_separator(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        component = CohereRanker(meta_fields_to_embed=["meta_field_1", "meta_field_2"])
        documents = [
            Document(
                content=f"document number {i}",
                meta={
                    "meta_field_1": f"meta_value_1 {i}",
                    "meta_field_2": f"meta_value_2 {i+5}",
                    "meta_field_3": f"meta_value_3 {i+15}",
                },
            )
            for i in range(5)
        ]

        texts = component._prepare_cohere_input_docs(documents=documents)

        assert texts == [
            "meta_value_1 0\nmeta_value_2 5\ndocument number 0",
            "meta_value_1 1\nmeta_value_2 6\ndocument number 1",
            "meta_value_1 2\nmeta_value_2 7\ndocument number 2",
            "meta_value_1 3\nmeta_value_2 8\ndocument number 3",
            "meta_value_1 4\nmeta_value_2 9\ndocument number 4",
        ]

    def test_prepare_cohere_input_docs_custom_separator(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        component = CohereRanker(meta_fields_to_embed=["meta_field_1", "meta_field_2"], meta_data_separator=" ")
        documents = [
            Document(
                content=f"document number {i}",
                meta={
                    "meta_field_1": f"meta_value_1 {i}",
                    "meta_field_2": f"meta_value_2 {i+5}",
                    "meta_field_3": f"meta_value_3 {i+15}",
                },
            )
            for i in range(5)
        ]

        texts = component._prepare_cohere_input_docs(documents=documents)

        assert texts == [
            "meta_value_1 0 meta_value_2 5 document number 0",
            "meta_value_1 1 meta_value_2 6 document number 1",
            "meta_value_1 2 meta_value_2 7 document number 2",
            "meta_value_1 3 meta_value_2 8 document number 3",
            "meta_value_1 4 meta_value_2 9 document number 4",
        ]

    def test_prepare_cohere_input_docs_no_meta_data(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        component = CohereRanker(meta_fields_to_embed=["meta_field_1", "meta_field_2"], meta_data_separator=" ")
        documents = [Document(content=f"document number {i}") for i in range(5)]

        texts = component._prepare_cohere_input_docs(documents=documents)

        assert texts == [
            "document number 0",
            "document number 1",
            "document number 2",
            "document number 3",
            "document number 4",
        ]

    def test_prepare_cohere_input_docs_no_docs(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        component = CohereRanker(meta_fields_to_embed=["meta_field_1", "meta_field_2"], meta_data_separator=" ")
        documents = []

        texts = component._prepare_cohere_input_docs(documents=documents)

        assert texts == []

    def test_run_negative_topk_in_init(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        ranker = CohereRanker(top_k=-2)
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
        with pytest.raises(ValueError, match="top_k must be > 0, but got *"):
            ranker.run(query, documents)

    def test_run_zero_topk_in_init(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        ranker = CohereRanker(top_k=0)
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
        with pytest.raises(ValueError, match="top_k must be > 0, but got *"):
            ranker.run(query, documents)

    def test_run_negative_topk_in_run(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        ranker = CohereRanker()
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
        with pytest.raises(ValueError, match="top_k must be > 0, but got *"):
            ranker.run(query, documents, -3)

    def test_run_zero_topk_in_run_and_init(self, monkeypatch):
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        ranker = CohereRanker(top_k=0)
        query = "test"
        documents = [Document(content="doc1"), Document(content="doc2"), Document(content="doc3")]
        with pytest.raises(ValueError, match="top_k must be > 0, but got *"):
            ranker.run(query, documents, 0)

    def test_run_documents_provided(self, monkeypatch, mock_ranker_response):  # noqa: ARG002
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        ranker = CohereRanker()
        query = "test"
        documents = [
            Document(id="abcd", content="doc1", meta={"meta_field": "meta_value_1"}),
            Document(id="efgh", content="doc2", meta={"meta_field": "meta_value_2"}),
            Document(id="ijkl", content="doc3", meta={"meta_field": "meta_value_3"}),
        ]
        ranker_results = ranker.run(query, documents, 2)

        assert isinstance(ranker_results, dict)
        reranked_docs = ranker_results["documents"]
        assert reranked_docs == [
            Document(id="ijkl", content="doc3", meta={"meta_field": "meta_value_3"}, score=0.98),
            Document(id="efgh", content="doc2", meta={"meta_field": "meta_value_2"}, score=0.95),
        ]

    def test_run_topk_set_in_init(self, monkeypatch, mock_ranker_response):  # noqa: ARG002
        monkeypatch.setenv("CO_API_KEY", "test-api-key")
        ranker = CohereRanker(top_k=2)
        query = "test"
        documents = [
            Document(id="abcd", content="doc1"),
            Document(id="efgh", content="doc2"),
            Document(id="ijkl", content="doc3"),
        ]

        ranker_results = ranker.run(query, documents)

        assert isinstance(ranker_results, dict)
        reranked_docs = ranker_results["documents"]
        assert reranked_docs == [
            Document(id="ijkl", content="doc3", score=0.98),
            Document(id="efgh", content="doc2", score=0.95),
        ]

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        component = CohereRanker()
        documents = [
            Document(id="abcd", content="Paris is in France"),
            Document(id="efgh", content="Berlin is in Germany"),
            Document(id="ijkl", content="Lyon is in France"),
        ]

        ranker_result = component.run("Cities in France", documents, 2)
        expected_documents = [documents[0], documents[2]]
        expected_documents_content = [doc.content for doc in expected_documents]
        result_documents_contents = [doc.content for doc in ranker_result["documents"]]

        assert isinstance(ranker_result, dict)
        assert isinstance(ranker_result["documents"], list)
        assert len(ranker_result["documents"]) == 2
        assert all(isinstance(doc, Document) for doc in ranker_result["documents"])
        assert set(result_documents_contents) == set(expected_documents_content)

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_topk_greater_than_docs(self):
        component = CohereRanker()
        documents = [
            Document(id="abcd", content="Paris is in France"),
            Document(id="efgh", content="Berlin is in Germany"),
            Document(id="ijkl", content="Lyon is in France"),
        ]

        ranker_result = component.run("Cities in France", documents, 5)
        expected_documents = [documents[0], documents[2], documents[1]]
        expected_documents_content = [doc.content for doc in expected_documents]
        result_documents_contents = [doc.content for doc in ranker_result["documents"]]

        assert isinstance(ranker_result, dict)
        assert isinstance(ranker_result["documents"], list)
        assert len(ranker_result["documents"]) == 3
        assert all(isinstance(doc, Document) for doc in ranker_result["documents"])
        assert set(result_documents_contents) == set(expected_documents_content)
