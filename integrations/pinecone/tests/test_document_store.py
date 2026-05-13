# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
from unittest.mock import patch

import numpy as np
import pytest
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers import SentenceWindowRetriever
from haystack.dataclasses import ByteStream, SparseEmbedding
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountDocumentsTest,
    CountUniqueMetadataByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
    UpdateByFilterTest,
    WriteDocumentsTest,
)
from haystack.utils import Secret
from pinecone import Pinecone, PodSpec, ServerlessSpec

from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_init_is_lazy(_mock_client):
    _ = PineconeDocumentStore(api_key=Secret.from_token("fake-api-key"))
    _mock_client.assert_not_called()


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_init(mock_pinecone):
    mock_pinecone.return_value.Index.return_value.describe_index_stats.return_value = {"dimension": 60}

    document_store = PineconeDocumentStore(
        api_key=Secret.from_token("fake-api-key"),
        index="my_index",
        namespace="test",
        batch_size=50,
        dimension=30,
        metric="euclidean",
    )

    # Trigger an actual connection
    _ = document_store._initialize_index()

    mock_pinecone.assert_called_with(api_key="fake-api-key", source_tag="haystack")

    assert document_store.index_name == "my_index"
    assert document_store.namespace == "test"
    assert document_store.batch_size == 50
    assert document_store.dimension == 60
    assert document_store.metric == "euclidean"


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_init_api_key_in_environment_variable(mock_pinecone, monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")

    ds = PineconeDocumentStore(
        index="my_index",
        namespace="test",
        batch_size=50,
        dimension=30,
        metric="euclidean",
    )

    # Trigger an actual connection
    _ = ds._initialize_index()

    mock_pinecone.assert_called_with(api_key="env-api-key", source_tag="haystack")


@patch("haystack_integrations.document_stores.pinecone.document_store.Pinecone")
def test_to_from_dict(mock_pinecone, monkeypatch):
    mock_pinecone.return_value.Index.return_value.describe_index_stats.return_value = {"dimension": 60}
    monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")
    document_store = PineconeDocumentStore(
        index="my_index",
        namespace="test",
        batch_size=50,
        dimension=30,
        metric="euclidean",
    )

    # Trigger an actual connection
    document_store._initialize_index()

    dict_output = {
        "type": "haystack_integrations.document_stores.pinecone.document_store.PineconeDocumentStore",
        "init_parameters": {
            "api_key": {
                "env_vars": [
                    "PINECONE_API_KEY",
                ],
                "strict": True,
                "type": "env_var",
            },
            "index": "my_index",
            "dimension": 60,
            "namespace": "test",
            "batch_size": 50,
            "metric": "euclidean",
            "spec": {"serverless": {"region": "us-east-1", "cloud": "aws"}},
            "show_progress": True,
        },
    }
    assert document_store.to_dict() == dict_output

    document_store = PineconeDocumentStore.from_dict(dict_output)
    assert document_store.api_key == Secret.from_env_var("PINECONE_API_KEY", strict=True)
    assert document_store.index_name == "my_index"
    assert document_store.namespace == "test"
    assert document_store.batch_size == 50
    assert document_store.dimension == 60
    assert document_store.metric == "euclidean"
    assert document_store.spec == {"serverless": {"region": "us-east-1", "cloud": "aws"}}
    assert document_store.show_progress is True


def test_init_fails_wo_api_key(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        _ = PineconeDocumentStore(
            index="my_index",
        )._initialize_index()


def test_convert_dict_spec_to_pinecone_object_serverless():
    dict_spec = {"serverless": {"region": "us-east-1", "cloud": "aws"}}
    pinecone_object = PineconeDocumentStore._convert_dict_spec_to_pinecone_object(dict_spec)
    assert isinstance(pinecone_object, ServerlessSpec)
    assert pinecone_object.region == "us-east-1"
    assert pinecone_object.cloud == "aws"


def test_convert_dict_spec_to_pinecone_object_pod():
    dict_spec = {"pod": {"replicas": 1, "shards": 1, "pods": 1, "pod_type": "p1.x1", "environment": "us-west1-gcp"}}
    pinecone_object = PineconeDocumentStore._convert_dict_spec_to_pinecone_object(dict_spec)

    assert isinstance(pinecone_object, PodSpec)
    assert pinecone_object.replicas == 1
    assert pinecone_object.shards == 1
    assert pinecone_object.pods == 1
    assert pinecone_object.pod_type == "p1.x1"
    assert pinecone_object.environment == "us-west1-gcp"


def test_convert_dict_spec_to_pinecone_object_fail():
    dict_spec = {
        "strange_key": {"replicas": 1, "shards": 1, "pods": 1, "pod_type": "p1.x1", "environment": "us-west1-gcp"}
    }
    with pytest.raises(ValueError):
        PineconeDocumentStore._convert_dict_spec_to_pinecone_object(dict_spec)


def test_discard_invalid_meta_invalid():
    invalid_metadata_doc = Document(
        content="The moonlight shimmered ",
        meta={
            "source_id": "62049ba1d1e1d5ebb1f6230b0b00c5356b8706c56e0b9c36b1dfc86084cd75f0",
            "page_number": 1,
            "split_id": 0,
            "split_idx_start": 0,
            "_split_overlap": [
                {"doc_id": "68ed48ba830048c5d7815874ed2de794722e6d10866b6c55349a914fd9a0df65", "range": (0, 20)}
            ],
        },
    )
    result = PineconeDocumentStore._discard_invalid_meta(invalid_metadata_doc)

    assert result.meta["source_id"] == "62049ba1d1e1d5ebb1f6230b0b00c5356b8706c56e0b9c36b1dfc86084cd75f0"
    assert result.meta["page_number"] == 1
    assert result.meta["split_id"] == 0
    assert result.meta["split_idx_start"] == 0
    assert "_split_overlap" not in result.meta


def test_discard_invalid_meta_valid():
    valid_metadata_doc = Document(
        content="The moonlight shimmered ",
        meta={
            "source_id": "62049ba1d1e1d5ebb1f6230b0b00c5356b8706c56e0b9c36b1dfc86084cd75f0",
            "page_number": 1,
        },
    )
    result = PineconeDocumentStore._discard_invalid_meta(valid_metadata_doc)

    assert result.meta["source_id"] == "62049ba1d1e1d5ebb1f6230b0b00c5356b8706c56e0b9c36b1dfc86084cd75f0"
    assert result.meta["page_number"] == 1


def test_convert_meta_to_int():
    # Test with floats
    meta_data = {"split_id": 1.0, "split_idx_start": 2.0, "page_number": 3.0}
    assert PineconeDocumentStore._convert_meta_to_int(meta_data) == {
        "split_id": 1,
        "split_idx_start": 2,
        "page_number": 3,
    }

    # Test with floats and ints
    meta_data = {"split_id": 1.0, "split_idx_start": 2, "page_number": 3.0}
    assert PineconeDocumentStore._convert_meta_to_int(meta_data) == {
        "split_id": 1,
        "split_idx_start": 2,
        "page_number": 3,
    }

    # Test with floats and strings
    meta_data = {"split_id": 1.0, "other": "other_data", "page_number": 3.0}
    assert PineconeDocumentStore._convert_meta_to_int(meta_data) == {
        "split_id": 1,
        "other": "other_data",
        "page_number": 3,
    }

    # Test with empty dict
    meta_data = {}
    assert PineconeDocumentStore._convert_meta_to_int(meta_data) == {}


@pytest.mark.parametrize(
    ("documents", "expected", "warning_fragment"),
    [
        ([], {}, None),
        (
            [Document(content="hello", meta={"flag": True})],
            {"content": {"type": "text"}, "flag": {"type": "boolean"}},
            None,
        ),
        (
            [Document(content=None, meta={"tags": ["a", "b"]})],
            {"tags": {"type": "keyword"}},
            None,
        ),
        (
            [Document(content=None, meta={"counts": [1, 2]})],
            {"counts": {"type": "long"}},
            None,
        ),
        (
            [Document(content=None, meta={"empty": []})],
            {"empty": {"type": "keyword"}},
            None,
        ),
        (
            [Document(content=None, meta={"pi": 3.14})],
            {"pi": {"type": "long"}},
            None,
        ),
        (
            [
                Document(content=None, meta={"value": 1}),
                Document(content=None, meta={"value": "two"}),
            ],
            {"value": {"type": "keyword"}},
            "mixed types",
        ),
    ],
)
def test_get_metadata_fields_info_impl_type_inference(documents, expected, warning_fragment, caplog):
    with caplog.at_level("WARNING"):
        result = PineconeDocumentStore._get_metadata_fields_info_impl(documents)
    assert result == expected
    if warning_fragment:
        assert warning_fragment in caplog.text


def test_get_metadata_field_min_max_impl_strips_meta_prefix_and_handles_missing():
    docs = [
        Document(content="a", meta={"priority": 1}),
        Document(content="b", meta={"priority": 5}),
    ]
    assert PineconeDocumentStore._get_metadata_field_min_max_impl(docs, "meta.priority") == {"min": 1, "max": 5}
    assert PineconeDocumentStore._get_metadata_field_min_max_impl(docs, "missing") == {"min": None, "max": None}


def test_get_metadata_field_unique_values_impl_pagination_search_and_lists():
    docs = [
        Document(content="a", meta={"tags": ["python", "java"]}),
        Document(content="b", meta={"tags": ["rust", "go"]}),
        Document(content="c", meta={"tags": ["python"]}),
    ]

    values, total = PineconeDocumentStore._get_metadata_field_unique_values_impl(
        docs, "tags", search_term=None, from_=0, size=10
    )
    assert total == 4
    assert values == ["go", "java", "python", "rust"]

    values, total = PineconeDocumentStore._get_metadata_field_unique_values_impl(
        docs, "tags", search_term=None, from_=1, size=2
    )
    assert total == 4
    assert values == ["java", "python"]

    values, total = PineconeDocumentStore._get_metadata_field_unique_values_impl(
        docs, "tags", search_term="PY", from_=0, size=10
    )
    assert total == 1
    assert values == ["python"]


def test_prepare_documents_for_writing_edge_cases(caplog):
    ds = PineconeDocumentStore(api_key=Secret.from_token("fake-api-key"))

    with pytest.raises(ValueError, match="must contain a list of objects of type Document"):
        ds._prepare_documents_for_writing(["not-a-document"], policy=DuplicatePolicy.NONE)

    docs = [
        Document(content="no-embedding"),
        Document(content="with-blob", embedding=[0.1] * 768, blob=ByteStream(data=b"data")),
        Document(
            content="with-sparse",
            embedding=[0.1] * 768,
            sparse_embedding=SparseEmbedding(indices=[0], values=[1.0]),
        ),
    ]
    with caplog.at_level("WARNING"):
        result = ds._prepare_documents_for_writing(docs, policy=DuplicatePolicy.SKIP)

    assert len(result) == 3
    assert result[0][1] == ds._dummy_vector
    assert "only supports `DuplicatePolicy.OVERWRITE`" in caplog.text
    assert "has no embedding" in caplog.text
    assert "blob" in caplog.text
    assert "sparse_embedding" in caplog.text


@pytest.mark.asyncio
async def test_validation_errors_on_empty_query_and_non_dict_meta():
    ds = PineconeDocumentStore(api_key=Secret.from_token("fake-api-key"))
    filters = {"field": "meta.category", "operator": "==", "value": "A"}

    with pytest.raises(ValueError, match="query_embedding must be a non-empty list"):
        ds._embedding_retrieval(query_embedding=[])
    with pytest.raises(ValueError, match="query_embedding must be a non-empty list"):
        await ds._embedding_retrieval_async(query_embedding=[])

    with pytest.raises(ValueError, match="meta must be a dictionary"):
        ds.update_by_filter(filters=filters, meta="not-a-dict")
    with pytest.raises(ValueError, match="meta must be a dictionary"):
        await ds.update_by_filter_async(filters=filters, meta="not-a-dict")


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("PINECONE_API_KEY"), reason="PINECONE_API_KEY not set")
def test_serverless_index_creation_from_scratch(delete_sleep_time):
    # we use a fixed index name to avoid hitting the limit of Pinecone's free tier (max 5 indexes)
    # the index name is defined in the test matrix of the GitHub Actions workflow
    # the default value is provided for local testing
    index_name = os.environ.get("INDEX_NAME", "serverless-test-index")

    client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    try:
        client.delete_index(name=index_name)
    except Exception:  # noqa S110
        pass

    time.sleep(delete_sleep_time)

    ds = PineconeDocumentStore(
        index=index_name,
        namespace="test",
        batch_size=50,
        dimension=30,
        metric="euclidean",
        spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
        show_progress=False,
    )
    # Trigger the connection
    _ = ds._initialize_index()

    index_description = client.describe_index(name=index_name)
    assert index_description["name"] == index_name
    assert index_description["dimension"] == 30
    assert index_description["metric"] == "euclidean"
    assert index_description["spec"]["serverless"]["region"] == "us-east-1"
    assert index_description["spec"]["serverless"]["cloud"] == "aws"

    try:
        client.delete_index(name=index_name)
    except Exception:  # noqa S110
        pass


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("PINECONE_API_KEY"), reason="PINECONE_API_KEY not set")
class TestDocumentStore(
    CountDocumentsTest,
    DeleteDocumentsTest,
    WriteDocumentsTest,
    FilterableDocsFixtureMixin,
    UpdateByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    def test_write_documents(self, document_store: PineconeDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1

    @pytest.mark.xfail(
        run=True, reason="Pinecone supports overwriting by default, but it takes a while for it to take effect"
    )
    def test_write_documents_duplicate_overwrite(self, document_store: PineconeDocumentStore): ...

    @pytest.mark.skip(reason="Pinecone only supports UPSERT operations")
    def test_write_documents_duplicate_fail(self, document_store: PineconeDocumentStore): ...

    @pytest.mark.skip(reason="Pinecone only supports UPSERT operations")
    def test_write_documents_duplicate_skip(self, document_store: PineconeDocumentStore): ...

    @pytest.mark.skip(reason="Pinecone creates a namespace only when the first document is written")
    def test_delete_documents_empty_document_store(self, document_store: PineconeDocumentStore): ...

    def test_embedding_retrieval(self, document_store: PineconeDocumentStore):
        query_embedding = [0.1] * 768
        most_similar_embedding = [0.8] * 768
        second_best_embedding = [0.8] * 700 + [0.1] * 3 + [0.2] * 65
        another_embedding = [0.1] * 384 + [-0.1] * 384

        docs = [
            Document(content="Most similar document", embedding=most_similar_embedding),
            Document(content="2nd best document", embedding=second_best_embedding),
            Document(content="Not very similar document", embedding=another_embedding),
        ]

        document_store.write_documents(docs)

        results = document_store._embedding_retrieval(query_embedding=query_embedding, top_k=2, filters={})

        assert len(results) == 2
        # Pinecone does not seem to guarantee the order of the results
        assert "Most similar document" in [result.content for result in results]
        assert "2nd best document" in [result.content for result in results]

    def test_close(self, document_store: PineconeDocumentStore):
        document_store._initialize_index()
        assert document_store._index is not None

        document_store.close()
        assert document_store._index is None

        document_store._initialize_index()
        assert document_store._index is not None
        # test that the index is still usable after closing and reopening
        assert document_store.count_documents() == 0

    def test_sentence_window_retriever(self, document_store: PineconeDocumentStore):
        # indexing
        splitter = DocumentSplitter(split_length=10, split_overlap=5, split_by="word")
        text = (
            "Whose woods these are I think I know. His house is in the village though; He will not see me stopping "
            "here To watch his woods fill up with snow."
        )
        docs = splitter.run(documents=[Document(content=text)])

        for idx, doc in enumerate(docs["documents"]):
            if idx == 2:
                doc.embedding = [0.1] * 768
                continue
            doc.embedding = np.random.rand(768).tolist()
        document_store.write_documents(docs["documents"])

        # query
        embedding_retriever = PineconeEmbeddingRetriever(document_store=document_store)
        query_embedding = [0.1] * 768
        retrieved_doc = embedding_retriever.run(query_embedding=query_embedding, top_k=1, filters={})
        sentence_window_retriever = SentenceWindowRetriever(document_store=document_store, window_size=2)
        result = sentence_window_retriever.run(retrieved_documents=[retrieved_doc["documents"][0]])

        assert len(result["context_windows"]) == 1

    def test_get_metadata_fields_info_consistent_types(self, document_store: PineconeDocumentStore):
        # Test that all documents are checked for type consistency
        docs = [
            Document(content="Doc 1", meta={"score": 85}),
            Document(content="Doc 2", meta={"score": 90}),
            Document(content="Doc 3", meta={"score": 78}),
        ]
        document_store.write_documents(docs)

        field_info = document_store.get_metadata_fields_info()
        assert "score" in field_info
        assert field_info["score"]["type"] == "long"

    def test_get_metadata_field_min_max_boolean_and_string(self, document_store: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"priority": 1, "score": 85.5, "active": True, "category": "Zebra"}),
            Document(content="Doc 2", meta={"priority": 5, "score": 92.3, "active": False, "category": "Alpha"}),
            Document(content="Doc 3", meta={"priority": 3, "score": 78.9, "active": True, "category": "Beta"}),
            Document(content="Doc 4", meta={"priority": 7, "score": 95.1, "active": False, "category": "Gamma"}),
        ]
        document_store.write_documents(docs)

        # Get min/max for boolean field
        min_max = document_store.get_metadata_field_min_max("active")
        assert min_max["min"] is False
        assert min_max["max"] is True

        # Get min/max for string field (alphabetical ordering)
        min_max = document_store.get_metadata_field_min_max("category")
        assert min_max["min"] == "Alpha"
        assert min_max["max"] == "Zebra"

    def test_get_metadata_field_min_max_no_values(self, document_store: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"tags": ["tag1", "tag2"]}),
            Document(content="Doc 2", meta={"tags": ["tag3", "tag4"]}),
        ]
        document_store.write_documents(docs)

        # Unsupported field type (list) — no comparable values collected
        assert document_store.get_metadata_field_min_max("tags") == {"min": None, "max": None}

        # Non-existent field
        assert document_store.get_metadata_field_min_max("nonexistent") == {"min": None, "max": None}

    def test_get_metadata_field_unique_values(self, document_store: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "Alpha"}),
            Document(content="Doc 2", meta={"category": "Beta"}),
            Document(content="Doc 3", meta={"category": "Gamma"}),
            Document(content="Doc 4", meta={"category": "Alpha"}),
            Document(content="Doc 5", meta={"category": "Delta"}),
            Document(content="Doc 6", meta={"category": "Beta"}),
        ]
        document_store.write_documents(docs)

        # Get all unique values
        values, total = document_store.get_metadata_field_unique_values("category", from_=0, size=10)
        assert total == 4  # Alpha, Beta, Delta, Gamma
        assert len(values) == 4
        assert set(values) == {"Alpha", "Beta", "Delta", "Gamma"}

        # Test pagination
        values, total = document_store.get_metadata_field_unique_values("category", from_=0, size=2)
        assert total == 4
        assert len(values) == 2  # First 2 values (alphabetically sorted)

        values, total = document_store.get_metadata_field_unique_values("category", from_=2, size=2)
        assert total == 4
        assert len(values) == 2  # Next 2 values

        # Test search term
        values, total = document_store.get_metadata_field_unique_values("category", search_term="ta", size=10)
        assert total == 2  # Beta and Delta contain "ta"
        assert set(values) == {"Beta", "Delta"}

        # Test case-insensitive search
        values, total = document_store.get_metadata_field_unique_values("category", search_term="ALPHA", size=10)
        assert total == 1
        assert values == ["Alpha"]

    def test_get_metadata_field_unique_values_with_lists(self, document_store: PineconeDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"tags": ["python", "java"]}),
            Document(content="Doc 2", meta={"tags": ["python", "rust"]}),
            Document(content="Doc 3", meta={"tags": ["java", "go"]}),
        ]
        document_store.write_documents(docs)

        # Get unique tag values
        values, total = document_store.get_metadata_field_unique_values("tags", size=10)
        assert total == 4  # python, java, rust, go
        assert set(values) == {"go", "java", "python", "rust"}
