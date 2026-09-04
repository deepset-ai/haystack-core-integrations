# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers import SentenceWindowRetriever
from haystack.dataclasses import ByteStream, Document
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

from haystack_integrations.components.retrievers.amazon_s3_vectors import S3VectorsEmbeddingRetriever
from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore


def test_init_is_lazy():
    store = S3VectorsDocumentStore(
        vector_bucket_name="my-bucket",
        index_name="my-index",
        dimension=768,
    )
    assert store._client is None


def test_init_default_params():
    store = S3VectorsDocumentStore(
        vector_bucket_name="my-bucket",
        index_name="my-index",
        dimension=768,
    )
    assert store.vector_bucket_name == "my-bucket"
    assert store.index_name == "my-index"
    assert store.dimension == 768
    assert store.distance_metric == "cosine"
    assert store.create_bucket_and_index is True
    assert store.non_filterable_metadata_keys == []


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_to_dict(_mock_boto3):
    store = S3VectorsDocumentStore(
        vector_bucket_name="my-bucket",
        index_name="my-index",
        dimension=768,
        distance_metric="euclidean",
        region_name="us-west-2",
        create_bucket_and_index=False,
    )
    d = store.to_dict()
    assert d == {
        "type": "haystack_integrations.document_stores.amazon_s3_vectors.document_store.S3VectorsDocumentStore",
        "init_parameters": {
            "vector_bucket_name": "my-bucket",
            "index_name": "my-index",
            "dimension": 768,
            "distance_metric": "euclidean",
            "region_name": "us-west-2",
            "aws_access_key_id": None,
            "aws_secret_access_key": None,
            "aws_session_token": None,
            "create_bucket_and_index": False,
            "non_filterable_metadata_keys": [],
        },
    }


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_from_dict(_mock_boto3):
    data = {
        "type": "haystack_integrations.document_stores.amazon_s3_vectors.document_store.S3VectorsDocumentStore",
        "init_parameters": {
            "vector_bucket_name": "my-bucket",
            "index_name": "my-index",
            "dimension": 768,
            "distance_metric": "euclidean",
            "region_name": "us-west-2",
            "aws_access_key_id": None,
            "aws_secret_access_key": None,
            "aws_session_token": None,
            "create_bucket_and_index": False,
            "non_filterable_metadata_keys": [],
        },
    }
    store = S3VectorsDocumentStore.from_dict(data)
    assert store.vector_bucket_name == "my-bucket"
    assert store.index_name == "my-index"
    assert store.dimension == 768
    assert store.distance_metric == "euclidean"
    assert store.region_name == "us-west-2"
    assert store.create_bucket_and_index is False


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents_missing_embedding_uses_placeholder(mock_boto3):
    """Every S3 Vectors record needs a vector, so embedding-less documents get the placeholder."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    docs = [Document(id="1", content="ok", embedding=[0.1] * 4), Document(id="2", content="missing")]
    assert store.write_documents(docs) == 2

    written = client.put_vectors.call_args.kwargs["vectors"]
    assert written[0]["data"]["float32"] == [0.1] * 4
    # Non-zero: a cosine index rejects zero-norm vectors.
    assert written[1]["data"]["float32"] == [-10.0] * 4
    # The caller's Document is not mutated.
    assert docs[1].embedding is None


@pytest.mark.parametrize("policy", [DuplicatePolicy.FAIL, DuplicatePolicy.SKIP])
@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents_unsupported_policy_overwrites(mock_boto3, policy):
    """
    put_vectors is an upsert and S3 Vectors has no cheap existence check, so policies other than
    OVERWRITE are ignored instead of paying for an extra read pass.
    """
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    assert store.write_documents([Document(id="1", content="Hello", embedding=[0.1] * 4)], policy=policy) == 1
    client.get_vectors.assert_not_called()
    client.put_vectors.assert_called_once()


def test_write_documents_invalid_input_raises():
    """write_documents must reject non-list / non-Document input before doing anything."""
    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, create_bucket_and_index=False)
    with pytest.raises(ValueError):
        store.write_documents("not a list")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.write_documents(["not a document"])  # type: ignore[list-item]


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_embedding_retrieval_score_conversion(mock_boto3):
    """Tests our distance-to-score conversion logic — the only non-trivial transform in retrieval."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.query_vectors.return_value = {
        "vectors": [{"key": "1", "distance": 0.05, "metadata": {"_content": "Hello", "category": "news"}}],
        "distanceMetric": "cosine",
    }
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    docs = store._embedding_retrieval(query_embedding=[0.1] * 4, top_k=5)
    assert len(docs) == 1
    assert docs[0].id == "1"
    assert docs[0].content == "Hello"
    assert docs[0].score == pytest.approx(0.95)  # cosine: 1.0 - 0.05
    assert docs[0].meta == {"category": "news"}


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_embedding_retrieval_euclidean_score(mock_boto3):
    """Tests euclidean distance-to-score conversion (negated)."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.query_vectors.return_value = {
        "vectors": [{"key": "1", "distance": 1.5, "metadata": {}}],
        "distanceMetric": "euclidean",
    }
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(
        vector_bucket_name="b", index_name="i", dimension=4, distance_metric="euclidean", region_name="us-east-1"
    )
    docs = store._embedding_retrieval(query_embedding=[0.1] * 4, top_k=5)
    assert docs[0].score == pytest.approx(-1.5)  # euclidean: negated


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_embedding_retrieval_passes_filters(mock_boto3):
    """Tests that Haystack filters are converted and passed to query_vectors."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.query_vectors.return_value = {"vectors": [], "distanceMetric": "cosine"}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    filters = {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": "news"}]}
    store._embedding_retrieval(query_embedding=[0.1] * 4, filters=filters, top_k=5)

    call_args = client.query_vectors.call_args[1]
    assert call_args["filter"] == {"$and": [{"category": {"$eq": "news"}}]}


def test_embedding_retrieval_empty_embedding_raises():
    """Tests our input validation — no mocking needed."""
    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, create_bucket_and_index=False)
    with pytest.raises(ValueError, match="non-empty"):
        store._embedding_retrieval(query_embedding=[])


def test_document_to_s3_vector():
    """Tests our Document → S3 vector conversion (pure function)."""
    doc = Document(
        id="test-1", content="Hello world", embedding=[0.1, 0.2, 0.3], meta={"category": "test", "year": 2024}
    )
    result = S3VectorsDocumentStore._document_to_s3_vector(doc)
    assert result["key"] == "test-1"
    assert result["data"] == {"float32": [0.1, 0.2, 0.3]}
    assert result["metadata"]["_content"] == "Hello world"
    assert result["metadata"]["category"] == "test"
    assert result["metadata"]["year"] == 2024


def test_s3_vector_to_document():
    """Tests our S3 vector → Document conversion (pure function)."""
    vector = {
        "key": "test-1",
        "data": {"float32": [0.1, 0.2, 0.3]},
        "metadata": {"_content": "Hello world", "category": "test"},
    }
    doc = S3VectorsDocumentStore._s3_vector_to_document(vector)
    assert doc.id == "test-1"
    assert doc.content == "Hello world"
    assert doc.embedding == [0.1, 0.2, 0.3]
    assert doc.meta == {"category": "test"}


def test_document_roundtrip():
    """Tests Document → S3 vector → Document is lossless."""
    doc = Document(
        id="test-1", content="Hello world", embedding=[0.1, 0.2, 0.3], meta={"category": "test", "year": 2024}
    )
    vector = S3VectorsDocumentStore._document_to_s3_vector(doc)
    restored = S3VectorsDocumentStore._s3_vector_to_document(vector)
    assert restored.id == doc.id
    assert restored.content == doc.content
    assert restored.embedding == doc.embedding
    assert restored.meta == doc.meta


def test_document_roundtrip_without_embedding():
    """A document written without an embedding must read back without one, not with the placeholder."""
    placeholder = [-10.0] * 3
    doc = Document(id="test-1", content="Hello world", meta={"category": "test"})

    vector = S3VectorsDocumentStore._document_to_s3_vector(doc, fallback_embedding=placeholder)
    assert vector["data"]["float32"] == placeholder

    restored = S3VectorsDocumentStore._s3_vector_to_document(vector, placeholder_embedding=placeholder)
    assert restored.embedding is None
    assert restored.content == doc.content
    assert restored.meta == doc.meta


def test_s3_vector_to_document_keeps_embedding_matching_no_placeholder():
    """A real embedding is never mistaken for the placeholder."""
    vector = {"key": "test-1", "data": {"float32": [0.1, 0.2, 0.3]}, "metadata": {}}
    doc = S3VectorsDocumentStore._s3_vector_to_document(vector, placeholder_embedding=[-10.0] * 3)
    assert doc.embedding == [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    ("value", "supported"),
    [
        # Every case below was verified against the live S3 Vectors API: the "False" rows all come
        # back as `ValidationException` from PutVectors.
        ("text", True),
        ("", True),
        (True, True),
        (0, True),
        (2**63, True),
        (1.5, True),
        (["x", "y"], True),
        ([1, 2.5], True),
        (None, False),
        ({"a": 1}, False),
        ([], False),
        ([1, "x"], False),  # an array may not mix strings and numbers
        ([True, False], False),  # booleans are allowed only outside arrays
        ([{"doc_id": "a", "range": [0, 24]}], False),  # DocumentSplitter's `_split_overlap`
        ([[1, 2]], False),
        (["x", None], False),
        (float("nan"), False),
        (float("inf"), False),
    ],
)
def test_is_supported_metadata_value(value, supported):
    assert S3VectorsDocumentStore._is_supported_metadata_value(value) is supported


def test_document_to_s3_vector_drops_unstorable_metadata(caplog):
    """
    Values S3 Vectors rejects are dropped with one warning, instead of failing the whole write.

    The `_split_overlap` case is the one that bites in practice: `DocumentSplitter` with
    `split_overlap` adds it to every chunk, so a plain splitter-to-store pipeline would otherwise
    fail with a `ParamValidationError`.
    """
    doc = Document(
        id="test-1",
        content="Hello world",
        meta={
            "keep_str": "x",
            "keep_list": ["a", "b"],
            "_split_overlap": [{"doc_id": "a", "range": (0, 24)}],
            "nothing": None,
            "nested": {"a": 1},
            "empty": [],
        },
    )
    result = S3VectorsDocumentStore._document_to_s3_vector(doc)

    assert result["metadata"] == {"_content": "Hello world", "keep_str": "x", "keep_list": ["a", "b"]}
    assert "['_split_overlap', 'empty', 'nested', 'nothing']" in caplog.text


def test_document_to_s3_vector_converts_tuples_to_lists():
    """A tuple is a valid array for S3 Vectors, but botocore's validator only accepts `list`."""
    doc = Document(id="test-1", meta={"tags": ("a", "b")}, embedding=[0.1])
    result = S3VectorsDocumentStore._document_to_s3_vector(doc)
    assert result["metadata"]["tags"] == ["a", "b"]


def test_blob_roundtrip():
    """A blob and its (dict) metadata survive the round-trip; the dict is stored JSON-encoded."""
    doc = Document(
        id="test-1",
        content="Hello world",
        blob=ByteStream(data=b"\x00\x01binary", meta={"name": "f.pdf", "pages": 3}, mime_type="application/pdf"),
        embedding=[0.1, 0.2, 0.3],
    )
    vector = S3VectorsDocumentStore._document_to_s3_vector(doc)
    # S3 Vectors rejects dicts, so the blob meta must go over the wire as a string.
    assert vector["metadata"]["_blob_meta"] == '{"name": "f.pdf", "pages": 3}'

    restored = S3VectorsDocumentStore._s3_vector_to_document(vector)
    assert restored.blob is not None
    assert restored.blob.data == b"\x00\x01binary"
    assert restored.blob.meta == {"name": "f.pdf", "pages": 3}
    assert restored.blob.mime_type == "application/pdf"
    assert restored.meta == {}


def test_blob_roundtrip_with_undecodable_meta(caplog):
    """Corrupt stored blob metadata degrades to an empty dict rather than raising."""
    vector = {
        "key": "test-1",
        "data": {"float32": [0.1]},
        "metadata": {"_blob_data": base64.b64encode(b"x").decode("ascii"), "_blob_meta": "not json"},
    }
    doc = S3VectorsDocumentStore._s3_vector_to_document(vector)
    assert doc.blob is not None
    assert doc.blob.meta == {}
    assert "Could not decode the stored blob metadata" in caplog.text


@pytest.mark.parametrize(
    ("documents", "expected", "warns"),
    [
        ([], {}, False),
        ([Document(content="a", meta={"n": 1})], {"content": {"type": "text"}, "n": {"type": "long"}}, False),
        ([Document(content="a", meta={"f": 1.5})], {"content": {"type": "text"}, "f": {"type": "long"}}, False),
        ([Document(content="a", meta={"s": "x"})], {"content": {"type": "text"}, "s": {"type": "keyword"}}, False),
        # bool must win over int: bool is a subclass of int in Python.
        ([Document(content="a", meta={"b": True})], {"content": {"type": "text"}, "b": {"type": "boolean"}}, False),
        # A list is typed by its first element; an empty list carries no type info.
        (
            [Document(content="a", meta={"tags": ["x"]})],
            {"content": {"type": "text"}, "tags": {"type": "keyword"}},
            False,
        ),
        ([Document(content="a", meta={"tags": []})], {"content": {"type": "text"}, "tags": {"type": "keyword"}}, False),
        # Mixed types across documents fall back to keyword, with a warning.
        (
            [Document(content="a", meta={"m": 1}), Document(content="b", meta={"m": "x"})],
            {"content": {"type": "text"}, "m": {"type": "keyword"}},
            True,
        ),
        # No content anywhere means no synthetic "content" field.
        ([Document(meta={"n": 1}, embedding=[0.1])], {"n": {"type": "long"}}, False),
    ],
)
def test_get_metadata_fields_info_impl(documents, expected, warns, caplog):
    """Type inference is a pure function over already-fetched documents."""
    assert S3VectorsDocumentStore._get_metadata_fields_info_impl(documents) == expected
    assert ("mixed types" in caplog.text) is warns


def test_get_metadata_field_min_max_impl():
    """min/max strips the meta. prefix, compares numerically, and reports None when absent."""
    docs = [
        Document(content="a", meta={"priority": 5}),
        Document(content="b", meta={"priority": 1}),
        # 10 vs 5 catches an implementation comparing numbers as strings ("10" < "5").
        Document(content="c", meta={"priority": 10}),
        Document(content="d", meta={}),
    ]
    assert S3VectorsDocumentStore._get_metadata_field_min_max_impl(docs, "priority") == {"min": 1, "max": 10}
    assert S3VectorsDocumentStore._get_metadata_field_min_max_impl(docs, "meta.priority") == {"min": 1, "max": 10}
    assert S3VectorsDocumentStore._get_metadata_field_min_max_impl(docs, "absent") == {"min": None, "max": None}
    assert S3VectorsDocumentStore._get_metadata_field_min_max_impl([], "priority") == {"min": None, "max": None}

    strings = [Document(content="a", meta={"k": "beta"}), Document(content="b", meta={"k": "alpha"})]
    assert S3VectorsDocumentStore._get_metadata_field_min_max_impl(strings, "k") == {"min": "alpha", "max": "beta"}

    # Mixed str/numeric would raise on a naive min(); numerics win instead of blowing up.
    mixed = [Document(content="a", meta={"k": 3}), Document(content="b", meta={"k": "zzz"})]
    assert S3VectorsDocumentStore._get_metadata_field_min_max_impl(mixed, "k") == {"min": 3, "max": 3}


def test_get_metadata_field_unique_values_impl_search_and_pagination():
    """Distinct values sort stably, then search_term and from_/size are applied."""
    docs = [Document(content=f"d{i}", meta={"category": f"category_{i}"}) for i in range(5)]
    docs.append(Document(content="dup", meta={"category": "category_0"}))

    values, total = S3VectorsDocumentStore._get_metadata_field_unique_values_impl(docs, "category", None, 0, 10)
    assert values == [f"category_{i}" for i in range(5)]
    assert total == 5

    first, total = S3VectorsDocumentStore._get_metadata_field_unique_values_impl(docs, "category", None, 0, 2)
    second, _ = S3VectorsDocumentStore._get_metadata_field_unique_values_impl(docs, "category", None, 2, 2)
    assert first == ["category_0", "category_1"]
    assert second == ["category_2", "category_3"]
    assert total == 5

    # search_term matches the value, case-insensitively — not the document content.
    hits, total = S3VectorsDocumentStore._get_metadata_field_unique_values_impl(docs, "meta.category", "GORY_3", 0, 10)
    assert hits == ["category_3"]
    assert total == 1


def test_get_metadata_field_unique_values_impl_keeps_types_distinct():
    """`1`, `1.0`, `"1"` and `True` all compare equal in Python but must stay distinct."""
    docs = [
        Document(content="a", meta={"p": 1}),
        Document(content="b", meta={"p": "1"}),
        Document(content="c", meta={"p": 1.0}),
        Document(content="d", meta={"p": True}),
        Document(content="e", meta={"p": 1}),
    ]
    values, total = S3VectorsDocumentStore._get_metadata_field_unique_values_impl(docs, "p", None, 0, 10)
    assert total == 4
    assert {type(v) for v in values} == {int, str, float, bool}

    # List-valued metadata contributes each item, not the list.
    listed = [Document(content="a", meta={"tags": ["x", "y"]}), Document(content="b", meta={"tags": ["y"]})]
    values, total = S3VectorsDocumentStore._get_metadata_field_unique_values_impl(listed, "tags", None, 0, 10)
    assert values == ["x", "y"]
    assert total == 2


def test_count_unique_metadata_impl():
    """Distinct-value counts are per field and keep differently-typed values apart."""
    docs = [
        Document(content="a", meta={"category": "A", "status": "active"}),
        Document(content="b", meta={"category": "B", "status": "active"}),
        Document(content="c", meta={"category": "A", "status": "inactive"}),
    ]
    assert S3VectorsDocumentStore._count_unique_metadata_impl(docs, ["category", "status"]) == {
        "category": 2,
        "status": 2,
    }
    assert S3VectorsDocumentStore._count_unique_metadata_impl(docs, ["absent"]) == {"absent": 0}
    assert S3VectorsDocumentStore._count_unique_metadata_impl([], ["category"]) == {"category": 0}

    listed = [Document(content="a", meta={"tags": ["x", "y"]}), Document(content="b", meta={"tags": ["y", "z"]})]
    assert S3VectorsDocumentStore._count_unique_metadata_impl(listed, ["tags"]) == {"tags": 3}


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_close_releases_the_client_and_allows_reopen(mock_boto3):
    """close() closes the boto3 client; the store stays usable and lazily builds a new one."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    store._get_client()
    assert store._client is not None

    store.close()
    client.close.assert_called_once()
    assert store._client is None

    store._get_client()
    assert store._client is not None


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_close_is_a_noop_when_never_connected(mock_boto3):
    """close() before any call must not create a client just to close it."""
    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    store.close()
    mock_boto3.client.assert_not_called()


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_close_is_exception_safe(mock_boto3):
    """A failure while closing must not propagate to the caller."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {}
    client.close.side_effect = RuntimeError("boom")
    mock_boto3.client.return_value = client

    store = S3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    store._get_client()
    store.close()  # must not raise
    assert store._client is None


@pytest.mark.integration
class TestDocumentStore(
    CountDocumentsTest,
    WriteDocumentsTest,
    DeleteDocumentsTest,
    DeleteAllTest,
    DeleteByFilterTest,
    UpdateByFilterTest,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
    FilterableDocsFixtureMixin,
):
    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]) -> None:
        """
        Compare documents while tolerating two S3 Vectors quirks:

        * embeddings round-trip through float32 storage, so we use `pytest.approx`;
        * the `score` field is not set by `filter_documents`, only by retrieval, so
          we ignore it for equality.
        """
        assert len(received) == len(expected)
        received.sort(key=lambda d: d.id)
        expected.sort(key=lambda d: d.id)
        for r, e in zip(received, expected, strict=True):
            r_norm = replace(r, embedding=None, score=None)
            e_norm = replace(e, embedding=None, score=None)
            assert r_norm == e_norm
            if r.embedding is not None and e.embedding is not None:
                assert r.embedding == pytest.approx(e.embedding, abs=1e-5)

    def test_write_documents(self, document_store: S3VectorsDocumentStore) -> None:
        """
        Default behaviour is OVERWRITE (S3 Vectors `put_vectors` is upsert), so
        writing the same document twice succeeds and returns 1 each time.
        """
        docs = [Document(id="1", content="hello", embedding=[0.1] * 768)]
        assert document_store.write_documents(docs) == 1
        assert document_store.write_documents(docs) == 1

    def test_close_and_reopen(self, document_store: S3VectorsDocumentStore) -> None:
        """close() drops the boto3 client; the store keeps working against the same index."""
        document_store._get_client()
        assert document_store._client is not None

        document_store.close()
        assert document_store._client is None

        assert document_store.count_documents() == 0
        assert document_store._client is not None

    def test_sentence_window_retriever(self, document_store: S3VectorsDocumentStore) -> None:
        """The store composes with SentenceWindowRetriever, which pulls neighbours via filter_documents."""
        splitter = DocumentSplitter(split_length=10, split_overlap=5, split_by="word")
        splitter.warm_up()
        text = (
            "Whose woods these are I think I know. His house is in the village though; He will not see me stopping "
            "here To watch his woods fill up with snow."
        )
        docs = splitter.run(documents=[Document(content=text)])["documents"]

        rng = np.random.default_rng(seed=42)
        # One chunk gets the query vector itself, so retrieval deterministically lands on it.
        docs = [
            replace(doc, embedding=[0.1] * 768 if idx == 2 else rng.random(768).tolist())
            for idx, doc in enumerate(docs)
        ]
        document_store.write_documents(docs)

        retrieved = S3VectorsEmbeddingRetriever(document_store=document_store).run(query_embedding=[0.1] * 768, top_k=1)
        result = SentenceWindowRetriever(document_store=document_store, window_size=2).run(
            retrieved_documents=[retrieved["documents"][0]]
        )
        assert len(result["context_windows"]) == 1

    def test_get_metadata_fields_info_consistent_types(self, document_store: S3VectorsDocumentStore) -> None:
        """A field with the same type in every document keeps that type, with no mixed-type fallback."""
        docs = [
            Document(content="Doc 1", meta={"score": 85}),
            Document(content="Doc 2", meta={"score": 90}),
            Document(content="Doc 3", meta={"score": 78}),
        ]
        document_store.write_documents(docs)

        fields_info = document_store.get_metadata_fields_info()
        assert fields_info["score"] == {"type": "long"}

    def test_get_metadata_field_min_max_boolean_and_string(self, document_store: S3VectorsDocumentStore) -> None:
        """min/max works on booleans (False < True) and on strings (alphabetical), not just numbers."""
        docs = [
            Document(content="Doc 1", meta={"active": True, "category": "Zebra"}),
            Document(content="Doc 2", meta={"active": False, "category": "Alpha"}),
            Document(content="Doc 3", meta={"active": True, "category": "Beta"}),
            Document(content="Doc 4", meta={"active": False, "category": "Gamma"}),
        ]
        document_store.write_documents(docs)

        assert document_store.get_metadata_field_min_max("active") == {"min": False, "max": True}
        assert document_store.get_metadata_field_min_max("category") == {"min": "Alpha", "max": "Zebra"}

    def test_get_metadata_field_min_max_no_values(self, document_store: S3VectorsDocumentStore) -> None:
        """A list-valued field has no comparable scalar, so min/max is None — as for a missing field."""
        docs = [
            Document(content="Doc 1", meta={"tags": ["tag1", "tag2"]}),
            Document(content="Doc 2", meta={"tags": ["tag3", "tag4"]}),
        ]
        document_store.write_documents(docs)

        assert document_store.get_metadata_field_min_max("tags") == {"min": None, "max": None}
        assert document_store.get_metadata_field_min_max("nonexistent") == {"min": None, "max": None}

    def test_get_metadata_field_unique_values_with_lists(self, document_store: S3VectorsDocumentStore) -> None:
        """A list-valued field contributes each item as a distinct value, not the list itself."""
        docs = [
            Document(content="Doc 1", meta={"tags": ["python", "java"]}),
            Document(content="Doc 2", meta={"tags": ["python", "rust"]}),
            Document(content="Doc 3", meta={"tags": ["java", "go"]}),
        ]
        document_store.write_documents(docs)

        values, total = document_store.get_metadata_field_unique_values("tags", size=10)
        assert total == 4
        assert set(values) == {"go", "java", "python", "rust"}

    def test_write_documents_without_embedding_reads_back_as_none(self, document_store: S3VectorsDocumentStore) -> None:
        """
        S3 Vectors stores a placeholder vector for embedding-less documents; `filter_documents`
        must strip it again rather than hand back a vector the caller never supplied.

        `assert_documents_are_equal` ignores embeddings, so this needs its own assertion.
        """
        document_store.write_documents([Document(id="no-emb", content="No embedding here")])

        docs = document_store.filter_documents()
        assert len(docs) == 1
        assert docs[0].content == "No embedding here"
        assert docs[0].embedding is None

    @pytest.mark.skip(reason="S3 Vectors put_vectors is an upsert; only DuplicatePolicy.OVERWRITE is supported")
    def test_write_documents_duplicate_fail(self, document_store: S3VectorsDocumentStore) -> None: ...

    @pytest.mark.skip(reason="S3 Vectors put_vectors is an upsert; only DuplicatePolicy.OVERWRITE is supported")
    def test_write_documents_duplicate_skip(self, document_store: S3VectorsDocumentStore) -> None: ...
