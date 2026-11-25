# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from botocore.exceptions import ClientError
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers import SentenceWindowRetriever
from haystack.dataclasses import ByteStream, Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteAllTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    WriteDocumentsTest,
)
from haystack.utils import Secret

from haystack_integrations.components.retrievers.amazon_s3_vectors import AmazonS3VectorsEmbeddingRetriever
from haystack_integrations.document_stores.amazon_s3_vectors import AmazonS3VectorsDocumentStore


def test_init_is_lazy():
    store = AmazonS3VectorsDocumentStore(
        vector_bucket_name="my-bucket",
        index_name="my-index",
        dimension=768,
    )
    assert store._client is None


def test_init_default_params():
    store = AmazonS3VectorsDocumentStore(
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
    store = AmazonS3VectorsDocumentStore(
        vector_bucket_name="my-bucket",
        index_name="my-index",
        dimension=768,
        distance_metric="euclidean",
        region_name="us-west-2",
        create_bucket_and_index=False,
    )
    d = store.to_dict()
    assert d == {
        "type": "haystack_integrations.document_stores.amazon_s3_vectors.document_store.AmazonS3VectorsDocumentStore",
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
def test_to_dict_with_credentials(mock_boto3, monkeypatch):
    credentials = {}
    for name in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
        monkeypatch.setenv(name.upper(), "test-" + name)
        credentials[name] = Secret.from_env_var(name.upper())
    store = AmazonS3VectorsDocumentStore(
        vector_bucket_name="my-bucket",
        index_name="my-index",
        dimension=768,
        region_name="us-west-2",
        create_bucket_and_index=False,
        **credentials,
    )
    data = store.to_dict()
    for name, secret in credentials.items():
        assert data["init_parameters"][name] == secret.to_dict()
    restored = AmazonS3VectorsDocumentStore.from_dict(data)
    mock_boto3.client.assert_not_called()
    mock_boto3.client.return_value.get_index.return_value = {"index": {"dimension": 768}}
    restored._get_client()
    mock_boto3.client.assert_called_once_with(
        service_name="s3vectors",
        region_name="us-west-2",
        **{name: "test-" + name for name in credentials},
    )


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_from_dict(_mock_boto3):
    data = {
        "type": "haystack_integrations.document_stores.amazon_s3_vectors.document_store.AmazonS3VectorsDocumentStore",
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
    store = AmazonS3VectorsDocumentStore.from_dict(data)
    assert store.vector_bucket_name == "my-bucket"
    assert store.index_name == "my-index"
    assert store.dimension == 768
    assert store.distance_metric == "euclidean"
    assert store.region_name == "us-west-2"
    assert store.create_bucket_and_index is False


@pytest.mark.parametrize("create_bucket_and_index", [True, False])
@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents_missing_embedding_uses_placeholder(mock_boto3, create_bucket_and_index):
    """Every S3 Vectors record needs a vector, so embedding-less documents get the placeholder."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {"index": {"dimension": 4}}
    mock_boto3.client.return_value = client

    store = AmazonS3VectorsDocumentStore(
        vector_bucket_name="b",
        index_name="i",
        dimension=3,
        region_name="us-east-1",
        create_bucket_and_index=create_bucket_and_index,
    )
    docs = [Document(id="1", content="ok", embedding=[0.1] * 4), Document(id="2", content="missing")]
    client.list_vectors.return_value = {
        "vectors": [{"key": "2", "data": {"float32": [-10.0] * 4}, "metadata": {"_content": "missing"}}]
    }
    # Reading first initializes the placeholder using the index dimension, not the constructor's.
    assert store.filter_documents() == [docs[1]]
    assert store.dimension == 4
    assert store.write_documents(docs) == 2

    written = client.put_vectors.call_args.kwargs["vectors"]
    assert written[0]["data"]["float32"] == [0.1] * 4
    # Non-zero: a cosine index rejects zero-norm vectors.
    assert written[1]["data"]["float32"] == [-10.0] * 4
    # The caller's Document is not mutated.
    assert docs[1].embedding is None


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_filter_documents_follows_pagination(mock_boto3):
    client = mock_boto3.client.return_value
    client.get_index.return_value = {"index": {"dimension": 4}}
    client.list_vectors.side_effect = [
        {"vectors": [{"key": "1", "metadata": {"_content": "First"}}], "nextToken": "page-2"},
        {"vectors": [{"key": "2", "metadata": {"_content": "Second"}}]},
    ]
    store = AmazonS3VectorsDocumentStore(
        vector_bucket_name="b", index_name="i", dimension=4, create_bucket_and_index=False
    )
    assert store.filter_documents() == [Document(id="1", content="First"), Document(id="2", content="Second")]
    kwargs = {"vectorBucketName": "b", "indexName": "i", "returnData": True, "returnMetadata": True}
    assert client.list_vectors.call_args_list == [call(**kwargs), call(**kwargs, nextToken="page-2")]


@pytest.mark.parametrize("policy", [DuplicatePolicy.FAIL, DuplicatePolicy.SKIP])
@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents_unsupported_policy_overwrites(mock_boto3, policy):
    """
    put_vectors is an upsert and S3 Vectors has no cheap existence check, so policies other than
    OVERWRITE are ignored instead of paying for an extra read pass.
    """
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {"index": {"dimension": 4}}
    mock_boto3.client.return_value = client

    store = AmazonS3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    assert store.write_documents([Document(id="1", content="Hello", embedding=[0.1] * 4)], policy=policy) == 1
    client.get_vectors.assert_not_called()
    client.put_vectors.assert_called_once()


@pytest.mark.parametrize("fail_second_batch", [False, True])
@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_write_documents_batches(mock_boto3, fail_second_batch):
    client = mock_boto3.client.return_value
    client.get_index.return_value = {"index": {"dimension": 4}}
    if fail_second_batch:
        client.put_vectors.side_effect = [
            {},
            ClientError({"Error": {"Code": "InternalServerException", "Message": "second batch"}}, "PutVectors"),
        ]
    store = AmazonS3VectorsDocumentStore(
        vector_bucket_name="b", index_name="i", dimension=4, create_bucket_and_index=False
    )
    docs = [Document(id=str(i), embedding=[0.1] * 4) for i in range(501)]
    with pytest.raises(ClientError, match="second batch") if fail_second_batch else nullcontext():
        assert store.write_documents(docs) == 501
    batches = [c.kwargs["vectors"] for c in client.put_vectors.call_args_list]
    assert [len(batch) for batch in batches] == [500, 1]
    assert [v["key"] for batch in batches for v in batch] == [doc.id for doc in docs]


@pytest.mark.parametrize("fail_second_batch", [False, True])
@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_delete_documents_batches(mock_boto3, fail_second_batch):
    client = mock_boto3.client.return_value
    client.get_index.return_value = {"index": {"dimension": 4}}
    if fail_second_batch:
        client.delete_vectors.side_effect = [
            {},
            ClientError({"Error": {"Code": "InternalServerException", "Message": "second batch"}}, "DeleteVectors"),
        ]
    store = AmazonS3VectorsDocumentStore(
        vector_bucket_name="b", index_name="i", dimension=4, create_bucket_and_index=False
    )
    ids = [str(i) for i in range(501)]
    with pytest.raises(ClientError, match="second batch") if fail_second_batch else nullcontext():
        store.delete_documents(ids)
    assert client.delete_vectors.call_args_list == [
        call(vectorBucketName="b", indexName="i", keys=ids[:500]),
        call(vectorBucketName="b", indexName="i", keys=ids[500:]),
    ]


def test_write_documents_invalid_input_raises():
    """write_documents must reject non-list / non-Document input before doing anything."""
    store = AmazonS3VectorsDocumentStore(
        vector_bucket_name="b", index_name="i", dimension=4, create_bucket_and_index=False
    )
    with pytest.raises(ValueError):
        store.write_documents("not a list")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.write_documents(["not a document"])  # type: ignore[list-item]


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_embedding_retrieval_score_conversion(mock_boto3):
    """Tests our distance-to-score conversion logic — the only non-trivial transform in retrieval."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {"index": {"dimension": 4}}
    client.query_vectors.return_value = {
        "vectors": [{"key": "1", "distance": 0.05, "metadata": {"_content": "Hello", "category": "news"}}],
        "distanceMetric": "cosine",
    }
    mock_boto3.client.return_value = client

    store = AmazonS3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
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
    client.get_index.return_value = {"index": {"dimension": 4}}
    client.query_vectors.return_value = {
        "vectors": [{"key": "1", "distance": 1.5, "metadata": {}}],
        "distanceMetric": "euclidean",
    }
    mock_boto3.client.return_value = client

    store = AmazonS3VectorsDocumentStore(
        vector_bucket_name="b", index_name="i", dimension=4, distance_metric="euclidean", region_name="us-east-1"
    )
    docs = store._embedding_retrieval(query_embedding=[0.1] * 4, top_k=5)
    assert docs[0].score == pytest.approx(-1.5)  # euclidean: negated


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_embedding_retrieval_passes_filters(mock_boto3):
    """Tests that Haystack filters are converted and passed to query_vectors."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {"index": {"dimension": 4}}
    client.query_vectors.return_value = {"vectors": [], "distanceMetric": "cosine"}
    mock_boto3.client.return_value = client

    store = AmazonS3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    filters = {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": "news"}]}
    store._embedding_retrieval(query_embedding=[0.1] * 4, filters=filters, top_k=5)

    call_args = client.query_vectors.call_args[1]
    assert call_args["filter"] == {"$and": [{"category": {"$eq": "news"}}]}


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_embedding_retrieval_follows_pagination(mock_boto3):
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {"index": {"dimension": 4}}
    client.query_vectors.side_effect = [
        {"vectors": [{"key": "1", "distance": 0.1, "metadata": {}}], "distanceMetric": "cosine", "nextToken": "t"},
        {"vectors": [{"key": "2", "distance": 0.2, "metadata": {}}], "distanceMetric": "cosine"},
    ]
    mock_boto3.client.return_value = client

    store = AmazonS3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    docs = store._embedding_retrieval(query_embedding=[0.1] * 4, top_k=200)

    assert [d.id for d in docs] == ["1", "2"]
    assert client.query_vectors.call_args_list[0][1]["topK"] == 200
    assert client.query_vectors.call_args_list[1][1]["nextToken"] == "t"


def test_embedding_retrieval_empty_embedding_raises():
    """Tests our input validation — no mocking needed."""
    store = AmazonS3VectorsDocumentStore(
        vector_bucket_name="b", index_name="i", dimension=4, create_bucket_and_index=False
    )
    with pytest.raises(ValueError, match="non-empty"):
        store._embedding_retrieval(query_embedding=[])


def test_document_to_s3_vector():
    """Tests our Document → S3 vector conversion (pure function)."""
    doc = Document(
        id="test-1", content="Hello world", embedding=[0.1, 0.2, 0.3], meta={"category": "test", "year": 2024}
    )
    result = AmazonS3VectorsDocumentStore._document_to_s3_vector(doc)
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
    doc = AmazonS3VectorsDocumentStore._s3_vector_to_document(vector)
    assert doc.id == "test-1"
    assert doc.content == "Hello world"
    assert doc.embedding == [0.1, 0.2, 0.3]
    assert doc.meta == {"category": "test"}


def test_document_roundtrip():
    """Tests Document → S3 vector → Document is lossless."""
    doc = Document(
        id="test-1", content="Hello world", embedding=[0.1, 0.2, 0.3], meta={"category": "test", "year": 2024}
    )
    vector = AmazonS3VectorsDocumentStore._document_to_s3_vector(doc)
    restored = AmazonS3VectorsDocumentStore._s3_vector_to_document(vector)
    assert restored.id == doc.id
    assert restored.content == doc.content
    assert restored.embedding == doc.embedding
    assert restored.meta == doc.meta


def test_document_roundtrip_without_embedding():
    """A document written without an embedding must read back without one, not with the placeholder."""
    placeholder = [-10.0] * 3
    doc = Document(id="test-1", content="Hello world", meta={"category": "test"})

    vector = AmazonS3VectorsDocumentStore._document_to_s3_vector(doc, fallback_embedding=placeholder)
    assert vector["data"]["float32"] == placeholder

    restored = AmazonS3VectorsDocumentStore._s3_vector_to_document(vector, placeholder_embedding=placeholder)
    assert restored.embedding is None
    assert restored.content == doc.content
    assert restored.meta == doc.meta


def test_s3_vector_to_document_keeps_embedding_matching_no_placeholder():
    """A real embedding is never mistaken for the placeholder."""
    vector = {"key": "test-1", "data": {"float32": [0.1, 0.2, 0.3]}, "metadata": {}}
    doc = AmazonS3VectorsDocumentStore._s3_vector_to_document(vector, placeholder_embedding=[-10.0] * 3)
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
    assert AmazonS3VectorsDocumentStore._is_supported_metadata_value(value) is supported


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
    result = AmazonS3VectorsDocumentStore._document_to_s3_vector(doc)

    assert result["metadata"] == {"_content": "Hello world", "keep_str": "x", "keep_list": ["a", "b"]}
    assert "['_split_overlap', 'empty', 'nested', 'nothing']" in caplog.text


def test_document_to_s3_vector_converts_tuples_to_lists():
    """A tuple is a valid array for S3 Vectors, but botocore's validator only accepts `list`."""
    doc = Document(id="test-1", meta={"tags": ("a", "b")}, embedding=[0.1])
    result = AmazonS3VectorsDocumentStore._document_to_s3_vector(doc)
    assert result["metadata"]["tags"] == ["a", "b"]


def test_document_to_s3_vector_skips_blob(caplog):
    doc = Document(id="1", content="x", embedding=[0.1], blob=ByteStream(data=b"binary", mime_type="application/pdf"))
    vector = AmazonS3VectorsDocumentStore._document_to_s3_vector(doc)
    assert vector["metadata"] == {"_content": "x"}
    assert "does not store" in caplog.text


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_close_releases_the_client_and_allows_reopen(mock_boto3):
    """close() closes the boto3 client; the store stays usable and lazily builds a new one."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {"index": {"dimension": 4}}
    mock_boto3.client.return_value = client

    store = AmazonS3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
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
    store = AmazonS3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    store.close()
    mock_boto3.client.assert_not_called()


@patch("haystack_integrations.document_stores.amazon_s3_vectors.document_store.boto3")
def test_close_is_exception_safe(mock_boto3):
    """A failure while closing must not propagate to the caller."""
    client = MagicMock()
    client.get_vector_bucket.return_value = {}
    client.get_index.return_value = {"index": {"dimension": 4}}
    client.close.side_effect = RuntimeError("boom")
    mock_boto3.client.return_value = client

    store = AmazonS3VectorsDocumentStore(vector_bucket_name="b", index_name="i", dimension=4, region_name="us-east-1")
    store._get_client()
    store.close()  # must not raise
    assert store._client is None


@pytest.mark.integration
class TestDocumentStore(
    CountDocumentsTest,
    WriteDocumentsTest,
    DeleteDocumentsTest,
    DeleteAllTest,
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
        for r, e in zip(sorted(received, key=lambda d: d.id), sorted(expected, key=lambda d: d.id), strict=True):
            assert replace(r, embedding=None, score=None) == replace(e, embedding=None, score=None)
            assert r.embedding == pytest.approx(e.embedding, abs=1e-5)

    def test_write_documents(self, document_store: AmazonS3VectorsDocumentStore) -> None:
        """
        Default behaviour is OVERWRITE (S3 Vectors `put_vectors` is upsert), so
        writing the same document twice succeeds and returns 1 each time.
        """
        docs = [Document(id="1", content="hello", embedding=[0.1] * 768)]
        assert document_store.write_documents(docs) == 1
        assert document_store.write_documents(docs) == 1

    def test_close_and_reopen(self, document_store: AmazonS3VectorsDocumentStore) -> None:
        """close() drops the boto3 client; the store keeps working against the same index."""
        document_store._get_client()
        assert document_store._client is not None

        document_store.close()
        assert document_store._client is None

        assert document_store.count_documents() == 0
        assert document_store._client is not None

    def test_sentence_window_retriever(self, document_store: AmazonS3VectorsDocumentStore) -> None:
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

        retrieved = AmazonS3VectorsEmbeddingRetriever(document_store=document_store).run(
            query_embedding=[0.1] * 768, top_k=1
        )
        result = SentenceWindowRetriever(document_store=document_store, window_size=2).run(
            retrieved_documents=[retrieved["documents"][0]]
        )
        assert len(result["context_windows"]) == 1

    def test_write_documents_without_embedding_reads_back_as_none(
        self, document_store: AmazonS3VectorsDocumentStore
    ) -> None:
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
    def test_write_documents_duplicate_fail(self, document_store: AmazonS3VectorsDocumentStore) -> None: ...

    @pytest.mark.skip(reason="S3 Vectors put_vectors is an upsert; only DuplicatePolicy.OVERWRITE is supported")
    def test_write_documents_duplicate_skip(self, document_store: AmazonS3VectorsDocumentStore) -> None: ...
