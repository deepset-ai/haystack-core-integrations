from haystack.dataclasses import ByteStream, Document
from numpy import array

from haystack_integrations.document_stores.pgvector.converters import (
    _from_haystack_to_pg_documents,
    _from_pg_to_haystack_documents,
)


def test_from_haystack_to_pg_documents():
    haystack_docs = [
        Document(
            id="1",
            content="This is a text",
            meta={"meta_key": "meta_value"},
            embedding=[0.1, 0.2, 0.3],
            score=0.5,
        ),
        Document(
            id="2",
            content="This is another text",
            meta={"meta_key": "meta_value"},
            embedding=[0.4, 0.5, 0.6],
            score=0.6,
        ),
        Document(
            id="3",
            blob=ByteStream(b"test", meta={"blob_meta_key": "blob_meta_value"}, mime_type="mime_type"),
            meta={"meta_key": "meta_value"},
            embedding=[0.7, 0.8, 0.9],
            score=0.7,
        ),
        Document(
            id="4",
            content="This is another text\x00",
            meta={"meta_key": "meta_value"},
            embedding=[0.7, 0.8, 0.9],
            score=0.8,
        ),
    ]

    pg_docs = _from_haystack_to_pg_documents(haystack_docs)

    assert pg_docs[0]["id"] == "1"
    assert pg_docs[0]["content"] == "This is a text"
    assert pg_docs[0]["blob_data"] is None
    assert pg_docs[0]["blob_meta"] is None
    assert pg_docs[0]["blob_mime_type"] is None
    assert "dataframe" not in pg_docs[0]
    assert pg_docs[0]["meta"].obj == {"meta_key": "meta_value"}
    assert pg_docs[0]["embedding"] == [0.1, 0.2, 0.3]
    assert "score" not in pg_docs[0]

    assert pg_docs[1]["id"] == "2"
    assert pg_docs[1]["content"] == "This is another text"
    assert pg_docs[1]["blob_data"] is None
    assert pg_docs[1]["blob_meta"] is None
    assert pg_docs[1]["blob_mime_type"] is None
    assert "dataframe" not in pg_docs[1]
    assert pg_docs[1]["meta"].obj == {"meta_key": "meta_value"}
    assert pg_docs[1]["embedding"] == [0.4, 0.5, 0.6]
    assert "score" not in pg_docs[1]

    assert pg_docs[2]["id"] == "3"
    assert pg_docs[2]["content"] is None
    assert pg_docs[2]["blob_data"] == b"test"
    assert pg_docs[2]["blob_meta"].obj == {"blob_meta_key": "blob_meta_value"}
    assert pg_docs[2]["blob_mime_type"] == "mime_type"
    assert "dataframe" not in pg_docs[2]
    assert pg_docs[2]["meta"].obj == {"meta_key": "meta_value"}
    assert pg_docs[2]["embedding"] == [0.7, 0.8, 0.9]
    assert "score" not in pg_docs[2]

    assert pg_docs[3]["id"] == "4"
    assert pg_docs[3]["content"] == "This is another text"
    assert pg_docs[3]["blob_data"] is None
    assert pg_docs[3]["blob_meta"] is None
    assert pg_docs[3]["blob_mime_type"] is None
    assert "dataframe" not in pg_docs[3]
    assert pg_docs[3]["meta"].obj == {"meta_key": "meta_value"}
    assert pg_docs[3]["embedding"] == [0.7, 0.8, 0.9]
    assert "score" not in pg_docs[3]


def test_from_pg_to_haystack_documents():
    pg_docs = [
        {
            "id": "1",
            "content": "This is a text",
            "blob_data": None,
            "blob_meta": None,
            "blob_mime_type": None,
            "meta": {"meta_key": "meta_value"},
            "embedding": array([0.1, 0.2, 0.3]),
        },
        {
            "id": "2",
            "content": "This is another text",
            "blob_data": None,
            "blob_meta": None,
            "blob_mime_type": None,
            "meta": {"meta_key": "meta_value"},
            "embedding": array([0.4, 0.5, 0.6]),
        },
        {
            "id": "3",
            "content": None,
            "blob_data": b"test",
            "blob_meta": {"blob_meta_key": "blob_meta_value"},
            "blob_mime_type": "mime_type",
            "meta": {"meta_key": "meta_value"},
            "embedding": array([0.7, 0.8, 0.9]),
        },
    ]

    haystack_docs = _from_pg_to_haystack_documents(pg_docs)

    assert haystack_docs[0].id == "1"
    assert haystack_docs[0].content == "This is a text"
    assert haystack_docs[0].blob is None
    assert haystack_docs[0].meta == {"meta_key": "meta_value"}
    assert haystack_docs[0].embedding == [0.1, 0.2, 0.3]
    assert haystack_docs[0].score is None

    assert haystack_docs[1].id == "2"
    assert haystack_docs[1].content == "This is another text"
    assert haystack_docs[1].blob is None
    assert haystack_docs[1].meta == {"meta_key": "meta_value"}
    assert haystack_docs[1].embedding == [0.4, 0.5, 0.6]
    assert haystack_docs[1].score is None

    assert haystack_docs[2].id == "3"
    assert haystack_docs[2].content is None
    assert haystack_docs[2].blob.data == b"test"
    assert haystack_docs[2].blob.meta == {"blob_meta_key": "blob_meta_value"}
    assert haystack_docs[2].blob.mime_type == "mime_type"
    assert haystack_docs[2].meta == {"meta_key": "meta_value"}
    assert haystack_docs[2].embedding == [0.7, 0.8, 0.9]
    assert haystack_docs[2].score is None
