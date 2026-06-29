# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import Document

from haystack_integrations.document_stores.arcadedb.converters import (
    _from_arcadedb_to_haystack,
    _from_haystack_to_arcadedb,
)


def test_from_haystack_to_arcadedb():
    docs = [Document(content="hello", meta={"key": "val"}, embedding=[0.1, 0.2])]
    records = _from_haystack_to_arcadedb(docs)
    assert len(records) == 1
    assert records[0]["id"] == docs[0].id
    assert records[0]["content"] == "hello"
    assert records[0]["embedding"] == [0.1, 0.2]
    assert records[0]["meta"] == {"key": "val"}


def test_from_arcadedb_to_haystack_full():
    records = [{"id": "abc", "content": "world", "embedding": [0.3], "meta": {"x": 1}, "score": 0.9}]
    docs = _from_arcadedb_to_haystack(records)
    assert len(docs) == 1
    assert docs[0].id == "abc"
    assert docs[0].content == "world"
    assert docs[0].embedding == [0.3]
    assert docs[0].meta == {"x": 1}
    assert docs[0].score == 0.9


def test_from_arcadedb_to_haystack_missing_optional_fields():
    records = [{"id": "xyz"}]
    docs = _from_arcadedb_to_haystack(records)
    assert len(docs) == 1
    assert docs[0].id == "xyz"
    assert docs[0].content is None
    assert docs[0].embedding is None
    assert docs[0].meta == {}
    assert docs[0].score is None


def test_from_arcadedb_to_haystack_null_meta_becomes_empty_dict():
    records = [{"id": "m", "meta": None}]
    docs = _from_arcadedb_to_haystack(records)
    assert docs[0].meta == {}
