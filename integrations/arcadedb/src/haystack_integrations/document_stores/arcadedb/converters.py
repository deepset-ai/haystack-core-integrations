# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Convert between Haystack Documents and ArcadeDB records."""

from typing import Any

from haystack import Document


def _from_haystack_to_arcadedb(documents: list[Document]) -> list[dict[str, Any]]:
    """Convert Haystack Documents to dicts suitable for ArcadeDB INSERT."""
    records = []
    for doc in documents:
        record: dict[str, Any] = {
            "id": doc.id,
            "content": doc.content,
            "embedding": doc.embedding,
            "meta": doc.meta,
        }
        records.append(record)
    return records


def _from_arcadedb_to_haystack(records: list[dict[str, Any]]) -> list[Document]:
    """Convert ArcadeDB query result rows to Haystack Documents."""
    documents = []
    for record in records:
        doc = Document(
            id=record["id"],
            content=record.get("content"),
            embedding=record.get("embedding"),
            meta=record.get("meta") or {},
            score=record.get("score"),
        )
        documents.append(doc)
    return documents
