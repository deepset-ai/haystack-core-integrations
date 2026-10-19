# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses import Document


class FalkorDBDocumentStoreTestMixin:
    """Shared assertion helpers for FalkorDB document store tests."""

    @staticmethod
    def assert_documents_are_equal(received: list[Document], expected: list[Document]) -> None:
        """Compare documents while allowing for FalkorDB's float32 embeddings."""
        assert len(received) == len(expected), f"Expected {len(expected)} documents but got {len(received)}"
        received_sorted = sorted(received, key=lambda document: document.id)
        expected_sorted = sorted(expected, key=lambda document: document.id)
        for received_document, expected_document in zip(received_sorted, expected_sorted, strict=True):
            assert received_document.id == expected_document.id
            assert received_document.content == expected_document.content
            assert received_document.meta == expected_document.meta
            assert (received_document.embedding is None) == (expected_document.embedding is None)
