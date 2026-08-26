# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy


class SolrDocumentStoreTestMixin:
    """Assertions and overrides shared by the sync and async Solr document store test suites."""

    @staticmethod
    def assert_documents_are_equal(received: list[Document], expected: list[Document]) -> None:
        """
        Compare two lists of documents, tolerating what a Solr round-trip legitimately changes.

        Solr stores dense vectors as float32, so an embedding never comes back bit-identical to the
        float64 list that went in and has to be compared approximately. Scores are dropped because
        `filter_documents` does not request them.
        """
        assert len(received) == len(expected)

        received_sorted = sorted(received, key=lambda document: document.id)
        expected_sorted = sorted(expected, key=lambda document: document.id)

        for received_document, expected_document in zip(received_sorted, expected_sorted, strict=True):
            assert received_document.id == expected_document.id
            if expected_document.embedding is None:
                assert received_document.embedding is None
            else:
                assert received_document.embedding == pytest.approx(expected_document.embedding, abs=1e-5)

            # Compare everything else with the fields Solr is allowed to change nulled out.
            received_rest = {
                **received_document.to_dict(flatten=False),
                "embedding": None,
                "score": None,
            }
            expected_rest = {
                **expected_document.to_dict(flatten=False),
                "embedding": None,
                "score": None,
            }
            assert received_rest == expected_rest

    def test_write_documents(self, document_store) -> None:
        """`DuplicatePolicy.NONE` resolves to `FAIL`, so writing the same id twice raises."""
        documents = [Document(id="1", content="test doc")]
        assert document_store.write_documents(documents) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(documents, DuplicatePolicy.NONE)
