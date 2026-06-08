# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from haystack.dataclasses.document import Document
from haystack.testing.document_store import FilterDocumentsTest


@pytest.mark.integration
class TestFilters(FilterDocumentsTest):
    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        """
        Override to handle floating-point precision differences when embeddings are retrieved from PostgreSQL.
        """
        assert len(received) == len(expected)
        received.sort(key=lambda x: x.id)
        expected.sort(key=lambda x: x.id)
        for received_doc, expected_doc in zip(received, expected, strict=True):
            if received_doc.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert received_doc.embedding == pytest.approx(expected_doc.embedding)

            assert replace(received_doc, embedding=None) == replace(expected_doc, embedding=None)

    @pytest.mark.skip(reason="NOT operator is not supported in PgvectorDocumentStore")
    def test_not_operator(self, document_store, filterable_docs): ...
