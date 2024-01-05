# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import numpy as np
import pytest
from haystack import Document
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest


from pgvector_haystack.document_store import PGvectorDocumentStore


class TestDocumentStore(CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest):

    @pytest.fixture
    def docstore(self) -> PGvectorDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        return PGvectorDocumentStore() # FIXME
