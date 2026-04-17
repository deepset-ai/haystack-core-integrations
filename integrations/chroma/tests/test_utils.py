# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack_integrations.document_stores.chroma.errors import ChromaDocumentStoreConfigError
from haystack_integrations.document_stores.chroma.utils import get_embedding_function


def test_get_embedding_function_invalid_name_raises():
    with pytest.raises(ChromaDocumentStoreConfigError, match="Invalid function name"):
        get_embedding_function("NonExistentEmbeddingFunction")
