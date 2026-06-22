# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.embedders.twelvelabs import TwelveLabsDocumentEmbedder

PATCH_TARGET = "haystack_integrations.components.embedders.twelvelabs.document_embedder.embed_text"


def test_to_dict_and_from_dict(monkeypatch):
    monkeypatch.setenv("TWELVELABS_API_KEY", "tlk_env")
    data = TwelveLabsDocumentEmbedder().to_dict()
    assert data["type"].endswith("TwelveLabsDocumentEmbedder")
    restored = TwelveLabsDocumentEmbedder.from_dict(data)
    assert restored.model == "marengo3.0"


def test_run_sets_embeddings():
    embedder = TwelveLabsDocumentEmbedder(api_key=Secret.from_token("tlk_test"))
    docs = [Document(content="a cat playing piano"), Document(content="a dog")]
    with patch(PATCH_TARGET, return_value=[0.2] * 8) as mock_embed:
        result = embedder.run(documents=docs)
    assert mock_embed.call_count == 2
    assert all(d.embedding == [0.2] * 8 for d in result["documents"])
    assert result["meta"]["model"] == "marengo3.0"


def test_run_rejects_non_documents():
    embedder = TwelveLabsDocumentEmbedder(api_key=Secret.from_token("tlk_test"))
    with pytest.raises(TypeError):
        embedder.run(documents="a string")
