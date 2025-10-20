from __future__ import annotations
from unittest.mock import patch
from haystack.dataclasses import Document
from haystack.utils.auth import Secret
from haystack_integrations.components.embedders.isaacus.kanon2 import (
    Kanon2TextEmbedder, Kanon2DocumentEmbedder,
)


def _fake_post(*_args, **kwargs):
    class _Resp:
        def raise_for_status(self):
            return None
        def json(self):
            texts = kwargs.get("json", {}).get("texts", [])
            # Return a deterministic 4-dim vector for each input
            return {"embeddings": [{"embedding": [float(len(t))] * 4} for t in texts]}
    return _Resp()


def test_text_embedder_runs_and_returns_vector():
    with patch("requests.post", _fake_post):
        emb = Kanon2TextEmbedder(api_key=Secret.from_token("x"))
        out = emb.run("hello")
        assert "embedding" in out
        assert isinstance(out["embedding"], list)
        assert len(out["embedding"]) == 4


def test_document_embedder_sets_embeddings_on_documents():
    with patch("requests.post", _fake_post):
        docs = [Document(content="a"), Document(content="bb"), Document(content="")]
        emb = Kanon2DocumentEmbedder(api_key=Secret.from_token("x"), batch_size=2)
        out = emb.run(docs)
        docs2 = out["documents"]

        # First two documents should have non-empty embeddings
        assert isinstance(docs2[0].embedding, list) and len(docs2[0].embedding) == 4
        assert isinstance(docs2[1].embedding, list) and len(docs2[1].embedding) == 4

        # Empty-content doc should not get an embedding written
        # (Document has an 'embedding' field by default; it should remain None/falsy)
        assert not docs2[2].embedding  # passes if None or []
