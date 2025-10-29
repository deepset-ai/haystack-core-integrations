# SPDX-FileCopyrightText: 2025-present deepset GmbH
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

from haystack import Document, component


@component
class FastembedColbertReranker:
    """
    Rerank Documents using ColBERT late-interaction scoring via FastEmbed.

    This component expects a *retrieved* list of Documents (e.g., top 100–500)
    and reorders them by ColBERT MaxSim score with respect to the input query.

    Parameters
    ----------
    model : str, default: "colbert-ir/colbertv2.0"
        The ColBERT-compatible model name to load via FastEmbed.
    batch_size : int, default: 16
        Number of documents to encode per batch.
    threads : Optional[int], default: None
        Number of CPU threads for inference (passed to FastEmbed if supported).
    similarity : {"cosine", "dot"}, default: "cosine"
        Similarity for token–token interactions inside MaxSim.
    normalize : bool, default: True
        L2-normalize token embeddings before similarity (needed for cosine).
    max_query_tokens : Optional[int], default: None
        Truncate/limit tokens on the query side, if desired.
    max_doc_tokens : Optional[int], default: None
        Truncate/limit tokens on the document side, if desired.

    Notes
    -----
    - This is a *reranker*. Use it after a retriever (BM25/dense) with ~100–500 candidates.
    - Lives in the FastEmbed integration to avoid new core dependencies.
    """

    def __init__(
        self,
        model: str = "colbert-ir/colbertv2.0",
        batch_size: int = 16,
        threads: Optional[int] = None,
        similarity: str = "cosine",
        normalize: bool = True,
        max_query_tokens: Optional[int] = None,
        max_doc_tokens: Optional[int] = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.threads = threads
        self.similarity = similarity
        self.normalize = normalize
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens

        # Lazy-loaded in warm_up()
        self._q_encoder = None  # type: ignore[var-annotated]
        self._d_encoder = None  # type: ignore[var-annotated]
        self._ready = False

    def warm_up(self):
        """
        Load FastEmbed encoders and do a tiny dry run to JIT/initialize backends.

        We keep imports here to avoid importing fastembed unless needed.
        """
        if self._ready:
            return

        try:
            # NOTE: We'll wire these real classes in Step 3 when we implement encoding.
            # from fastembed import LateInteractionTextEmbedding
            # self._q_encoder = LateInteractionTextEmbedding(self.model, mode="query", threads=self.threads)
            # self._d_encoder = LateInteractionTextEmbedding(self.model, mode="document", threads=self.threads)

            # Minimal stub for now:
            self._q_encoder = object()
            self._d_encoder = object()

            # Optional: perform a tiny no-op dry-run later once encoders are real.
            self._ready = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize FastEmbed ColBERT encoders for model '{self.model}': {e}"
            ) from e

    def _ensure_ready(self):
        if not self._ready:
            # Allow implicit warm-up on first run for ergonomics
            self.warm_up()

    def _get_texts(self, documents: Sequence[Document]) -> List[str]:
        # Prefer Document.content; fall back to empty string to avoid crashes
        return [doc.content or "" for doc in documents]

    def run(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Rerank the input documents with respect to the query using ColBERT MaxSim.

        Parameters
        ----------
        query : str
            The user query.
        documents : Sequence[Document]
            Candidate documents to rerank (typically ~100–500).
        top_k : Optional[int]
            If given, only the top-k reranked documents are returned.

        Returns
        -------
        dict
            {"documents": List[Document]} with `Document.score` set to the ColBERT score.
        """
        self._ensure_ready()

        if not documents:
            return {"documents": []}

        # === Placeholder logic (wire real scoring in Step 3) ===
        # For now we keep the input order and set a dummy score (0.0).
        # Next step will replace this with actual FastEmbed + MaxSim scoring.
        reranked = list(documents)
        for d in reranked:
            d.score = 0.0

        if top_k is not None:
            reranked = reranked[:top_k]

        return {"documents": reranked}
