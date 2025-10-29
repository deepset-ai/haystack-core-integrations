# SPDX-FileCopyrightText: 2025-present deepset GmbH
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

import math
import numpy as np
from haystack import Document, component


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row vector in a 2D array. Safe for zero rows.
    """
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {mat.shape}")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # avoid division by zero
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def _maxsim_score(q_mat: np.ndarray, d_mat: np.ndarray, *, similarity: str = "cosine", normalize: bool = True) -> float:
    """
    ColBERT late-interaction (MaxSim) score.

    For each query token vector, take the maximum similarity across all doc token vectors,
    then sum over query tokens.

    Similarity is either cosine (row-wise normalized dot) or raw dot.
    """
    if q_mat.size == 0 or d_mat.size == 0:
        return 0.0

    if similarity not in ("cosine", "dot"):
        raise ValueError(f"Unsupported similarity '{similarity}'. Use 'cosine' or 'dot'.")

    if similarity == "cosine" and normalize:
        q = _l2_normalize_rows(q_mat)
        d = _l2_normalize_rows(d_mat)
    else:
        q = q_mat
        d = d_mat

    # [Lq, D] x [D, Ld] -> [Lq, Ld]
    sim = q @ d.T  # numpy matmul

    # Max over doc tokens for each query token, then sum
    # Guard against empty axis when Ld==0 (handled above by size check)
    max_per_q = sim.max(axis=1)
    score = float(max_per_q.sum())
    return score


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
        Truncate/limit tokens on the query side, if supported by the encoder.
    max_doc_tokens : Optional[int], default: None
        Truncate/limit tokens on the document side, if supported by the encoder.

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
        self._encoder = None  # LateInteractionTextEmbedding
        self._ready = False

    def warm_up(self):
        """
        Load FastEmbed encoders and do a tiny dry run to initialize backends.
        """
        if self._ready:
            return

        try:
            # Lazy import to avoid hard dependency outside this integration
            from fastembed import LateInteractionTextEmbedding  # type: ignore

            # LateInteractionTextEmbedding exposes .query_embed() and .embed() generators
            # Some fastembed versions use 'model_name' kw, others accept positional.
            # We'll pass by name for clarity.
            self._encoder = LateInteractionTextEmbedding(
                model_name=self.model,
                threads=self.threads,
                # Some fastembed versions accept kwargs like max_tokens; if not, they're ignored safely.
                max_tokens_query=self.max_query_tokens,     # optional / best-effort
                max_tokens_document=self.max_doc_tokens,    # optional / best-effort
            )

            # Tiny dry-run to trigger onnx initialization (doesn't fail if offline)
            _ = next(self._encoder.query_embed(["warmup"]), None)
            _ = next(self._encoder.embed(["warmup"]), None)

            self._ready = True
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "fastembed is not installed. Please install the FastEmbed integration:\n\n"
                "    pip install fastembed-haystack\n"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize FastEmbed ColBERT encoder for model '{self.model}': {e}"
            ) from e

    def _ensure_ready(self):
        if not self._ready:
            self.warm_up()

    @staticmethod
    def _get_texts(documents: Sequence[Document]) -> List[str]:
        # Prefer Document.content; fall back to empty string to avoid crashes
        return [doc.content or "" for doc in documents]

    def _encode_query(self, text: str) -> np.ndarray:
        """
        Encode a single query into a [Lq, D] numpy array of token embeddings.
        """
        assert self._encoder is not None
        # .query_embed returns an iterator/generator over embeddings
        gen = self._encoder.query_embed([text])
        arr = next(gen, None)
        if arr is None:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(arr, dtype=np.float32)

    def _encode_docs_batched(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode documents into token-embedding matrices, batched.
        Returns a list of [Ld, D] arrays aligned with `texts`.
        """
        assert self._encoder is not None

        results: List[np.ndarray] = []
        n = len(texts)
        if n == 0:
            return results

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = texts[start:end]
            # .embed returns a generator yielding one embedding per input string
            for emb in self._encoder.embed(batch):
                if emb is None:
                    results.append(np.zeros((0, 0), dtype=np.float32))
                else:
                    results.append(np.asarray(emb, dtype=np.float32))

        # Safety: ensure alignment
        if len(results) != n:
            raise RuntimeError(
                f"Encoder returned {len(results)} embeddings for {n} documents; batch logic out of sync."
            )
        return results

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

        docs_list = list(documents)
        if not docs_list:
            return {"documents": []}

        # Encode query once
        q_mat = self._encode_query(query)

        # Encode documents (batched)
        doc_texts = self._get_texts(docs_list)
        doc_mats = self._encode_docs_batched(doc_texts)

        # Compute scores
        scores: List[float] = []
        for d_mat in doc_mats:
            score = _maxsim_score(q_mat, d_mat, similarity=self.similarity, normalize=self.normalize)
            scores.append(score)

        # Attach and sort (descending)
        for d, s in zip(docs_list, scores):
            d.score = s

        docs_list.sort(key=lambda d: (d.score if d.score is not None else float("-inf")), reverse=True)

        # Slice top_k if requested
        if top_k is not None:
            docs_list = docs_list[:top_k]

        return {"documents": docs_list}