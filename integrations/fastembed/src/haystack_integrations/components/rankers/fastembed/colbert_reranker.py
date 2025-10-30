# SPDX-FileCopyrightText: 2025-present deepset GmbH
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from haystack import Document, component

_TWO_D = 2
_VALID_SIMS = {"cosine", "dot"}


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row vector in a 2D array. Safe for zero rows.
    """
    if mat.ndim != _TWO_D:
        msg = f"Expected 2D matrix, got shape {mat.shape}"
        raise ValueError(msg)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def _maxsim_score(
    q_mat: np.ndarray,
    d_mat: np.ndarray,
    *,
    similarity: str = "cosine",
    normalize: bool = True,
) -> float:
    """
    ColBERT late-interaction (MaxSim) score:
    For each query token vector, take the maximum similarity across all doc token vectors, then sum.
    """
    if q_mat.size == 0 or d_mat.size == 0:
        return 0.0
    if similarity not in _VALID_SIMS:
        msg = f"Unsupported similarity '{similarity}'. Use 'cosine' or 'dot'."
        raise ValueError(msg)

    if similarity == "cosine" and normalize:
        q = _l2_normalize_rows(q_mat)
        d = _l2_normalize_rows(d_mat)
    else:
        q = q_mat
        d = d_mat

    sim = q @ d.T  # [Lq, D] @ [D, Ld] -> [Lq, Ld]
    max_per_q = sim.max(axis=1)
    return float(max_per_q.sum())


@component
class FastembedColbertReranker:
    """
    Rerank Documents using ColBERT late-interaction scoring via FastEmbed.

    This component expects a *retrieved* list of Documents (e.g., top 100-500)
    and reorders them by ColBERT MaxSim score with respect to the input query.
    """

    def __init__(
        self,
        model: str = "colbert-ir/colbertv2.0",
        batch_size: int = 16,
        threads: int | None = None,
        similarity: str = "cosine",
        normalize: bool = True,
        max_query_tokens: int | None = None,
        max_doc_tokens: int | None = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.threads = threads
        self.similarity = similarity
        self.normalize = normalize
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens

        if similarity not in _VALID_SIMS:
            msg = f"similarity must be one of {_VALID_SIMS}, got {similarity!r}"
            raise ValueError(msg)
        if batch_size <= 0:
            msg = f"batch_size must be > 0, got {batch_size}"
            raise ValueError(msg)

        self._encoder = None  # LateInteractionTextEmbedding
        self._ready = False

    def warm_up(self):
        if self._ready:
            return
        try:
            from fastembed import LateInteractionTextEmbedding  # type: ignore  # noqa: PLC0415

            kwargs = {"model_name": self.model, "threads": self.threads}
            for k, v in {
                "max_tokens_query": self.max_query_tokens,
                "max_tokens_document": self.max_doc_tokens,
            }.items():
                if v is not None:
                    kwargs[k] = v

            try:
                self._encoder = LateInteractionTextEmbedding(**kwargs)
            except TypeError:
                self._encoder = LateInteractionTextEmbedding(model_name=self.model, threads=self.threads)

            gen_q = self._encoder.query_embed(["warmup"])
            next(gen_q, None)
            gen_d = self._encoder.embed(["warmup"])
            next(gen_d, None)

            self._ready = True
        except ModuleNotFoundError as e:
            msg = (
                "fastembed is not installed. Please install the FastEmbed integration:\n\n"
                "    pip install fastembed-haystack\n"
            )
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"Failed to initialize FastEmbed ColBERT encoder for model '{self.model}': {e}"
            raise RuntimeError(msg) from e

    def _ensure_ready(self):
        if not self._ready:
            self.warm_up()

    @staticmethod
    def _get_texts(documents: Sequence[Document]) -> list[str]:
        return [doc.content or "" for doc in documents]

    def _encode_query(self, text: str) -> np.ndarray:
        if self._encoder is None:
            msg = "Encoder is not initialized. Call warm_up() first."
            raise RuntimeError(msg)
        arr = next(self._encoder.query_embed([text]), None)
        if arr is None:
            return np.zeros((0, 0), dtype=np.float32)
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != _TWO_D:
            a = a.reshape(-1, a.shape[-1])
        return a

    def _encode_docs_batched(self, texts: list[str]) -> list[np.ndarray]:
        if self._encoder is None:
            msg = "Encoder is not initialized. Call warm_up() first."
            raise RuntimeError(msg)
        results: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            for emb in self._encoder.embed(batch):
                if emb is None:
                    results.append(np.zeros((0, 0), dtype=np.float32))
                else:
                    a = np.asarray(emb, dtype=np.float32)
                    if a.ndim != _TWO_D:
                        a = a.reshape(-1, a.shape[-1])
                    results.append(a)
        if len(results) != len(texts):
            msg = f"Encoder returned {len(results)} embeddings for {len(texts)} documents."
            raise RuntimeError(msg)
        return results

    def run(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_ready()

        docs_list = list(documents)
        if not docs_list:
            return {"documents": []}

        if top_k is not None and top_k < 0:
            msg = f"top_k must be >= 0, got {top_k}"
            raise ValueError(msg)

        for i, d in enumerate(docs_list):
            if getattr(d, "meta", None) is None:
                d.meta = {}
            d.meta.setdefault("_orig_idx", i)

        q_mat = self._encode_query(query)
        doc_texts = self._get_texts(docs_list)
        doc_mats = self._encode_docs_batched(doc_texts)

        for d, d_mat in zip(docs_list, doc_mats):
            d.score = _maxsim_score(q_mat, d_mat, similarity=self.similarity, normalize=self.normalize)

        docs_list.sort(
            key=lambda d: (
                d.score if d.score is not None else float("-inf"),
                d.meta.get("_orig_idx", 0),
            ),
            reverse=True,
        )

        for d in docs_list:
            d.meta.pop("_orig_idx", None)

        if top_k is not None:
            docs_list = docs_list[:top_k]

        return {"documents": docs_list}
