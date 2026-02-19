# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Haystack integration for `pyversity <https://github.com/Pringled/pyversity>`_.

Wraps pyversity's diversification algorithms as a Haystack ``@component``,
making it easy to drop result diversification into any Haystack pipeline.
"""

import logging

import numpy as np
from haystack import Document, component

from pyversity import Strategy, diversify

logger = logging.getLogger(__name__)


@component
class PyversityReranker:
    """Rerank documents using pyversity's diversification algorithms.

    Balances relevance and diversity in a ranked list of documents. Documents
    must have both ``score`` and ``embedding`` populated (e.g. as returned by
    a dense retriever with ``return_embedding=True``).

    Args:
        k: Number of documents to return after diversification.
        strategy: Pyversity diversification strategy (e.g. ``Strategy.MMR``).
        diversity: Trade-off between relevance and diversity in [0, 1].
            ``0.0`` keeps only the most relevant documents; ``1.0`` maximises
            diversity regardless of relevance. Defaults to ``0.5``.
    """

    def __init__(self, k: int, *, strategy: Strategy = Strategy.DPP, diversity: float = 0.5) -> None:
        if k <= 0:
            msg = f"k must be a positive integer, got {k}"
            raise ValueError(msg)
        if not 0.0 <= diversity <= 1.0:
            msg = f"diversity must be in [0, 1], got {diversity}"
            raise ValueError(msg)
        self._k = k
        self._strategy = strategy
        self._diversity = diversity

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict:
        if not documents:
            return {"documents": []}

        valid_docs = [doc for doc in documents if doc.score is not None and doc.embedding is not None]
        skipped = len(documents) - len(valid_docs)
        if skipped:
            logger.warning(
                "%d document(s) are missing 'score' or 'embedding' and will be skipped.",
                skipped,
            )

        if not valid_docs:
            return {"documents": []}

        embeddings = np.array([doc.embedding for doc in valid_docs])
        scores = np.array([doc.score for doc in valid_docs])

        result = diversify(
            embeddings=embeddings,
            scores=scores,
            k=min(self._k, len(valid_docs)),
            strategy=self._strategy,
            diversity=self._diversity,
        )

        return {"documents": [valid_docs[i] for i in result.indices]}
