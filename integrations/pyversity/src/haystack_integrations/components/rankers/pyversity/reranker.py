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
    """
    Reranks documents using [pyversity](https://github.com/Pringled/pyversity)'s diversification algorithms.

    Balances relevance and diversity in a ranked list of documents. Documents
    must have both `score` and `embedding` populated (e.g. as returned by
    a dense retriever with `return_embedding=True`).

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.rankers.pyversity import PyversityReranker
    from pyversity import Strategy

    ranker = PyversityReranker(k=5, strategy=Strategy.MMR, diversity=0.5)

    docs = [
        Document(content="Paris", score=0.9, embedding=[0.1, 0.2]),
        Document(content="Berlin", score=0.8, embedding=[0.3, 0.4]),
    ]
    output = ranker.run(documents=docs)
    docs = output["documents"]
    ```
    """

    def __init__(self, k: int, *, strategy: Strategy = Strategy.DPP, diversity: float = 0.5) -> None:
        """
        Creates an instance of PyversityReranker.

        :param k: Number of documents to return after diversification.
        :param strategy: Pyversity diversification strategy (e.g. `Strategy.MMR`). Defaults to `Strategy.DPP`.
        :param diversity: Trade-off between relevance and diversity in [0, 1].
            `0.0` keeps only the most relevant documents; `1.0` maximises
            diversity regardless of relevance. Defaults to `0.5`.

        :raises ValueError: If `k` is not a positive integer or `diversity` is not in [0, 1].
        """
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
        """
        Rerank the list of documents using pyversity's diversification algorithm.

        Documents missing `score` or `embedding` are skipped with a warning.

        :param documents: List of Documents to rerank. Each document must have `score` and `embedding` set.
        :returns:
            A dictionary with the following keys:
            - `documents`: List of up to `k` reranked Documents, ordered by the diversification algorithm.
        """
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
