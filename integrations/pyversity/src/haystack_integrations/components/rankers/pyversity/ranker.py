# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Haystack integration for `pyversity <https://github.com/Pringled/pyversity>`_.

Wraps pyversity's diversification algorithms as a Haystack ``@component``,
making it easy to drop result diversification into any Haystack pipeline.
"""

from dataclasses import replace
from typing import Any

import numpy as np
from haystack import Document, component, default_from_dict, default_to_dict, logging

from pyversity import Strategy, diversify

logger = logging.getLogger(__name__)


@component
class PyversityRanker:
    """
    Reranks documents using [pyversity](https://github.com/Pringled/pyversity)'s diversification algorithms.

    Balances relevance and diversity in a ranked list of documents. Documents
    must have both `score` and `embedding` populated (e.g. as returned by
    a dense retriever with `return_embedding=True`).

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.rankers.pyversity import PyversityRanker
    from pyversity import Strategy

    ranker = PyversityRanker(top_k=5, strategy=Strategy.MMR, diversity=0.5)

    docs = [
        Document(content="Paris", score=0.9, embedding=[0.1, 0.2]),
        Document(content="Berlin", score=0.8, embedding=[0.3, 0.4]),
    ]
    output = ranker.run(documents=docs)
    docs = output["documents"]
    ```
    """

    def __init__(self, top_k: int | None = None, *, strategy: Strategy = Strategy.DPP, diversity: float = 0.5) -> None:
        """
        Creates an instance of PyversityRanker.

        :param top_k: Number of documents to return after diversification.
            If `None`, all documents are returned in diversified order.
        :param strategy: Pyversity diversification strategy (e.g. `Strategy.MMR`). Defaults to `Strategy.DPP`.
        :param diversity: Trade-off between relevance and diversity in [0, 1].
            `0.0` keeps only the most relevant documents; `1.0` maximises
            diversity regardless of relevance. Defaults to `0.5`.

        :raises ValueError: If `top_k` is not a positive integer or `diversity` is not in [0, 1].
        """
        if top_k is not None and top_k <= 0:
            msg = f"top_k must be a positive integer, got {top_k}"
            raise ValueError(msg)
        if not 0.0 <= diversity <= 1.0:
            msg = f"diversity must be in [0, 1], got {diversity}"
            raise ValueError(msg)
        self.top_k = top_k
        self.strategy = strategy
        self.diversity = diversity

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            top_k=self.top_k,
            strategy=self.strategy.value,
            diversity=self.diversity,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PyversityRanker":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component instance.
        """
        if strategy := data.get("init_parameters", {}).get("strategy"):
            data["init_parameters"]["strategy"] = Strategy(strategy)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        top_k: int | None = None,
        strategy: Strategy | None = None,
        diversity: float | None = None,
    ) -> dict[str, list[Document]]:
        """
        Rerank the list of documents using pyversity's diversification algorithm.

        Documents missing `score` or `embedding` are skipped with a warning.

        :param documents: List of Documents to rerank. Each document must have `score` and `embedding` set.
        :param top_k: Overrides the initialized `top_k` for this call. `None` falls back to the initialized value.
        :param strategy: Overrides the initialized `strategy` for this call. `None` falls back to the initialized value.
        :param diversity: Overrides the initialized `diversity` for this call.
            `None` falls back to the initialized value.
        :returns:
            A dictionary with the following keys:
            - `documents`: List of up to `top_k` reranked Documents, ordered by the diversification algorithm.
        :raises ValueError: If `top_k` is not a positive integer or `diversity` is not in [0, 1].
        """
        if top_k is not None and top_k <= 0:
            msg = f"top_k must be a positive integer, got {top_k}"
            raise ValueError(msg)
        if diversity is not None and not 0.0 <= diversity <= 1.0:
            msg = f"diversity must be in [0, 1], got {diversity}"
            raise ValueError(msg)

        effective_top_k = top_k if top_k is not None else self.top_k
        effective_strategy = strategy if strategy is not None else self.strategy
        effective_diversity = diversity if diversity is not None else self.diversity

        if not documents:
            return {"documents": []}

        valid_docs = [doc for doc in documents if doc.score is not None and doc.embedding is not None]
        skipped = len(documents) - len(valid_docs)
        if skipped:
            logger.warning(
                "{skipped} document(s) are missing 'score' or 'embedding' and will be skipped.",
                skipped=skipped,
            )

        if not valid_docs:
            return {"documents": []}

        embeddings = np.array([doc.embedding for doc in valid_docs])
        scores = np.array([doc.score for doc in valid_docs])

        if effective_top_k is not None:
            k = min(effective_top_k, len(valid_docs))
        else:
            k = len(valid_docs)
        result = diversify(
            embeddings=embeddings,
            scores=scores,
            k=k,
            strategy=effective_strategy,
            diversity=effective_diversity,
        )

        return {
            "documents": [
                replace(valid_docs[i], score=float(score))
                for i, score in zip(result.indices, result.selection_scores, strict=True)  # type: ignore[call-overload]
            ]
        }
