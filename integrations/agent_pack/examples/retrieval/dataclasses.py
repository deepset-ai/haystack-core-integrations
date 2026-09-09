# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from typing import Any

from haystack import Document

from haystack_integrations.agent_pack.evaluation import EvalCase


@dataclass(kw_only=True)
class RetrievalEvalCase(EvalCase):
    """
    One eval case for a retrieval pipeline, which is scored at a rank cutoff.

    :param k: The rank cutoff the eval case is scored at, giving recall@k and precision@k. Only the first `k` returned
        documents count, in the order the run returned them, so a pipeline is measured on what it put at the top
        rather than on how much it returned.
    """

    k: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with a stable document order."""
        data = super().to_dict()
        data["evidence"] = dict(sorted(self.evidence.items()))
        return data


@dataclass(kw_only=True)
class RetrievalEvalCaseMetrics:
    """
    Score for one retrieval eval case.

    :param question: The question that was posed.
    :param passed: Whether the run met every expectation, which is true exactly when `failures` is empty.
    :param score: What quality aggregates over eval cases, which is recall@k rather than whether the eval case
        passed. Recall over a handful of expected documents moves in steps of a half or a third, so a threshold
        on it reports a configuration that went from finding none of the evidence to two thirds of it as no
        change at all.
    :param stage_outputs: How many items each component emitted, by component name and output socket.
    :param failures: Every expectation the run missed, named.
    :param recall_at_k: The share of the expected documents found within the first `k` returned.
    :param precision_at_k: The share of the first `k` returned that were expected.
    :param retrieved: Everything the run returned, which is separate from how deep it was scored: returning more
        than `k` is not a fault, it simply earns nothing for the documents past the cutoff.
    :param missed_document_ids: The expected documents the run did not return, sorted.
    :param queries: What the configuration actually asked the store. A recall failure is usually explained by the
        wording of these rather than by how many there were, which makes them what an optimizer acts on.
    :param latency_ms: Measured wall-clock duration of the run.
    """

    question: str
    passed: bool
    score: float
    stage_outputs: dict[str, dict[str, int]]
    failures: tuple[str, ...]
    recall_at_k: float
    precision_at_k: float
    retrieved: int
    missed_document_ids: tuple[str, ...]
    queries: tuple[str, ...]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        data = asdict(self)
        for key in ("failures", "missed_document_ids", "queries"):
            data[key] = list(getattr(self, key))
        return data


@dataclass
class RetrievalOutcome:
    """What one pipeline run produced, independent of how the pipeline was wired."""

    documents: list[Document] = field(default_factory=list)
    queries: tuple[str, ...] = ()
