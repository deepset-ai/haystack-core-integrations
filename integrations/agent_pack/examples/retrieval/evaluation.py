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


@dataclass(frozen=True, kw_only=True)
class RetrievalEvalCaseMetrics:
    """
    Score for one retrieval eval case.

    `queries` is the evidence an optimizer acts on: it is what the configuration actually asked the store, and a
    recall failure is usually explained by the wording of those queries rather than by the number of them.

    `score` is what quality aggregates, and it is recall@k rather than whether the eval case passed. Recall over a
    handful of expected documents moves in steps of a half or a third, so a threshold on it reports a
    configuration that went from finding none of the evidence to finding two thirds of it as no change at all.
    `retrieved` counts everything the run returned, which is separate from how deep it was scored: returning more
    than `k` is not a fault, it simply earns nothing for the documents past the cutoff.
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


def score_retrieval_result(
    outcome: RetrievalOutcome,
    eval_case: RetrievalEvalCase,
    *,
    latency_ms: float,
    stage_outputs: dict[str, dict[str, int]] | None = None,
) -> RetrievalEvalCaseMetrics:
    """
    Score one retrieval run against its labelled evidence.

    :param outcome: The documents the pipeline retrieved and the queries it issued.
    :param eval_case: The expectations to score against.
    :param latency_ms: Measured wall-clock duration of the run.
    :param stage_outputs: How many items each component emitted, by component name and output socket.
    :returns: The score, naming every expectation the run missed.
    """
    # Deduplicated in the order the run returned them, since which documents fall past the cutoff depends on
    # how it ranked them.
    returned_ids = list(dict.fromkeys(document.id for document in outcome.documents))
    scored_ids = set(returned_ids[: eval_case.k] if eval_case.k is not None else returned_ids)
    matched = scored_ids & eval_case.expected_document_ids
    recall_at_k = len(matched) / len(eval_case.expected_document_ids)
    precision_at_k = len(matched) / len(scored_ids) if scored_ids else 0.0

    failures: list[str] = []
    if recall_at_k < eval_case.min_recall:
        failures.append(f"recall_below_{eval_case.min_recall:g}")
    if precision_at_k < eval_case.min_precision:
        failures.append(f"precision_below_{eval_case.min_precision:g}")

    return RetrievalEvalCaseMetrics(
        question=eval_case.question,
        passed=not failures,
        stage_outputs=stage_outputs or {},
        score=recall_at_k,
        failures=tuple(failures),
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        retrieved=len(returned_ids),
        missed_document_ids=tuple(sorted(eval_case.expected_document_ids - matched)),
        queries=outcome.queries,
        latency_ms=latency_ms,
    )
