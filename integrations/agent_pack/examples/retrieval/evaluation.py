# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from typing import Any

from haystack import Document


@dataclass(frozen=True, kw_only=True)
class RetrievalEvaluationCase:
    """
    Retrieval expectations for one labelled question, forming one eval case.

    :param question: The query to pose to the pipeline.
    :param expected_document_ids: The documents the answer needs. Recall is measured against these.
    :param min_recall: Minimum share of `expected_document_ids` that must be retrieved.
    :param min_precision: Minimum share of retrieved documents that must be expected. Left at 0 by default,
        because a pipeline that widens its candidate set on purpose is not thereby worse; raise it to make
        over-retrieval cost something.
    :param k: The rank cutoff the case is scored at, giving recall@k and precision@k. Only the first `k` returned
        documents count, in the order the run returned them, so a pipeline is measured on what it put at the top
        rather than on how much it returned.
    """

    question: str
    expected_document_ids: frozenset[str]
    min_recall: float = 1.0
    min_precision: float = 0.0
    k: int | None = None

    def __post_init__(self) -> None:
        """Require ground truth to score against."""
        if not self.expected_document_ids:
            msg = f"Case {self.question!r} needs expected document IDs."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with a stable document-ID order."""
        data = asdict(self)
        data["expected_document_ids"] = sorted(self.expected_document_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalEvaluationCase":
        """
        Create a case from its serialized representation.

        :param data: The dictionary to build the case from.
        :returns: The created case.
        """
        arguments = dict(data)
        arguments["expected_document_ids"] = frozenset(arguments.get("expected_document_ids") or ())
        return cls(**arguments)


@dataclass(frozen=True, kw_only=True)
class RetrievalCaseMetrics:
    """
    Score for one retrieval eval case.

    `queries` is the evidence an optimizer acts on: it is what the configuration actually asked the store, and a
    recall failure is usually explained by the wording of those queries rather than by the number of them.

    `score` is what quality aggregates, and it is recall@k rather than whether the case passed. Recall over a
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
    case: RetrievalEvaluationCase,
    *,
    latency_ms: float,
    stage_outputs: dict[str, dict[str, int]] | None = None,
) -> RetrievalCaseMetrics:
    """
    Score one retrieval run against its labelled evidence.

    :param outcome: The documents the pipeline retrieved and the queries it issued.
    :param case: The expectations to score against.
    :param latency_ms: Measured wall-clock duration of the run.
    :param stage_outputs: How many items each component emitted, by component name and output socket.
    :returns: The score, naming every expectation the run missed.
    """
    # Deduplicated in the order the run returned them, since which documents fall past the cutoff depends on
    # how it ranked them.
    returned_ids = list(dict.fromkeys(document.id for document in outcome.documents))
    scored_ids = set(returned_ids[: case.k] if case.k is not None else returned_ids)
    matched = scored_ids & case.expected_document_ids
    recall_at_k = len(matched) / len(case.expected_document_ids)
    precision_at_k = len(matched) / len(scored_ids) if scored_ids else 0.0

    failures: list[str] = []
    if recall_at_k < case.min_recall:
        failures.append(f"recall_below_{case.min_recall:g}")
    if precision_at_k < case.min_precision:
        failures.append(f"precision_below_{case.min_precision:g}")

    return RetrievalCaseMetrics(
        question=case.question,
        passed=not failures,
        stage_outputs=stage_outputs or {},
        score=recall_at_k,
        failures=tuple(failures),
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        retrieved=len(returned_ids),
        missed_document_ids=tuple(sorted(case.expected_document_ids - matched)),
        queries=outcome.queries,
        latency_ms=latency_ms,
    )
