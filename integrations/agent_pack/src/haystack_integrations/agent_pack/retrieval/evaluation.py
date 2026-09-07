# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation primitives for one-shot retrieval pipelines.

A retrieval pipeline is scored on what it retrieved, not on what it said about it. That is what makes it cheap to
optimize: the labelled evidence already names the documents an answer needs, so no answer has to be generated to
find out whether they were found.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

from haystack import Document


@dataclass(frozen=True, kw_only=True)
class RetrievalEvaluationCase:
    """
    Retrieval expectations for one labelled question.

    :param question: The query to pose to the pipeline.
    :param expected_document_ids: The documents the answer needs. Recall is measured against these.
    :param min_recall: Minimum share of `expected_document_ids` that must be retrieved.
    :param min_precision: Minimum share of retrieved documents that must be expected. Left at 0 by default,
        because a pipeline that widens its candidate set on purpose is not thereby worse; raise it to make
        over-retrieval cost something.
    :param max_queries: Optional cap on how many queries the pipeline may issue for one question. Expanding a
        query buys recall with model calls, and without a cap the cheapest way to pass every case is to expand
        without limit.
    :param max_retrieved: Optional cap on how many documents the pipeline may return for one question. Recall
        alone has a degenerate optimum: a pipeline that returns most of the corpus reaches it, and measuring a
        retrieval pipeline that generates no answer cannot see the cost of doing so, because nothing downstream
        reads the documents. This cap is what makes the size of the answer set matter. It applies to what the
        pipeline returns rather than to what it considers, so retrieving widely and then ranking the result down
        satisfies it while retrieving widely alone does not.
    """

    question: str
    expected_document_ids: frozenset[str]
    min_recall: float = 1.0
    min_precision: float = 0.0
    max_queries: int | None = None
    max_retrieved: int | None = None

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
    Score for one retrieval case.

    `queries` is the evidence an optimizer acts on: it is what the configuration actually asked the store, and a
    recall failure is usually explained by the wording of those queries rather than by the number of them.

    `score` is what quality aggregates, and it is the case's recall rather than whether it passed. Recall over a
    handful of expected documents moves in steps of a half or a third, so a threshold on it reports a
    configuration that went from finding none of the evidence to finding two thirds of it as no change at all.
    A case that broke one of its budgets scores nothing, because a budget is a constraint on the answer rather
    than a matter of degree, and partial credit for exceeding one would restore the incentive it exists to remove.
    """

    question: str
    passed: bool
    score: float
    failures: tuple[str, ...]
    recall: float
    precision: float
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
    outcome: RetrievalOutcome, case: RetrievalEvaluationCase, *, latency_ms: float
) -> RetrievalCaseMetrics:
    """
    Score one retrieval run against its labelled evidence.

    :param outcome: The documents the pipeline retrieved and the queries it issued.
    :param case: The expectations to score against.
    :param latency_ms: Measured wall-clock duration of the run.
    :returns: The score, naming every expectation the run missed.
    """
    retrieved_ids = {document.id for document in outcome.documents}
    matched = retrieved_ids & case.expected_document_ids
    recall = len(matched) / len(case.expected_document_ids)
    precision = len(matched) / len(retrieved_ids) if retrieved_ids else 0.0

    failures: list[str] = []
    if recall < case.min_recall:
        failures.append(f"recall_below_{case.min_recall:g}")
    if precision < case.min_precision:
        failures.append(f"precision_below_{case.min_precision:g}")
    if case.max_queries is not None and len(outcome.queries) > case.max_queries:
        failures.append(f"queries_over_budget:{len(outcome.queries)}")
    if case.max_retrieved is not None and len(retrieved_ids) > case.max_retrieved:
        failures.append(f"retrieved_over_budget:{len(retrieved_ids)}")

    # Every failure except falling short on recall is a broken constraint rather than a partial result.
    breached_budget = [failure for failure in failures if not failure.startswith("recall_below")]
    return RetrievalCaseMetrics(
        question=case.question,
        passed=not failures,
        score=0.0 if breached_budget else recall,
        failures=tuple(failures),
        recall=recall,
        precision=precision,
        retrieved=len(retrieved_ids),
        missed_document_ids=tuple(sorted(case.expected_document_ids - matched)),
        queries=outcome.queries,
        latency_ms=latency_ms,
    )
