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
    :param max_queries: Optional cap on how many queries the pipeline may issue for one question. Expanding a
        query buys recall with model calls, and without a cap the cheapest way to pass every case is to expand
        without limit.
    :param max_retrieved: Optional limit on how many returned documents are scored for one question. Recall
        alone has a degenerate optimum: a pipeline that returns most of the corpus reaches it, and measuring a
        retrieval pipeline that generates no answer cannot see the cost of doing so, because nothing downstream
        reads the documents. Scoring only the first this-many documents is what makes the size of the answer set
        matter, while leaving a run that returns one document too many worth almost exactly what it found. It
        applies to what the pipeline returns rather than to what it considers, so retrieving widely and then
        ranking the result down keeps every document scored while retrieving widely alone does not.
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
    Score for one retrieval eval case.

    `queries` is the evidence an optimizer acts on: it is what the configuration actually asked the store, and a
    recall failure is usually explained by the wording of those queries rather than by the number of them.

    `score` is what quality aggregates, and it is the case's recall rather than whether it passed. Recall over a
    handful of expected documents moves in steps of a half or a third, so a threshold on it reports a
    configuration that went from finding none of the evidence to finding two thirds of it as no change at all.
    Returning more documents than the case scores is reported as a failure but is not itself scored as one: only
    the first `max_retrieved` count towards recall, so overshooting costs whatever was pushed past the limit and
    nothing more. A ranker that returns eleven documents where ten are scored has made a small mistake and should
    measure as having made a small mistake. Issuing more queries than allowed still scores nothing, because a
    query already cost what it cost and there is no equivalent of ignoring it.
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
    # Deduplicated in the order the pipeline returned them, since which documents fall past the limit depends on
    # how the run ranked them.
    returned_ids = list(dict.fromkeys(document.id for document in outcome.documents))
    scored_ids = set(returned_ids[: case.max_retrieved] if case.max_retrieved is not None else returned_ids)
    matched = scored_ids & case.expected_document_ids
    recall = len(matched) / len(case.expected_document_ids)
    precision = len(matched) / len(scored_ids) if scored_ids else 0.0

    failures: list[str] = []
    if recall < case.min_recall:
        failures.append(f"recall_below_{case.min_recall:g}")
    if precision < case.min_precision:
        failures.append(f"precision_below_{case.min_precision:g}")
    if case.max_queries is not None and len(outcome.queries) > case.max_queries:
        failures.append(f"queries_over_budget:{len(outcome.queries)}")
    if case.max_retrieved is not None and len(returned_ids) > case.max_retrieved:
        failures.append(f"retrieved_over_budget:{len(returned_ids)}")

    # Returning too many documents is already paid for by the ones past the limit going unscored. Issuing too many
    # queries is not recoverable that way, so it stays a constraint rather than a matter of degree.
    return RetrievalCaseMetrics(
        question=case.question,
        passed=not failures,
        score=0.0 if any(failure.startswith("queries_over_budget") for failure in failures) else recall,
        failures=tuple(failures),
        recall=recall,
        precision=precision,
        retrieved=len(returned_ids),
        missed_document_ids=tuple(sorted(case.expected_document_ids - matched)),
        queries=outcome.queries,
        latency_ms=latency_ms,
    )
