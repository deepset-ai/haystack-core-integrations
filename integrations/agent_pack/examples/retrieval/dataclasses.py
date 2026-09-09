# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from typing import Any


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
    :param stage_texts: A capped sample of whatever each component emitted as text, by component name and
        output socket. For a pipeline that rewrites the question, this is what it actually asked the store,
        which is usually what explains a recall failure rather than how many queries there were.
    :param failures: Every expectation the run missed, named.
    :param recall_at_k: The share of the expected documents found within the first `k` returned.
    :param precision_at_k: The share of the first `k` returned that were expected.
    :param retrieved: Everything the run returned, which is separate from how deep it was scored: returning more
        than `k` is not a fault, it simply earns nothing for the documents past the cutoff.
    :param missed_document_ids: The expected documents the run did not return, sorted.
    :param latency_ms: Measured wall-clock duration of the run.
    """

    question: str
    passed: bool
    score: float
    stage_outputs: dict[str, dict[str, int]]
    stage_texts: dict[str, dict[str, list[str]]]
    failures: tuple[str, ...]
    recall_at_k: float
    precision_at_k: float
    retrieved: int
    missed_document_ids: tuple[str, ...]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        data = asdict(self)
        for key in ("failures", "missed_document_ids"):
            data[key] = list(getattr(self, key))
        return data
