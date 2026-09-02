# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Persistence that lets an experiment resume where it stopped."""

import json
from pathlib import Path
from threading import RLock

from haystack_integrations.agent_pack.optimization.experiment.dataclasses import CandidateEvaluation


class ExperimentJournal:
    """
    Append-only JSON-lines journal used to resume candidate evaluation.

    A completed evaluation is never re-run. A failed one is retried, because a failure is usually a timeout, a rate
    limit, or a transient provider error rather than a permanent property of the candidate.
    """

    def __init__(self, path: str | Path) -> None:
        """
        Open or create a journal.

        :param path: Where the journal is written. Existing records are loaded on construction.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._records: dict[str, CandidateEvaluation] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = CandidateEvaluation.from_dict(data=json.loads(line))
                    self._records[record.candidate_id] = record

    def get(self, candidate_id: str, *, include_failures: bool = False) -> "CandidateEvaluation | None":
        """
        Return a previously completed evaluation, ignoring failed attempts unless asked for them.

        :param candidate_id: The candidate to look up.
        :param include_failures: Whether a journaled failure counts as a result.
        :returns: The journaled evaluation, or None if the candidate should be evaluated again.
        """
        with self._lock:
            record = self._records.get(candidate_id)
        if record is None or (record.failure is not None and not include_failures):
            return None
        return record

    def append(self, evaluation: CandidateEvaluation) -> None:
        """
        Persist an evaluation, leaving an already-completed evaluation untouched.

        :param evaluation: The outcome to record.
        """
        with self._lock:
            existing = self._records.get(evaluation.candidate_id)
            if existing is not None and existing.succeeded:
                return
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(evaluation.to_dict(), sort_keys=True) + "\n")
            self._records[evaluation.candidate_id] = evaluation
