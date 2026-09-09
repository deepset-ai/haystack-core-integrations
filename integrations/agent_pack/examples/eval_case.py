# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(kw_only=True)
class EvalCase:
    """
    What every harness needs to score one labelled question: the question, and the documents that answer it.

    Harnesses subclass this and add what only they can measure.

    :param question: The question to put to whatever is under evaluation.
    :param evidence: Ground truth, as `{document id: the quote found in that document}`. The keys are the
        documents recall is measured against, and the values say what each one was needed for:

            {"a1b2c3...": "Tyreek Hill now needs to ...", "d4e5f6...": "The Dolphins went on to ..."}

    :param min_recall: Minimum share of the needed documents that must be found.
    :param min_precision: Minimum share of what came back that must be needed. Left at 0 by default, because
        returning more than was asked for is not itself a fault; raise it to make over-retrieval cost something.
    """

    question: str
    evidence: dict[str, str]
    min_recall: float = 1.0
    min_precision: float = 0.0

    def __post_init__(self) -> None:
        """Require ground truth to score against."""
        if not self.evidence:
            msg = f"Case {self.question!r} needs evidence to score against."
            raise ValueError(msg)

    @property
    def expected_document_ids(self) -> frozenset[str]:
        """The documents an answer needs, which are the ones its evidence was found in."""
        return frozenset(self.evidence)
