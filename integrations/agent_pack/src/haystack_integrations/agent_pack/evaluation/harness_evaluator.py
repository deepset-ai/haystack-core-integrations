# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from .dataclasses import EVAL_CASES_KEY, EvaluationMetrics


class HarnessEvaluator(Protocol):
    """Measure a target configuration over labelled eval cases."""

    def evaluate(self, target: Any, eval_cases: list[Any]) -> EvaluationMetrics:
        """
        Measure a target configuration over the supplied eval cases.

        :param target: Target configuration to evaluate.
        :param eval_cases: The labelled expectations to score it against.
        :returns: Returns EvaluationMetrics, which includes the score and other relevant metrics.
        """
        ...

    async def evaluate_async(self, target: Any, eval_cases: list[Any]) -> EvaluationMetrics:
        """
        Measure a materialized configuration over the supplied eval cases.

        :param target: Materialized configuration to evaluate, of whatever kind this evaluator measures.
        :param eval_cases: The labelled expectations to score it against.
        :returns: Returns EvaluationMetrics, which includes the score and other relevant metrics.
        """
        ...

    def fingerprint(self, eval_cases: list[Any]) -> dict[str, Any]:
        """
        Describe what this evaluator measures, so two measurements are comparable only when it matches.

        :param eval_cases: The labelled expectations being scored.
        :returns: Every eval case, ordered by question.
        """
        return {
            EVAL_CASES_KEY: sorted(
                (eval_case.to_dict() for eval_case in eval_cases), key=lambda entry: str(entry["question"])
            )
        }

    def validate(self, target: Any) -> None:  # noqa: ARG002
        """
        Validate a target configuration to catch errors before harness runs another evaluation.

        :param target: Candidate configuration deserialized from YAML.
        :raises ValueError: If the candidate is missing something the evaluator requires.
        """
        return None
