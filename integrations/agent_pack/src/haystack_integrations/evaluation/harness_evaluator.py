# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from .dataclasses import EvalMetrics


class HarnessEvaluator(Protocol):
    """Measure a target configuration over labelled eval cases."""

    def evaluate(self, target: Any, eval_cases: list[Any]) -> EvalMetrics:
        """
        Measure a target configuration over the supplied eval cases.

        :param target: Target configuration to evaluate.
        :param eval_cases: The labelled expectations to score it against.
        :returns: Returns EvalMetrics, which includes the score and other relevant metrics.
        """
        ...

    async def evaluate_async(self, target: Any, eval_cases: list[Any]) -> EvalMetrics:
        """
        Measure a materialized configuration over the supplied eval cases.

        :param target: Materialized configuration to evaluate, of whatever kind this evaluator measures.
        :param eval_cases: The labelled expectations to score it against.
        :returns: Returns EvalMetrics, which includes the score and other relevant metrics.
        """
        ...

    def validate(self, target: Any) -> None:
        """
        Validate a target configuration to catch errors before harness runs another evaluation.

        :param target: Candidate configuration deserialized from YAML.
        :raises ValueError: If the candidate is missing something the evaluator requires.
        """
        ...
