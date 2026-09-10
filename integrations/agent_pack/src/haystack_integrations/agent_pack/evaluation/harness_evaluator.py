# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any, Protocol

from haystack_integrations.agent_pack.dataclasses import EvaluationMetrics
from haystack_integrations.agent_pack.run_digest import EVAL_CASES_KEY

from .dataclasses import RetrievalEvalCase


class HarnessEvaluator(Protocol):
    """
    Measure materialized configurations over labelled eval cases using normalized quality scores.

    An evaluator measures one kind of target — an Agent for one harness, a Pipeline for another — and says so in
    its own signatures. `target` is typed loosely here so an implementation can name the kind it measures.

    Eval cases are independent and each spends its time waiting on a model, so an evaluator measures several at once
    and `evaluate_async` is the one that does the work. `evaluate` runs it to completion for a caller that has no event
    loop of its own.

    Subclass it to inherit `fingerprint` and a `validate` that checks nothing; implement it structurally to supply both.

    :param eval_cases: The labelled expectations being scored, keyed by question.
    """

    eval_cases: Mapping[str, RetrievalEvalCase]

    def evaluate(self, target: Any) -> EvaluationMetrics:
        """
        Measure a materialized configuration over every eval case, from synchronous code.

        :param target: Materialized configuration to evaluate, of whatever kind this evaluator measures.
        :returns: Normalized quality in `[0.0, 1.0]`, latency, and raw model-usage measurements.
        """
        ...

    async def evaluate_async(self, target: Any) -> EvaluationMetrics:
        """
        Measure a materialized configuration over every eval case.

        :param target: Materialized configuration to evaluate, of whatever kind this evaluator measures.
        :returns: Normalized quality in `[0.0, 1.0]`, latency, and raw model-usage measurements.
        """
        ...

    def fingerprint(self) -> dict[str, Any]:
        """
        Describe what this evaluator measures, so two measurements are comparable only when it matches.

        :returns: Every configured eval case, ordered by question.
        """
        return {
            EVAL_CASES_KEY: sorted(
                (eval_case.to_dict() for eval_case in self.eval_cases.values()),
                key=lambda entry: str(entry["question"]),
            )
        }

    def validate(self, target: Any) -> None:  # noqa: ARG002
        """
        Validate a target configuration to catch errors before harness runs another evaluation.

        :param target: Candidate configuration deserialized from YAML.
        :raises ValueError: If the candidate is missing something the evaluator requires.
        """
        return None
