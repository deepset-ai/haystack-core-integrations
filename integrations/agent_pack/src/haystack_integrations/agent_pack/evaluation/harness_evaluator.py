# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from haystack import Pipeline
from haystack.components.agents import Agent

from haystack_integrations.agent_pack.dataclasses import EvaluationMetrics, RunRecord


class HarnessEvaluator(Protocol):
    """Measure materialized configurations over reference runs using normalized quality scores."""

    def evaluate(self, target: Agent | Pipeline, reference_runs: list[RunRecord]) -> EvaluationMetrics:
        """
        Measure an Agent or Pipeline over the supplied reference runs.

        :param target: Materialized configuration to evaluate.
        :param reference_runs: Successful runs supplying inputs and optional evaluator-specific reference outputs.
        :returns: Normalized quality in `[0.0, 1.0]`, latency, and raw model-usage measurements.
        """
        ...
