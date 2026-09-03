# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for evaluating Agent Pack harnesses."""

from typing import Protocol

from haystack.components.agents import Agent

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics


class HarnessEvaluator(Protocol):
    """Measure materialized Agents over reference runs using normalized quality scores."""

    def evaluate(self, agent: Agent, reference_runs: list[AgentRunRecord]) -> EvaluationMetrics:
        """
        Measure an Agent over the supplied reference runs.

        :param agent: Materialized Agent configuration to evaluate.
        :param reference_runs: Successful runs supplying inputs and optional evaluator-specific reference outputs.
        :returns: Normalized quality in `[0.0, 1.0]`, latency, and raw model-usage measurements.
        """
        ...


__all__ = ["HarnessEvaluator"]
