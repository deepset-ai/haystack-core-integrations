# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Protocols implemented by campaign evaluators and proposers."""

from haystack_integrations.agent_pack.optimization.campaign.types.protocol import HarnessEvaluator, RecipeProposer

__all__ = ["HarnessEvaluator", "RecipeProposer"]
