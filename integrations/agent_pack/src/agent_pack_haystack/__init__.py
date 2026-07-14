# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Short-import alias for the agent pack. Allows `from agent_pack_haystack import create_deep_research_agent`."""

from haystack_integrations.components.agents.agent_pack import create_deep_research_agent

__all__ = ["create_deep_research_agent"]
