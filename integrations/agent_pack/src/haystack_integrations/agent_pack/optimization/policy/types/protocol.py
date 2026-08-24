# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Protocol for tool invocation policy providers."""

from typing import Any, Protocol

from haystack_integrations.agent_pack.optimization.policy.dataclasses import PolicyEvaluation


class PolicyProvider(Protocol):
    """Provider of synchronous, invocation-time tool policy decisions."""

    def evaluate(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> PolicyEvaluation:
        """
        Evaluate one proposed tool invocation.

        :param tool_name: The name of the tool the model wants to call.
        :param tool_description: The tool's description.
        :param tool_params: The arguments the model proposed.
        :param context: The request-scoped confirmation strategy context, if any.
        :returns: A sanitized decision.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the provider.

        :returns: The serialized provider.
        """
        ...
