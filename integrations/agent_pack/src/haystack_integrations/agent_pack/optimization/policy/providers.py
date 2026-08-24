# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Tool invocation policy providers."""

from typing import Any

from haystack.core.serialization import default_from_dict, default_to_dict

from haystack_integrations.agent_pack.optimization.policy.dataclasses import PolicyEvaluation


class StaticPolicyProvider:
    """
    Fail-closed allowlist policy suitable for local campaigns and tests.
    """

    def __init__(self, *, allowed_tools: list[str], policy_version: str = "local/v1") -> None:
        """
        Create a static policy provider.

        :param allowed_tools: Tools this policy allows. Everything else hits the default-deny rule.
        :param policy_version: Version recorded on every decision.
        """
        self.allowed_tools = frozenset(allowed_tools)
        self.policy_version = policy_version

    def evaluate(
        self,
        *,
        tool_name: str,
        tool_description: str,  # noqa: ARG002
        tool_params: dict[str, Any],  # noqa: ARG002
        context: dict[str, Any] | None,  # noqa: ARG002
    ) -> PolicyEvaluation:
        """
        Allow known tools and reject unknown tools.

        :param tool_name: The name of the tool the model wants to call.
        :param tool_description: The tool's description. Unused by this provider.
        :param tool_params: The arguments the model proposed. Unused by this provider.
        :param context: The request-scoped confirmation strategy context. Unused by this provider.
        :returns: An allow decision for an allowlisted tool, otherwise a default-deny decision.
        """
        if tool_name in self.allowed_tools:
            return PolicyEvaluation(
                decision="allow",
                policy_version=self.policy_version,
                rule_id=f"allow:{tool_name}",
                reason_code="tool_allowed",
            )
        return PolicyEvaluation(
            decision="deny",
            policy_version=self.policy_version,
            rule_id="default-deny",
            reason_code="tool_not_allowed",
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the static policy provider.

        :returns: A dictionary with the provider type and its init parameters.
        """
        return default_to_dict(self, allowed_tools=sorted(self.allowed_tools), policy_version=self.policy_version)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StaticPolicyProvider":
        """
        Deserialize a static policy provider.

        :param data: The dictionary to build the provider from.
        :returns: The created object.
        """
        return default_from_dict(cls, data)
