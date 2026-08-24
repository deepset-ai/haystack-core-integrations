# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Programmatic tool-call policy enforcement through the Human-in-the-Loop confirmation hook."""

from typing import Any

from haystack import logging
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.hooks.human_in_the_loop import ToolExecutionDecision
from haystack.utils.deserialization import deserialize_component_inplace

from haystack_integrations.agent_pack.optimization.policy.dataclasses import PolicyEvaluation
from haystack_integrations.agent_pack.optimization.policy.types.protocol import PolicyProvider

logger = logging.getLogger(__name__)

POLICY_DECISIONS_CONTEXT_KEY = "agent_pack.policy_decisions"


class PolicyEnforcementStrategy:
    """
    A programmatic allow/deny `ConfirmationStrategy` for `ConfirmationHook`.

    Register it at the `before_tool` hook point to enforce invocation-time tool policy without a human in the loop.
    The strategy only ever allows the original call or rejects it; it never rewrites tool arguments. Provider
    errors, unknown tools, missing policy, and indeterminate decisions all fail closed.

    Decisions are appended to `confirmation_strategy_context` under `POLICY_DECISIONS_CONTEXT_KEY` so an evaluator
    can journal the policy version, matched rule, and reason code alongside the run they belong to.
    """

    def __init__(self, provider: PolicyProvider) -> None:
        """
        Create a policy enforcement strategy.

        :param provider: The policy provider consulted for every tool call.
        """
        self.provider = provider

    def _evaluate(
        self, *, tool_name: str, tool_description: str, tool_params: dict[str, Any], context: dict[str, Any] | None
    ) -> PolicyEvaluation:
        """Consult the provider, turning a provider error or an undecided answer into a rejection."""
        try:
            evaluation = self.provider.evaluate(
                tool_name=tool_name, tool_description=tool_description, tool_params=tool_params, context=context
            )
        except Exception:
            # A provider that raises is indistinguishable from a provider that denies, so the call is rejected. The
            # exception is logged because a misconfigured provider otherwise looks exactly like a policy decision.
            logger.exception(
                "Policy provider failed while evaluating tool {tool_name}; rejecting the call.", tool_name=tool_name
            )
            return PolicyEvaluation(
                decision="deny",
                policy_version="unavailable",
                rule_id="provider-error",
                reason_code="policy_provider_error",
            )
        if evaluation.decision in ("allow", "deny"):
            return evaluation
        logger.warning(
            "Policy provider returned an indeterminate decision for tool {tool_name}; rejecting the call.",
            tool_name=tool_name,
        )
        return PolicyEvaluation(
            decision="deny",
            policy_version=evaluation.policy_version,
            rule_id=evaluation.rule_id,
            reason_code="indeterminate_decision",
        )

    def run(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        """
        Return an unchanged allow decision or a safe, fail-closed rejection.

        :param tool_name: The name of the tool the model wants to call.
        :param tool_description: The tool's description, passed through to the policy provider.
        :param tool_params: The arguments the model proposed. Returned unchanged when the call is allowed, and never
            included in the rejection feedback.
        :param tool_call_id: Identifier correlating the decision with the tool invocation.
        :param confirmation_strategy_context: Request-scoped context. Decisions are appended to it under
            `POLICY_DECISIONS_CONTEXT_KEY`.
        :returns: A decision that either executes the original call or rejects it with a stable reason code.
        """
        evaluation = self._evaluate(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_params=tool_params,
            context=confirmation_strategy_context,
        )

        if isinstance(confirmation_strategy_context, dict):
            records = confirmation_strategy_context.setdefault(POLICY_DECISIONS_CONTEXT_KEY, [])
            if isinstance(records, list):
                records.append({"tool_name": tool_name, **evaluation.to_dict()})

        allowed = evaluation.decision == "allow"
        if not allowed:
            logger.info(
                "Tool call rejected by policy: tool={tool_name} rule={rule_id} reason={reason_code}",
                tool_name=tool_name,
                rule_id=evaluation.rule_id,
                reason_code=evaluation.reason_code,
            )
        return ToolExecutionDecision(
            tool_name=tool_name,
            execute=allowed,
            tool_call_id=tool_call_id,
            final_tool_params=tool_params if allowed else None,
            feedback=None if allowed else f"Tool call rejected by policy ({evaluation.reason_code}).",
        )

    async def run_async(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        """
        Use the same deterministic policy logic during asynchronous Agent runs.

        :param tool_name: The name of the tool the model wants to call.
        :param tool_description: The tool's description, passed through to the policy provider.
        :param tool_params: The arguments the model proposed.
        :param tool_call_id: Identifier correlating the decision with the tool invocation.
        :param confirmation_strategy_context: Request-scoped context that collects the recorded decisions.
        :returns: A decision that either executes the original call or rejects it with a stable reason code.
        """
        return self.run(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_params=tool_params,
            tool_call_id=tool_call_id,
            confirmation_strategy_context=confirmation_strategy_context,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the strategy and its provider.

        :returns: A dictionary with the strategy type and its init parameters.
        """
        return default_to_dict(self, provider=self.provider.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyEnforcementStrategy":
        """
        Deserialize the strategy and its provider.

        :param data: The dictionary to build the strategy from.
        :returns: The created object.
        """
        deserialize_component_inplace(data["init_parameters"], key="provider")
        return default_from_dict(cls, data)
