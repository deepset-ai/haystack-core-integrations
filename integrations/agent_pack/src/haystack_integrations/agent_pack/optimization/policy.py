# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Approved-asset validation and programmatic tool-call policy enforcement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from haystack.components.agents import Agent
from haystack.core.serialization import component_from_dict, component_to_dict, default_from_dict, default_to_dict
from haystack.hooks.human_in_the_loop import ToolExecutionDecision
from haystack.tools import AgentTool, flatten_tools_or_toolsets
from haystack.utils.deserialization import deserialize_component_inplace

POLICY_DECISIONS_CONTEXT_KEY = "agent_pack.policy_decisions"


@dataclass(frozen=True)
class ModelAsset:
    """An approved model deployment and the facts needed to rank it."""

    model_id: str
    provider: str
    deployment: str
    sovereign: bool = False
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    generator_factory: Callable[[Any], Any] | None = field(default=None, repr=False, compare=False)

    def build_generator(self, reference_generator: Any) -> Any:
        """Create this model's generator using a factory or the reference generator's serialized configuration."""
        if self.generator_factory is not None:
            return self.generator_factory(reference_generator)
        serialized = component_to_dict(reference_generator, "chat_generator")
        init_parameters = serialized.get("init_parameters")
        if not isinstance(init_parameters, dict) or "model" not in init_parameters:
            msg = (
                f"Model asset {self.model_id!r} needs a generator_factory because "
                f"{type(reference_generator).__name__} does not serialize a 'model' parameter."
            )
            raise ValueError(msg)
        init_parameters["model"] = self.model_id
        return component_from_dict(type(reference_generator), serialized, "chat_generator")


@dataclass(frozen=True)
class ToolAsset:
    """An approved tool exposed to candidate agents."""

    name: str
    provider: str = "local"
    sovereign: bool = True


@dataclass(frozen=True)
class AssetValidation:
    """Configuration-time validation result recorded in campaign journals."""

    allowed: bool
    model_ids: tuple[str, ...]
    tool_names: tuple[str, ...]
    reason_codes: tuple[str, ...] = ()


class ApprovedAssetCatalog:
    """Allowlist used to materialize and recursively validate candidate agents."""

    def __init__(self, *, models: list[ModelAsset], tools: list[ToolAsset]) -> None:
        self.models = {asset.model_id: asset for asset in models}
        self.tools = {asset.name: asset for asset in tools}
        if len(self.models) != len(models):
            msg = "Model asset IDs must be unique."
            raise ValueError(msg)
        if len(self.tools) != len(tools):
            msg = "Tool asset names must be unique."
            raise ValueError(msg)

    def model(self, model_id: str) -> ModelAsset:
        """Return an approved model or fail closed."""
        try:
            return self.models[model_id]
        except KeyError as error:
            msg = f"Model {model_id!r} is not in the approved asset catalog."
            raise ValueError(msg) from error

    def validate_agent(self, agent: Agent) -> AssetValidation:
        """Recursively validate models and tools, including delegated ``AgentTool`` agents."""
        model_ids: set[str] = set()
        tool_names: set[str] = set()
        reasons: set[str] = set()
        visited: set[int] = set()

        def record_model(model_id: Any) -> None:
            if not isinstance(model_id, str):
                reasons.add("model_not_identifiable")
                return
            model_ids.add(model_id)
            if model_id not in self.models:
                reasons.add(f"model_not_approved:{model_id}")

        def record_tool(tool_name: Any) -> None:
            if not isinstance(tool_name, str):
                reasons.add("tool_not_identifiable")
                return
            tool_names.add(tool_name)
            if tool_name not in self.tools:
                reasons.add(f"tool_not_approved:{tool_name}")

        def visit_serialized_models(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    visit_serialized_models(item)
                return
            if not isinstance(value, dict):
                return

            component_type = value.get("type")
            init_parameters = value.get("init_parameters") or value.get("data") or {}
            if isinstance(component_type, str) and "Generator" in component_type and isinstance(init_parameters, dict):
                record_model(init_parameters.get("model"))
            for nested in value.values():
                visit_serialized_models(nested)

        def visit_serialized_agent(data: dict[str, Any]) -> None:
            init_parameters = data.get("init_parameters") or {}
            generator = init_parameters.get("chat_generator") or {}
            generator_parameters = generator.get("init_parameters") or generator.get("data") or {}
            record_model(generator_parameters.get("model"))
            visit_serialized_models(init_parameters.get("hooks") or {})
            for serialized_tool in init_parameters.get("tools") or []:
                if not isinstance(serialized_tool, dict):
                    reasons.add("tool_not_identifiable")
                    continue
                tool_data = serialized_tool.get("data") or serialized_tool.get("init_parameters") or {}
                record_tool(tool_data.get("name"))
                if serialized_tool.get("type", "").endswith(".AgentTool"):
                    nested_agent = tool_data.get("agent")
                    if isinstance(nested_agent, dict):
                        visit_serialized_agent(nested_agent)
                    else:
                        reasons.add("delegated_agent_not_identifiable")

        def visit(candidate: Agent) -> None:
            if id(candidate) in visited:
                return
            visited.add(id(candidate))
            record_model(getattr(candidate.chat_generator, "model", None))

            try:
                serialized_hooks = candidate.to_dict().get("init_parameters", {}).get("hooks") or {}
            except Exception:
                reasons.add("hooks_not_serializable")
            else:
                visit_serialized_models(serialized_hooks)

            for tool in flatten_tools_or_toolsets(candidate.tools):
                record_tool(tool.name)
                if isinstance(tool, AgentTool):
                    try:
                        serialized = tool.to_dict()
                    except Exception:
                        reasons.add("delegated_agent_not_serializable")
                        continue
                    nested_agent = serialized.get("data", {}).get("agent")
                    if isinstance(nested_agent, dict):
                        visit_serialized_agent(nested_agent)
                    else:
                        reasons.add("delegated_agent_not_identifiable")

        visit(agent)
        return AssetValidation(
            allowed=not reasons,
            model_ids=tuple(sorted(model_ids)),
            tool_names=tuple(sorted(tool_names)),
            reason_codes=tuple(sorted(reasons)),
        )

    def require_valid_agent(self, agent: Agent) -> AssetValidation:
        """Return validation details or raise before the candidate can execute."""
        validation = self.validate_agent(agent)
        if not validation.allowed:
            msg = f"Candidate uses unapproved assets: {', '.join(validation.reason_codes)}."
            raise ValueError(msg)
        return validation


@dataclass(frozen=True)
class PolicyEvaluation:
    """Sanitized result returned by a tool invocation policy provider."""

    decision: Literal["allow", "deny", "indeterminate"]
    policy_version: str
    rule_id: str
    reason_code: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the non-sensitive policy result for campaign journaling."""
        return asdict(self)


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
        """Evaluate one proposed tool invocation."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the provider."""
        ...


class StaticPolicyProvider:
    """Fail-closed allowlist policy suitable for local campaigns and tests."""

    def __init__(self, *, allowed_tools: list[str], policy_version: str = "local/v1") -> None:
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
        """Allow known tools and reject unknown tools."""
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
        """Serialize the static policy provider."""
        return default_to_dict(self, allowed_tools=sorted(self.allowed_tools), policy_version=self.policy_version)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StaticPolicyProvider:
        """Deserialize a static policy provider."""
        return default_from_dict(cls, data)


class PolicyEnforcementStrategy:
    """A programmatic allow/deny ``ConfirmationStrategy`` for ``ConfirmationHook``."""

    def __init__(self, provider: PolicyProvider) -> None:
        self.provider = provider

    def run(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        """Return an unchanged allow decision or a safe, fail-closed rejection."""
        try:
            evaluation = self.provider.evaluate(
                tool_name=tool_name,
                tool_description=tool_description,
                tool_params=tool_params,
                context=confirmation_strategy_context,
            )
            if evaluation.decision not in ("allow", "deny"):
                evaluation = PolicyEvaluation(
                    decision="deny",
                    policy_version=evaluation.policy_version,
                    rule_id=evaluation.rule_id,
                    reason_code="indeterminate_decision",
                )
        except Exception:
            evaluation = PolicyEvaluation(
                decision="deny",
                policy_version="unavailable",
                rule_id="provider-error",
                reason_code="policy_provider_error",
            )

        if confirmation_strategy_context is not None:
            records = confirmation_strategy_context.get(POLICY_DECISIONS_CONTEXT_KEY)
            if isinstance(records, list):
                records.append(evaluation.to_dict())

        allowed = evaluation.decision == "allow"
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
        """Use the same deterministic policy logic during asynchronous Agent runs."""
        return self.run(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_params=tool_params,
            tool_call_id=tool_call_id,
            confirmation_strategy_context=confirmation_strategy_context,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the strategy and its provider."""
        return default_to_dict(self, provider=self.provider.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyEnforcementStrategy:
        """Deserialize the strategy and its provider."""
        deserialize_component_inplace(data["init_parameters"], key="provider")
        return default_from_dict(cls, data)
