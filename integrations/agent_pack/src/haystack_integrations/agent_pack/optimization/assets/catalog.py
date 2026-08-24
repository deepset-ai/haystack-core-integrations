# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The approved asset allowlist and recursive candidate validation."""

from typing import Any

from haystack import logging
from haystack.components.agents import Agent
from haystack.tools import AgentTool, flatten_tools_or_toolsets

from haystack_integrations.agent_pack.optimization.assets.dataclasses import AssetValidation, ModelAsset, ToolAsset
from haystack_integrations.agent_pack.optimization.assets.model_identity import (
    generator_model_id,
    serialized_model_id,
)

logger = logging.getLogger(__name__)


class ApprovedAssetCatalog:
    """
    Allowlist used to materialize and recursively validate candidate agents.
    """

    def __init__(self, *, models: list[ModelAsset], tools: list[ToolAsset], strict_identification: bool = True) -> None:
        """
        Create an approved asset catalog.

        :param models: Approved model deployments.
        :param tools: Approved tools.
        :param strict_identification: Whether an asset that cannot be identified blocks the candidate. True by
            default, so an unreadable configuration fails closed. Set it to False to record such cases as warnings
            instead, which is useful when a harness legitimately contains components this catalog cannot introspect.
        :raises ValueError: If a model ID or tool name appears twice.
        """
        self.models = {asset.model_id: asset for asset in models}
        self.tools = {asset.name: asset for asset in tools}
        self.strict_identification = strict_identification
        if len(self.models) != len(models):
            msg = "Model asset IDs must be unique."
            raise ValueError(msg)
        if len(self.tools) != len(tools):
            msg = "Tool asset names must be unique."
            raise ValueError(msg)

    def model(self, model_id: str) -> ModelAsset:
        """
        Return an approved model or fail closed.

        :param model_id: The model identifier to look up.
        :returns: The approved model asset.
        :raises ValueError: If the model is not in the catalog.
        """
        try:
            return self.models[model_id]
        except KeyError as error:
            msg = f"Model {model_id!r} is not in the approved asset catalog."
            raise ValueError(msg) from error

    def tool(self, tool_name: str) -> ToolAsset:
        """
        Return an approved tool or fail closed.

        :param tool_name: The tool name to look up.
        :returns: The approved tool asset.
        :raises ValueError: If the tool is not in the catalog.
        """
        try:
            return self.tools[tool_name]
        except KeyError as error:
            msg = f"Tool {tool_name!r} is not in the approved asset catalog."
            raise ValueError(msg) from error

    def validate_agent(self, agent: Agent) -> AssetValidation:
        """
        Recursively validate models and tools, including delegated `AgentTool` agents.

        :param agent: The candidate Agent to inspect.
        :returns: Every asset found on the candidate, and the reasons it may or may not execute.
        """
        collector = _AssetCollector(catalog=self)
        collector.visit(candidate=agent)
        return collector.result(strict_identification=self.strict_identification)

    def require_valid_agent(self, agent: Agent) -> AssetValidation:
        """
        Return validation details or raise before the candidate can execute.

        :param agent: The candidate Agent to inspect.
        :returns: Every asset found on the candidate. Warnings are logged.
        :raises ValueError: If the candidate uses an asset outside the catalog.
        """
        validation = self.validate_agent(agent=agent)
        if not validation.allowed:
            msg = f"Candidate uses unapproved assets: {', '.join(validation.violations)}."
            raise ValueError(msg)
        if validation.warnings:
            logger.warning(
                "Candidate passed asset validation with warnings: {warnings}",
                warnings=", ".join(validation.warnings),
            )
        return validation


class _AssetCollector:
    """Walks a candidate Agent, recording every model and tool it is configured with."""

    def __init__(self, catalog: ApprovedAssetCatalog) -> None:
        """
        Create a collector.

        :param catalog: The allowlist every recorded asset is checked against.
        """
        self.catalog = catalog
        self.model_ids: set[str] = set()
        self.tool_names: set[str] = set()
        self.violations: set[str] = set()
        self.unidentified: set[str] = set()
        self._visited: set[int] = set()

    def result(self, *, strict_identification: bool) -> AssetValidation:
        """
        Return what the walk found.

        :param strict_identification: Whether unidentifiable assets block execution.
        :returns: The validation outcome.
        """
        violations = set(self.violations)
        warnings: set[str] = set()
        if strict_identification:
            violations |= self.unidentified
        else:
            warnings = set(self.unidentified)
        return AssetValidation(
            allowed=not violations,
            model_ids=tuple(sorted(self.model_ids)),
            tool_names=tuple(sorted(self.tool_names)),
            violations=tuple(sorted(violations)),
            warnings=tuple(sorted(warnings)),
        )

    def _record_model(self, model_id: Any) -> None:
        """Record a model, flagging it as unapproved or as unidentifiable."""
        if not isinstance(model_id, str):
            self.unidentified.add("model_not_identifiable")
            return
        self.model_ids.add(model_id)
        if model_id not in self.catalog.models:
            self.violations.add(f"model_not_approved:{model_id}")

    def _record_tool(self, tool_name: Any) -> None:
        """Record a tool, flagging it as unapproved or as unidentifiable."""
        if not isinstance(tool_name, str):
            self.unidentified.add("tool_not_identifiable")
            return
        self.tool_names.add(tool_name)
        if tool_name not in self.catalog.tools:
            self.violations.add(f"tool_not_approved:{tool_name}")

    def _visit_nested_models(self, value: Any) -> None:
        """Record any model configured on a component nested inside a serialized structure, such as a hook."""
        if isinstance(value, list):
            for item in value:
                self._visit_nested_models(value=item)
            return
        if not isinstance(value, dict):
            return
        if isinstance(value.get("type"), str):
            nested_model = serialized_model_id(serialized_component=value)
            if nested_model is not None:
                self._record_model(model_id=nested_model)
        for nested in value.values():
            self._visit_nested_models(value=nested)

    def _visit_serialized_agent(self, data: dict[str, Any]) -> None:
        """Record the assets of a delegated Agent, which is only reachable in its serialized form."""
        init_parameters = data.get("init_parameters") or {}
        generator = init_parameters.get("chat_generator") or {}
        self._record_model(
            model_id=serialized_model_id(serialized_component=generator) if isinstance(generator, dict) else None
        )
        self._visit_nested_models(value=init_parameters.get("hooks") or {})
        for serialized_tool in init_parameters.get("tools") or []:
            if not isinstance(serialized_tool, dict):
                self.unidentified.add("tool_not_identifiable")
                continue
            tool_data = serialized_tool.get("data") or serialized_tool.get("init_parameters") or {}
            self._record_tool(tool_name=tool_data.get("name"))
            if serialized_tool.get("type", "").endswith(".AgentTool"):
                nested_agent = tool_data.get("agent")
                if isinstance(nested_agent, dict):
                    self._visit_serialized_agent(data=nested_agent)
                else:
                    self.unidentified.add("delegated_agent_not_identifiable")

    def visit(self, candidate: Agent) -> None:
        """
        Record every asset configured on a candidate Agent and its delegated agents.

        :param candidate: The Agent to inspect.
        """
        if id(candidate) in self._visited:
            return
        self._visited.add(id(candidate))
        self._record_model(model_id=generator_model_id(generator=candidate.chat_generator))

        if candidate.hooks:
            # Hooks can hold their own generators, and the only way to read them is the serialized form.
            try:
                serialized_hooks = candidate.to_dict().get("init_parameters", {}).get("hooks") or {}
            except Exception:
                self.unidentified.add("agent_not_serializable")
            else:
                self._visit_nested_models(value=serialized_hooks)

        for configured_tool in flatten_tools_or_toolsets(tools=candidate.tools):
            self._record_tool(tool_name=configured_tool.name)
            if isinstance(configured_tool, AgentTool):
                try:
                    serialized = configured_tool.to_dict()
                except Exception:
                    self.unidentified.add("delegated_agent_not_serializable")
                    continue
                nested_agent = serialized.get("data", {}).get("agent")
                if isinstance(nested_agent, dict):
                    self._visit_serialized_agent(data=nested_agent)
                else:
                    self.unidentified.add("delegated_agent_not_identifiable")
