# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Approved assets and sanitized policy results."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from haystack.core.serialization import component_from_dict, component_to_dict, import_class_by_name

from haystack_integrations.agent_pack.optimization.assets.model_identity import _model_id_path, serialized_model_id


@dataclass(frozen=True, kw_only=True)
class ModelAsset:
    """
    An approved model deployment and the facts needed to rank it.

    :param model_id: The identifier this deployment is known by, as it appears on a configured chat generator.
    :param provider: Free-form provider label recorded for governance.
    :param deployment: Free-form deployment label, for example a region or an on-premise cluster name.
    :param input_cost_per_million: Price per million input tokens, used to rank candidates.
    :param output_cost_per_million: Price per million output tokens, used to rank candidates.
    :param generator: Optional serialized chat generator (`{"type": ..., "init_parameters": {...}}`) used to build
        this model's generator from scratch. Required to substitute a model served by a different provider than the
        reference: the fallback reuses the reference generator's own class, which can only ever change the model
        identifier. Declared as data rather than a callable so a catalog stays serializable and journalable.
    """

    model_id: str
    provider: str
    deployment: str
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    generator: dict[str, Any] | None = field(default=None, compare=False)

    def build_generator(self, reference_generator: Any) -> Any:
        """
        Create this model's generator from its declared configuration, or from the reference generator's.

        :param reference_generator: The reference harness's chat generator, reused as a template when this asset
            declares no `generator` of its own.
        :returns: A new chat generator configured for this model.
        :raises ValueError: If the declared generator has no `type`, if it is configured for a different model than
            this asset claims, or if no generator is declared and the reference generator hides its model identifier.
        """
        if self.generator is not None:
            return self._build_declared_generator()

        serialized = deepcopy(component_to_dict(obj=reference_generator, name="chat_generator"))
        init_parameters = serialized.get("init_parameters")
        if not isinstance(init_parameters, dict):
            msg = f"{type(reference_generator).__name__} has no serializable init_parameters."
            raise ValueError(msg)
        path = _model_id_path(init_parameters=init_parameters)
        if path is None:
            msg = (
                f"Model asset {self.model_id!r} needs an explicit 'generator' configuration because "
                f"{type(reference_generator).__name__} does not serialize a model identifier."
            )
            raise ValueError(msg)
        target: Any = init_parameters
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = self.model_id
        return component_from_dict(cls=type(reference_generator), data=serialized, name="chat_generator")

    def _build_declared_generator(self) -> Any:
        """Build the generator this asset declares, checking it really is configured for this asset's model."""
        specification = deepcopy(self.generator)
        if specification is None:  # pragma: no cover - guarded by the caller
            msg = f"Model asset {self.model_id!r} declares no generator."
            raise ValueError(msg)
        try:
            component_type = specification["type"]
        except KeyError as error:
            msg = f"Model asset {self.model_id!r} declares a generator without a 'type'."
            raise ValueError(msg) from error
        declared = serialized_model_id(serialized_component=specification)
        if declared is not None and declared != self.model_id:
            msg = (
                f"Model asset {self.model_id!r} declares a generator configured for {declared!r}. The catalog "
                "identifier must match the generator, otherwise candidate validation rejects the model."
            )
            raise ValueError(msg)
        return component_from_dict(
            cls=import_class_by_name(fully_qualified_name=component_type),
            data=specification,
            name="chat_generator",
        )


@dataclass(frozen=True, kw_only=True)
class ToolAsset:
    """
    An approved tool exposed to candidate agents.

    :param name: The tool name, as configured on the Agent.
    :param provider: Free-form provider label recorded for governance.
    """

    name: str
    provider: str = "local"


@dataclass(frozen=True, kw_only=True)
class AssetValidation:
    """
    Configuration-time validation result recorded in campaign journals.

    :param allowed: Whether the candidate may execute. False whenever any violation was recorded.
    :param model_ids: Every model identifier found on the candidate, including delegated agents.
    :param tool_names: Every tool name found on the candidate, including delegated agents.
    :param violations: Reasons that block execution.
    :param warnings: Reasons worth recording that do not block execution.
    """

    allowed: bool
    model_ids: tuple[str, ...]
    tool_names: tuple[str, ...]
    violations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """
        Return every recorded reason, blocking or not.

        :returns: The violations followed by the warnings.
        """
        return (*self.violations, *self.warnings)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the AssetValidation into a dictionary.

        :returns: A dictionary with keys 'allowed', 'model_ids', 'tool_names', 'violations', and 'warnings'.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssetValidation":
        """
        Create a new AssetValidation object from a dictionary.

        :param data: The dictionary to build the AssetValidation object from.
        :returns: The created object.
        """
        return cls(
            allowed=data["allowed"],
            model_ids=tuple(data["model_ids"]),
            tool_names=tuple(data["tool_names"]),
            violations=tuple(data.get("violations") or ()),
            warnings=tuple(data.get("warnings") or ()),
        )
