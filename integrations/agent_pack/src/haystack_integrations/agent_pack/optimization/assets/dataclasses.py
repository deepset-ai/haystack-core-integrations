# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Approved assets and sanitized policy results."""

from copy import deepcopy
from dataclasses import dataclass, field
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
class HarnessPatch:
    """
    An approved change to a harness's configuration, declared as a patch over its serialized form.

    A patch reaches any init parameter of any component in the harness, which is what makes one mechanism enough for
    reasoning effort, a retriever's `top_k`, a hook's settings, and anything else. The values are declared here
    rather than proposed, so an optimizer picks a name and cannot ask for a parameter the component would reject or
    a value outside what was approved.

    Paths are dotted and relative to the Agent's init parameters. A path segment addressing a list of tools selects
    the tool by name, so `tools.search_documents.component.init_parameters.top_k` reaches the retriever behind the
    `search_documents` tool. Missing intermediate dictionaries are created, so a nested generation parameter can be
    set on a generator that has none.

    :param name: The name a proposal refers to this patch by.
    :param patch: Dotted paths mapped to the values to set.
    :param description: What the patch is for, passed to the optimizer so it can choose between patches.
    """

    name: str
    patch: dict[str, Any] = field(compare=False)
    description: str | None = None


@dataclass(frozen=True, kw_only=True)
class ToolAsset:
    """
    An approved tool exposed to candidate agents.

    :param name: The tool name, as configured on the Agent.
    :param provider: Free-form provider label recorded for governance.
    """

    name: str
    provider: str = "local"
