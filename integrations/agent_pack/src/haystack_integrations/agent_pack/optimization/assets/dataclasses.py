# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The models and configuration changes an experiment may build a candidate from."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from haystack_integrations.agent_pack.optimization.assets.model_identity import _model_id_path, serialized_model_id


@dataclass(frozen=True, kw_only=True)
class ModelAsset:
    """
    A model an experiment may switch a harness to, and the prices needed to rank it.

    :param model_id: The identifier this model is known by, as it appears on a configured chat generator.
    :param input_cost_per_million: Price per million input tokens, used to rank candidates.
    :param output_cost_per_million: Price per million output tokens, used to rank candidates.
    :param generator: Optional serialized chat generator (`{"type": ..., "init_parameters": {...}}`) used to build
        this model's generator from scratch. Required to substitute a model served by a different provider than the
        reference: the fallback reuses the reference generator's own class, which can only ever change the model
        identifier. Declared as data rather than a callable so a catalog stays serializable and journalable.
    """

    model_id: str
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    generator: dict[str, Any] | None = field(default=None, compare=False)

    def substitution_patch(self, *, serialized_generator: dict[str, Any]) -> dict[str, Any]:
        """
        Return the patch that switches a harness to this model.

        A declared `generator` replaces the whole component, which is how a model served by a different provider is
        substituted. Otherwise only the model identifier changes, and the key it lives under is read from the
        reference generator rather than assumed, so an Azure deployment or a Hugging Face API generator works
        without special handling.

        :param serialized_generator: The reference harness's serialized chat generator.
        :returns: Dotted paths mapped to the values to set.
        :raises ValueError: If a declared generator has no `type` or is configured for a different model, or if no
            generator is declared and the reference generator does not serialize a model identifier.
        """
        if self.generator is not None:
            specification = deepcopy(self.generator)
            if "type" not in specification:
                msg = f"Model asset {self.model_id!r} declares a generator without a 'type'."
                raise ValueError(msg)
            declared = serialized_model_id(serialized_component=specification)
            if declared is not None and declared != self.model_id:
                msg = (
                    f"Model asset {self.model_id!r} declares a generator configured for {declared!r}. The catalog "
                    "identifier must match the generator, otherwise the candidate runs a different model than the "
                    "one being measured."
                )
                raise ValueError(msg)
            return {"chat_generator": specification}

        init_parameters = serialized_generator.get("init_parameters")
        path = _model_id_path(init_parameters=init_parameters) if isinstance(init_parameters, dict) else None
        if path is None:
            msg = (
                f"Model asset {self.model_id!r} needs an explicit 'generator' configuration because "
                f"{serialized_generator.get('type', 'the reference generator')} does not serialize a model "
                "identifier."
            )
            raise ValueError(msg)
        return {".".join(("chat_generator", "init_parameters", *path)): self.model_id}


@dataclass(frozen=True, kw_only=True)
class HarnessPatch:
    """
    An approved change to a harness's configuration, declared as a patch over its serialized form.

    A patch reaches any init parameter of any component in the harness, so one mechanism covers reasoning effort, a
    retriever's `top_k`, a hook's settings, and anything else. The values are declared here rather than proposed, so
    an optimizer picks a name and cannot ask for a parameter the component would reject.

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
