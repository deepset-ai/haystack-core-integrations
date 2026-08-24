# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The explicit allowlist of trusted structural transformations."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from haystack.components.agents import Agent

from haystack_integrations.agent_pack.optimization.policy.catalog import ApprovedAssetCatalog

StructuralRecipeFactory = Callable[[Agent, ApprovedAssetCatalog, dict[str, Any]], Agent]

# Accepted parameter types for registered structural recipes. A trailing "?" marks the parameter optional.
SCHEMA_TYPES: dict[str, type | tuple[type, ...]] = {
    "str": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
    "list": list,
    "dict": dict,
}


def validate_parameters(name: str, schema: Mapping[str, str], parameters: Mapping[str, Any]) -> None:
    """
    Check proposed parameters against a registered recipe's declared schema, rejecting anything unexpected.

    :param name: The registered transformation's name, used in error messages.
    :param schema: Maps each parameter name to its declared type.
    :param parameters: The proposed parameters.
    :raises ValueError: If a parameter is missing, of the wrong type, or not declared by the schema.
    """
    unexpected = sorted(set(parameters) - set(schema))
    if unexpected:
        msg = f"Structural recipe {name!r} received unsupported parameters: {', '.join(unexpected)}."
        raise ValueError(msg)
    for key, declared in schema.items():
        optional = declared.endswith("?")
        type_name = declared[:-1] if optional else declared
        try:
            expected = SCHEMA_TYPES[type_name]
        except KeyError as error:
            msg = f"Structural recipe {name!r} declares an unsupported parameter type {declared!r} for {key!r}."
            raise ValueError(msg) from error
        if key not in parameters:
            if optional:
                continue
            msg = f"Structural recipe {name!r} requires parameter {key!r}."
            raise ValueError(msg)
        value = parameters[key]
        # `bool` is a subclass of `int`, so an isinstance check alone would let True through as an integer.
        if (isinstance(value, bool) and expected is not bool) or not isinstance(value, expected):
            msg = f"Structural recipe {name!r} expects {key!r} to be {type_name}."
            raise ValueError(msg)


class StructuralRecipeRegistry:
    """
    Explicit allowlist of named structural transformations.

    Each registration declares a parameter schema, so proposed parameters are checked before they reach a factory.
    A schema maps a parameter name to one of `str`, `int`, `float`, `bool`, `list`, or `dict`, with a trailing `?`
    marking it optional. Unknown parameters are rejected.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._factories: dict[str, tuple[StructuralRecipeFactory, dict[str, str]]] = {}

    def register(
        self, name: str, factory: StructuralRecipeFactory, *, parameters_schema: Mapping[str, str] | None = None
    ) -> None:
        """
        Register one trusted transformation by stable name.

        :param name: The name a recipe refers to this transformation by.
        :param factory: Builds the candidate from the reference Agent, the asset catalog, and the parameters.
        :param parameters_schema: The parameters this transformation accepts, and their types.
        :raises ValueError: If the name is already registered.
        """
        if name in self._factories:
            msg = f"Structural recipe {name!r} is already registered."
            raise ValueError(msg)
        self._factories[name] = (factory, dict(parameters_schema or {}))

    def schema(self, name: str) -> dict[str, str]:
        """
        Return the declared parameter schema for a registered transformation.

        :param name: The registered name.
        :returns: The parameter schema.
        """
        return dict(self._factories[name][1])

    def describe(self) -> dict[str, dict[str, str]]:
        """
        Return every registered transformation and its parameter schema, for inclusion in a proposal request.

        :returns: The registered names mapped to their parameter schemas.
        """
        return {name: dict(schema) for name, (_, schema) in self._factories.items()}

    def materialize(
        self, name: str, reference: Agent, assets: ApprovedAssetCatalog, parameters: dict[str, Any]
    ) -> Agent:
        """
        Validate parameters and execute one registered transformation, or reject the recipe.

        :param name: The registered name to execute.
        :param reference: The champion harness to transform.
        :param assets: The approved model and tool allowlist.
        :param parameters: The proposed parameters.
        :returns: The new candidate Agent.
        :raises ValueError: If the name is not registered, or the parameters do not match its schema.
        """
        try:
            factory, schema = self._factories[name]
        except KeyError as error:
            msg = f"Structural recipe {name!r} is not registered."
            raise ValueError(msg) from error
        validate_parameters(name=name, schema=schema, parameters=parameters)
        return factory(reference, assets, parameters)


@dataclass(frozen=True, kw_only=True)
class RegisteredStructuralRecipe:
    """
    Reference a trusted structural transformation and JSON-compatible parameters.

    :param name: The registered transformation to execute.
    :param parameters: Parameters passed to it, checked against its declared schema before it runs.
    :param registry: The registry holding the transformation. Excluded from equality and serialization because it is
        a runtime object rather than part of the recipe's identity.
    """

    name: str
    parameters: dict[str, Any]
    registry: StructuralRecipeRegistry = field(compare=False, repr=False)
    kind: ClassVar[str] = "registered_structure"

    def materialize(self, reference: Agent, assets: ApprovedAssetCatalog) -> Agent:
        """
        Materialize through the explicit structural recipe registry.

        :param reference: The champion harness to transform.
        :param assets: The approved model and tool allowlist.
        :returns: The new candidate Agent.
        """
        return self.registry.materialize(name=self.name, reference=reference, assets=assets, parameters=self.parameters)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the RegisteredStructuralRecipe into a dictionary, excluding the runtime registry.

        :returns: A dictionary with keys 'kind', 'name', and 'parameters'.
        """
        return {"kind": self.kind, "name": self.name, "parameters": self.parameters}
