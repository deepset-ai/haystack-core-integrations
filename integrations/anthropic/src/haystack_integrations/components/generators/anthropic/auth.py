# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, Type
import os

from haystack.core.errors import DeserializationError
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.lazy_imports import LazyImport

from anthropic import Anthropic, Stream

with LazyImport("Run pip install -U google-cloud-aiplatform \"anthropic[vertex]\"."):
    from anthropic import AnthropicVertex


with LazyImport("Run pip install anthropic."):
    from anthropic import Anthropic, Stream


class AnthropicModelAPIs(Enum):
    """
    Supported model apis for Anthropic.
    """

    ANTHROPIC_VERTEX = "anthropic_vertex"
    ANTHROPIC = "anthropic"

    def __str__(self):
        return self.value

    @staticmethod
    def from_class(model_class) -> "AnthropicModelAPIs":
        model_types = {
            AnthropicVertex: AnthropicModelAPIs.ANTHROPIC_VERTEX,
            Anthropic: AnthropicModelAPIs.ANTHROPIC,
        }
        return model_types[model_class]

@dataclass(frozen=True)
class AnthropicModelAdapter(ABC):
    """
    Base class for model APIs supported by AnthropicGenerator.
    """
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dictionary representation for serialization.
        """
        _fields = {}
        for _field in fields(self):
            if _field.type is Secret:
                _fields[_field.name] = getattr(self, _field.name).to_dict()
            else:
                _fields[_field.name] = getattr(self, _field.name)

        return {"type": str(AnthropicModelAPIs.from_class(self.__class__)), "init_parameters": _fields}
    

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AnthropicModelAdapter":
        """
        Converts a dictionary representation to an auth credentials object.
        """
        if "type" not in data:
            msg = "Missing 'type' in serialization data"
            raise DeserializationError(msg)

        auth_classes: Dict[str, Type[AnthropicModelAdapter]] = {
            str(AnthropicModelAPIs.ANTHROPIC): Anthropic,
            str(AnthropicModelAPIs.ANTHROPIC_VERTEX): AnthropicVertex,
        }

        return auth_classes[data["type"]]._from_dict(data)
    
    @classmethod
    @abstractmethod
    def _from_dict(cls, data: Dict[str, Any]):
        """
        Internal method to convert a dictionary representation to an auth credentials object.
        All subclasses must implement this method.
        """

    @abstractmethod
    def resolve_value(self):
        """
        Resolves all the secrets in the auth credentials object and returns the corresponding Weaviate object.
        All subclasses must implement this method.
        """

@dataclass (frozen=True)
class AnthropicAPI(AnthropicModelAdapter):
    """
    Model adapter for the Anthropic API. It will load the api_key from the environment variable `ANTHROPIC_API_KEY`.
    """
    api_key: Secret = field(default_factory=lambda: Secret.from_env_var("ANTHROPIC_API_KEY"))


    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Anthropic":
        deserialize_secrets_inplace(data["init_parameters"], ["api_key"])
        return cls(**data["init_parameters"])

    def resolve_value(self) -> Anthropic:
        return Anthropic(api_key=self.api_key.resolve_value())

@dataclass (frozen=True)
class AnthropicVertexAPI(AnthropicModelAdapter):
    """
    Model adapter for the Anthropic Vertex API. It will load the api_key from the environment variable `ANTHROPIC_API_KEY`, `REGION` and `PROJECT_ID`.
    """
    api_key: Secret = field(default_factory=lambda: Secret.from_env_var("ANTHROPIC_API_KEY"))
    region: str = field(default_factory=lambda: os.environ.get("REGION", "us-central1"))
    region: Secret = field(default_factory=lambda: Secret.from_env_var("REGION"))

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AnthropicVertex":
        deserialize_secrets_inplace(data["init_parameters"], ["api_key"])
        return cls(**data["init_parameters"])

    def resolve_value(self) -> AnthropicVertex:
        return AnthropicVertex(api_key=self.api_key.resolve_value())

