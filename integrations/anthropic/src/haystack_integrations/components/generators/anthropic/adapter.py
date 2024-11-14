# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, Type

from haystack.core.errors import DeserializationError
from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret

with LazyImport('Run pip install -U google-cloud-aiplatform "anthropic[vertex]".'):
    from anthropic import AnthropicVertex


with LazyImport("Run pip install anthropic."):
    from anthropic import Anthropic


@dataclass(frozen=True)
class BaseAdapter(ABC):
    """
    Base class for model APIs supported by AnthropicGenerator.
    """

    TYPE: ClassVar[str]

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

        return {"type": self.TYPE, "init_parameters": _fields}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BaseAdapter":
        """
        Converts a dictionary representation to an adapter object.
        """
        
    @abstractmethod
    def client(self):
        """
        Resolves all the secrets and evironment variables and returns the corresponding adapter object.
        All subclasses must implement this method.
        """

    @abstractmethod
    def set_model(self, model):
        """
        Sets the model name in the format required by the API.
        """


@dataclass(frozen=True)
class AnthropicAdapter(BaseAdapter):
    """
    Model adapter for the Anthropic API. It will load the api_key from the environment variable `ANTHROPIC_API_KEY`.
    """

    api_key: Secret
    TYPE = "anthropic"

    def client(self) -> Anthropic:
        return Anthropic(api_key=self.api_key.resolve_value())

    def set_model(self, model) -> str:
        return model  # default model name format is correct for Anthropic API


@dataclass(frozen=True)
class AnthropicVertexAdapter(BaseAdapter):
    """
    Model adapter for the Anthropic Vertex API. It authenticate using GCP authentication and select
    `REGION` and `PROJECT_ID` from the environment variable.
    """

    TYPE = "anthropic_vertex"
    region: str
    project_id: str

    def client(self) -> AnthropicVertex:
        return AnthropicVertex(region=self.region, project_id=self.project_id)

    def set_model(self, model) -> str:
        """
        Converts the model name to the format required by the Anthropic Vertex API.
        AnthropicVertex requires model name in the format `claude-3-sonnet@20240229`
        instead of `claude-3-sonnet-20240229`.
        """
        return model[::-1].replace("-", "@", 1)[::-1]
