# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from typing import Any

import yaml
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.marshal import YamlMarshaller


class _ReadableDumper(yaml.SafeDumper):
    pass


def _represent_string(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    """Dump a multiline string as a `|` block scalar and any other string as a plain scalar."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


_ReadableDumper.add_representer(str, _represent_string)


class _ReadableYamlMarshaller(YamlMarshaller):
    """Haystack YAML marshaller that dumps multiline strings as `|` blocks and keeps Unicode unescaped."""

    def marshal(self, dict_: dict[str, Any]) -> str:
        """Keep multiline prompts editable without escaped newlines or Unicode."""
        return yaml.dump(dict_, Dumper=_ReadableDumper, allow_unicode=True, width=120)


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML `SafeLoader` that rejects a mapping with a duplicate key."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        """Raise a `ConstructorError` when a mapping repeats a key."""
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        # The error reaches the optimizer through validate_config, telling it an edit left a duplicate key
        if len(keys) != len(set(keys)):
            raise yaml.constructor.ConstructorError(None, None, "Duplicate YAML key", node.start_mark)
        return super().construct_mapping(node, deep=deep)


def _configuration_id(text: str) -> str:
    """Hash parsed configuration, retaining resource identities but ignoring YAML formatting."""
    # _UniqueKeyLoader subclasses SafeLoader; it only adds duplicate-key rejection.
    return content_digest(payload=json.dumps(yaml.load(text, Loader=_UniqueKeyLoader), sort_keys=True))  # noqa: S506


def content_digest(payload: str) -> str:
    """
    Return a short, stable digest of serialized content.

    :param payload: The serialized content to identify.
    :returns: A twelve-character hexadecimal digest.
    """
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def dump_pipeline(pipeline: Pipeline) -> str:
    """Emit readable Haystack Pipeline YAML."""
    return pipeline.dumps(marshaller=_ReadableYamlMarshaller())


def dump_agent(agent: Agent) -> str:
    """Wrap an independent copy of the Agent and emit readable Haystack Pipeline YAML."""
    pipeline = Pipeline()
    pipeline.add_component("agent", agent.clone())
    return dump_pipeline(pipeline=pipeline)


def load_pipeline(text: str) -> Pipeline:
    """Load a Pipeline using Haystack's deserialization security, rejecting duplicate keys first."""
    # Haystack's loader keeps the last of any repeated key, so duplicates are checked first
    yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass
    return Pipeline.loads(text)


def load_agent(text: str) -> Agent:
    """Load an Agent from its one-component Pipeline using Haystack's deserialization security."""
    data = yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - SafeLoader subclass
    if not isinstance(data, dict) or set(data.get("components", {})) != {"agent"} or data.get("connections"):
        msg = "The outer Pipeline must contain exactly one component named 'agent' and no connections."
        raise ValueError(msg)
    agent = load_pipeline(text).get_component("agent")
    if not isinstance(agent, Agent):
        msg = "The 'agent' component must be a Haystack Agent."
        raise ValueError(msg)
    return agent
