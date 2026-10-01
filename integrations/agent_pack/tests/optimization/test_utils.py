import pytest
import yaml
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.errors import DeserializationError
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import ComponentTool

from haystack_integrations.agent_pack.optimization.utils import (
    _configuration_id,
    dump_agent,
    load_agent,
)

from .test_workspace import agent_yaml


class TestDumpAndLoad:
    def test_prompts_stay_editable(self):
        prompt = "First line — literal Unicode.\nSecond line: keep this exactly.\n"
        original = Agent(chat_generator=MockChatGenerator(), system_prompt=prompt)
        serialized = dump_agent(original)
        assert "system_prompt: |" in serialized
        assert "First line — literal Unicode." in serialized
        assert load_agent(serialized).system_prompt == prompt

    def test_duplicate_keys_and_untrusted_classes(self):
        with pytest.raises(yaml.constructor.ConstructorError, match="Duplicate YAML key"):
            load_agent("components: {}\ncomponents: {}\n")
        with pytest.raises(DeserializationError):
            load_agent("components:\n  agent:\n    type: subprocess.Popen\n    init_parameters: {}\nconnections: []")


class TestConfigurationId:
    def test_store_identity_is_kept(self):
        first = InMemoryDocumentStore(index="first")
        second = InMemoryDocumentStore(index="second")

        def configured(store):
            return agent_yaml(
                Agent(
                    chat_generator=MockChatGenerator(),
                    tools=[ComponentTool(component=InMemoryBM25Retriever(document_store=store))],
                )
            )

        assert _configuration_id(configured(first)) != _configuration_id(configured(second))
