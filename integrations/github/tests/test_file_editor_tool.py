# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

from haystack_integrations.prompts.github.file_editor_tool import FILE_EDITOR_PROMPT, FILE_EDITOR_SCHEMA
from haystack_integrations.tools.github.file_editor_tool import GitHubFileEditorTool


class TestGitHubFileEditorTool:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubFileEditorTool()
        assert tool.name == "file_editor"
        assert tool.description == FILE_EDITOR_PROMPT
        assert tool.parameters == FILE_EDITOR_SCHEMA

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool_dict = {
            "type": "haystack_integrations.tools.github.file_editor_tool.GitHubFileEditorTool",
            "init_parameters": {
                "name": "file_editor",
                "description": FILE_EDITOR_PROMPT,
                "parameters": FILE_EDITOR_SCHEMA,
                "github_token": {"env_vars": ["GITHUB_TOKEN"], "strict": True, "type": "env_var"},
                "repo": None,
                "branch": "main",
                "raise_on_failure": True,
            },
        }
        tool = GitHubFileEditorTool.from_dict(tool_dict)
        assert tool.name == "file_editor"
        assert tool.description == FILE_EDITOR_PROMPT
        assert tool.parameters == FILE_EDITOR_SCHEMA
        assert tool.github_token == Secret.from_env_var("GITHUB_TOKEN")
        assert tool.repo is None
        assert tool.branch == "main"
        assert tool.raise_on_failure

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubFileEditorTool()
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.file_editor_tool.GitHubFileEditorTool"
        assert tool_dict["init_parameters"]["name"] == "file_editor"
        assert tool_dict["init_parameters"]["description"] == FILE_EDITOR_PROMPT
        assert tool_dict["init_parameters"]["parameters"] == FILE_EDITOR_SCHEMA
        assert tool_dict["init_parameters"]["github_token"] == {
            "env_vars": ["GITHUB_TOKEN"],
            "strict": True,
            "type": "env_var",
        }
        assert tool_dict["init_parameters"]["repo"] is None
        assert tool_dict["init_parameters"]["branch"] == "main"
        assert tool_dict["init_parameters"]["raise_on_failure"]

    def test_pipeline_serialization(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        monkeypatch.setenv("OPENAI_API_KEY", "test-token")

        file_editor = GitHubFileEditorTool()

        agent = Agent(
            chat_generator=OpenAIChatGenerator(),
            tools=[file_editor],
        )

        pipeline = Pipeline()
        pipeline.add_component("agent", agent)

        pipeline_dict = pipeline.to_dict()

        deserialized_pipeline = Pipeline.from_dict(pipeline_dict)

        deserialized_components = [instance for _, instance in deserialized_pipeline.graph.nodes(data="instance")]

        deserialized_agent_component = deserialized_components[0]

        assert isinstance(deserialized_agent_component, Agent)

        agent_tools = deserialized_agent_component.tools
        assert len(agent_tools) == 1
        assert agent_tools[0].name == "file_editor"
        assert isinstance(agent_tools[0], GitHubFileEditorTool)

        # Verify the tool's parameters were preserved
        assert agent_tools[0].name == "file_editor"
        assert agent_tools[0].description == FILE_EDITOR_PROMPT
        assert agent_tools[0].parameters == FILE_EDITOR_SCHEMA
        assert agent_tools[0].github_token == Secret.from_env_var("GITHUB_TOKEN")
        assert agent_tools[0].repo is None
        assert agent_tools[0].branch == "main"
        assert agent_tools[0].raise_on_failure
