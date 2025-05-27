# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

from haystack_integrations.prompts.github.file_editor_prompt import FILE_EDITOR_PROMPT, FILE_EDITOR_SCHEMA
from haystack_integrations.tools.github.file_editor_tool import GitHubFileEditorTool


def custom_handler(value):
    """A test handler function for serialization tests."""
    return f"Processed: {value}"


# Make the handler available at module level
__all__ = ["custom_handler"]


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

    def test_to_dict_with_extra_params(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        tool = GitHubFileEditorTool(
            outputs_to_string={"source": "result", "handler": custom_handler},
            inputs_from_state={"repo_state": "repo"},
            outputs_to_state={"file_content": {"source": "content", "handler": custom_handler}},
        )

        tool_dict = tool.to_dict()
        assert tool_dict["init_parameters"]["outputs_to_string"] == {
            "source": "result",
            "handler": "tests.test_file_editor_tool.custom_handler",
        }
        assert tool_dict["init_parameters"]["inputs_from_state"] == {"repo_state": "repo"}
        assert tool_dict["init_parameters"]["outputs_to_state"] == {
            "file_content": {"source": "content", "handler": "tests.test_file_editor_tool.custom_handler"}
        }

    def test_from_dict_with_extra_params(self, monkeypatch):
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
                "outputs_to_string": {"source": "result", "handler": "tests.test_file_editor_tool.custom_handler"},
                "inputs_from_state": {"repo_state": "repo"},
                "outputs_to_state": {
                    "file_content": {"source": "content", "handler": "tests.test_file_editor_tool.custom_handler"}
                },
            },
        }

        tool = GitHubFileEditorTool.from_dict(tool_dict)
        assert tool.outputs_to_string["source"] == "result"
        assert tool.outputs_to_string["handler"] == custom_handler
        assert tool.inputs_from_state == {"repo_state": "repo"}
        assert tool.outputs_to_state["file_content"]["source"] == "content"
        assert tool.outputs_to_state["file_content"]["handler"] == custom_handler

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

        assert pipeline_dict == {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
                "agent": {
                    "type": "haystack.components.agents.agent.Agent",
                    "init_parameters": {
                        "chat_generator": {
                            "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                            "init_parameters": {
                                "model": "gpt-4o-mini",
                                "streaming_callback": None,
                                "api_base_url": None,
                                "organization": None,
                                "generation_kwargs": {},
                                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                                "timeout": None,
                                "max_retries": None,
                                "tools": None,
                                "tools_strict": False,
                                "http_client_kwargs": None,
                            },
                        },
                        "tools": [
                            {
                                "type": "haystack_integrations.tools.github.file_editor_tool.GitHubFileEditorTool",
                                "init_parameters": {
                                    "name": "file_editor",
                                    "description": FILE_EDITOR_PROMPT,
                                    "parameters": FILE_EDITOR_SCHEMA,
                                    "github_token": {"env_vars": ["GITHUB_TOKEN"], "strict": True, "type": "env_var"},
                                    "repo": None,
                                    "branch": "main",
                                    "raise_on_failure": True,
                                    "outputs_to_string": None,
                                    "inputs_from_state": None,
                                    "outputs_to_state": None,
                                },
                            }
                        ],
                        "system_prompt": None,
                        "exit_conditions": ["text"],
                        "state_schema": {},
                        "max_agent_steps": 100,
                        "raise_on_tool_invocation_failure": False,
                        "streaming_callback": None,
                    },
                }
            },
            "connections": [],
            "connection_type_validation": True,
        }

        deserialized_pipeline = Pipeline.from_dict(pipeline_dict)
        deserialized_components = [instance for _, instance in deserialized_pipeline.graph.nodes(data="instance")]
        deserialized_agent = deserialized_components[0]
        assert isinstance(deserialized_agent, Agent)

        agent_tools = deserialized_agent.tools
        assert len(agent_tools) == 1
        assert isinstance(agent_tools[0], GitHubFileEditorTool)
        assert agent_tools[0].name == "file_editor"

        # Verify the tool's parameters were preserved
        assert agent_tools[0].name == "file_editor"
        assert agent_tools[0].description == FILE_EDITOR_PROMPT
        assert agent_tools[0].parameters == FILE_EDITOR_SCHEMA
        assert agent_tools[0].github_token == Secret.from_env_var("GITHUB_TOKEN")
        assert agent_tools[0].repo is None
        assert agent_tools[0].branch == "main"
        assert agent_tools[0].raise_on_failure
