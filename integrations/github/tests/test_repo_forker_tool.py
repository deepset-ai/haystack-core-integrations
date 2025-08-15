# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.utils import Secret

from haystack_integrations.prompts.github.repo_forker_prompt import REPO_FORKER_PROMPT, REPO_FORKER_SCHEMA
from haystack_integrations.tools.github.repo_forker_tool import GitHubRepoForkerTool
from haystack_integrations.tools.github.utils import message_handler


class TestGitHubRepoForkerTool:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        tool = GitHubRepoForkerTool()
        assert tool.name == "repo_forker"
        assert tool.description == REPO_FORKER_PROMPT
        assert tool.parameters == REPO_FORKER_SCHEMA
        assert tool.github_token == Secret.from_env_var("GITHUB_TOKEN")
        assert tool.raise_on_failure is True
        assert tool.outputs_to_string is None
        assert tool.inputs_from_state is None
        assert tool.outputs_to_state is None

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool_dict = {
            "type": "haystack_integrations.tools.github.repo_forker_tool.GitHubRepoForkerTool",
            "data": {
                "name": "repo_forker",
                "description": REPO_FORKER_PROMPT,
                "parameters": REPO_FORKER_SCHEMA,
                "github_token": {"env_vars": ["GITHUB_TOKEN"], "strict": True, "type": "env_var"},
                "raise_on_failure": True,
                "outputs_to_string": None,
                "inputs_from_state": None,
                "outputs_to_state": None,
            },
        }
        tool = GitHubRepoForkerTool.from_dict(tool_dict)
        assert tool.name == "repo_forker"
        assert tool.description == REPO_FORKER_PROMPT
        assert tool.parameters == REPO_FORKER_SCHEMA
        assert tool.github_token == Secret.from_env_var("GITHUB_TOKEN")
        assert tool.raise_on_failure is True
        assert tool.outputs_to_string is None
        assert tool.inputs_from_state is None
        assert tool.outputs_to_state is None

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubRepoForkerTool()
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.repo_forker_tool.GitHubRepoForkerTool"
        assert tool_dict["data"]["name"] == "repo_forker"
        assert tool_dict["data"]["description"] == REPO_FORKER_PROMPT
        assert tool_dict["data"]["parameters"] == REPO_FORKER_SCHEMA
        assert tool_dict["data"]["github_token"] == {
            "env_vars": ["GITHUB_TOKEN"],
            "strict": True,
            "type": "env_var",
        }
        assert tool_dict["data"]["raise_on_failure"] is True
        assert tool_dict["data"]["outputs_to_string"] is None
        assert tool_dict["data"]["inputs_from_state"] is None
        assert tool_dict["data"]["outputs_to_state"] is None

    def test_to_dict_with_extra_params(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubRepoForkerTool(
            github_token=Secret.from_env_var("GITHUB_TOKEN"),
            raise_on_failure=False,
            outputs_to_string={"source": "docs", "handler": message_handler},
            inputs_from_state={"repository": "repo"},
            outputs_to_state={"documents": {"source": "docs", "handler": message_handler}},
        )
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.repo_forker_tool.GitHubRepoForkerTool"
        assert tool_dict["data"]["name"] == "repo_forker"
        assert tool_dict["data"]["description"] == REPO_FORKER_PROMPT
        assert tool_dict["data"]["parameters"] == REPO_FORKER_SCHEMA
        assert tool_dict["data"]["github_token"] == {
            "env_vars": ["GITHUB_TOKEN"],
            "strict": True,
            "type": "env_var",
        }
        assert tool_dict["data"]["raise_on_failure"] is False
        assert (
            tool_dict["data"]["outputs_to_string"]["handler"]
            == "haystack_integrations.tools.github.utils.message_handler"
        )
        assert tool_dict["data"]["inputs_from_state"] == {"repository": "repo"}
        assert tool_dict["data"]["outputs_to_state"]["documents"]["source"] == "docs"
        assert (
            tool_dict["data"]["outputs_to_state"]["documents"]["handler"]
            == "haystack_integrations.tools.github.utils.message_handler"
        )

    def test_from_dict_with_extra_params(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool_dict = {
            "type": "haystack_integrations.tools.github.repo_forker_tool.GitHubRepoForkerTool",
            "data": {
                "name": "repo_forker",
                "description": REPO_FORKER_PROMPT,
                "parameters": REPO_FORKER_SCHEMA,
                "github_token": {"env_vars": ["GITHUB_TOKEN"], "strict": True, "type": "env_var"},
                "raise_on_failure": False,
                "outputs_to_string": {"handler": "haystack_integrations.tools.github.utils.message_handler"},
                "inputs_from_state": {"repository": "repo"},
                "outputs_to_state": {
                    "documents": {
                        "source": "docs",
                        "handler": "haystack_integrations.tools.github.utils.message_handler",
                    }
                },
            },
        }
        tool = GitHubRepoForkerTool.from_dict(tool_dict)
        assert tool.name == "repo_forker"
        assert tool.description == REPO_FORKER_PROMPT
        assert tool.parameters == REPO_FORKER_SCHEMA
        assert tool.github_token == Secret.from_env_var("GITHUB_TOKEN")
        assert tool.raise_on_failure is False
        assert tool.outputs_to_string["handler"] == message_handler
        assert tool.inputs_from_state == {"repository": "repo"}
        assert tool.outputs_to_state["documents"]["source"] == "docs"
        assert tool.outputs_to_state["documents"]["handler"] == message_handler
