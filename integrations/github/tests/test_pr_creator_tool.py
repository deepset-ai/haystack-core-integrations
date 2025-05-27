# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.utils import Secret

from haystack_integrations.prompts.github.pr_creator_prompt import PR_CREATOR_PROMPT, PR_CREATOR_SCHEMA
from haystack_integrations.tools.github.pr_creator_tool import GitHubPRCreatorTool
from haystack_integrations.tools.github.utils import message_handler


class TestGitHubPRCreatorTool:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubPRCreatorTool()
        assert tool.name == "pr_creator"
        assert tool.description == PR_CREATOR_PROMPT
        assert tool.parameters == PR_CREATOR_SCHEMA
        assert tool.github_token == Secret.from_env_var("GITHUB_TOKEN")
        assert tool.raise_on_failure is True
        assert tool.outputs_to_string is None
        assert tool.inputs_from_state is None
        assert tool.outputs_to_state is None

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool_dict = {
            "type": "haystack_integrations.tools.github.pr_creator_tool.GitHubPRCreatorTool",
            "data": {
                "name": "pr_creator",
                "description": PR_CREATOR_PROMPT,
                "parameters": PR_CREATOR_SCHEMA,
                "github_token": {"env_vars": ["GITHUB_TOKEN"], "strict": True, "type": "env_var"},
                "raise_on_failure": True,
                "outputs_to_string": None,
                "inputs_from_state": None,
                "outputs_to_state": None,
            },
        }
        tool = GitHubPRCreatorTool.from_dict(tool_dict)
        assert tool.name == "pr_creator"
        assert tool.description == PR_CREATOR_PROMPT
        assert tool.parameters == PR_CREATOR_SCHEMA
        assert tool.github_token == Secret.from_env_var("GITHUB_TOKEN")
        assert tool.raise_on_failure is True
        assert tool.outputs_to_string is None
        assert tool.inputs_from_state is None
        assert tool.outputs_to_state is None

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubPRCreatorTool()
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.pr_creator_tool.GitHubPRCreatorTool"
        assert tool_dict["data"]["name"] == "pr_creator"
        assert tool_dict["data"]["description"] == PR_CREATOR_PROMPT
        assert tool_dict["data"]["parameters"] == PR_CREATOR_SCHEMA
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
        tool = GitHubPRCreatorTool(
            name="pr_creator",
            description="PR Creator Tool",
            parameters=PR_CREATOR_SCHEMA,
            github_token=Secret.from_env_var("GITHUB_TOKEN"),
            raise_on_failure=False,
            outputs_to_string={"handler": message_handler},
            inputs_from_state={"repository": "repo"},
            outputs_to_state={"documents": {"source": "docs", "handler": message_handler}},
        )
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.pr_creator_tool.GitHubPRCreatorTool"
        assert tool_dict["data"]["name"] == "pr_creator"
        assert tool_dict["data"]["description"] == "PR Creator Tool"
        assert tool_dict["data"]["parameters"] == PR_CREATOR_SCHEMA
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
            "type": "haystack_integrations.tools.github.pr_creator_tool.GitHubPRCreatorTool",
            "data": {
                "name": "pr_creator",
                "description": "PR Creator Tool",
                "parameters": PR_CREATOR_SCHEMA,
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
        tool = GitHubPRCreatorTool.from_dict(tool_dict)
        assert tool.name == "pr_creator"
        assert tool.description == "PR Creator Tool"
        assert tool.parameters == PR_CREATOR_SCHEMA
        assert tool.github_token == Secret.from_env_var("GITHUB_TOKEN")
        assert tool.raise_on_failure is False
        assert tool.outputs_to_string["handler"] == message_handler
        assert tool.inputs_from_state == {"repository": "repo"}
        assert tool.outputs_to_state["documents"]["source"] == "docs"
        assert tool.outputs_to_state["documents"]["handler"] == message_handler
