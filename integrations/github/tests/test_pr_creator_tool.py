# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.utils import Secret

from haystack_integrations.prompts.github.pr_creator_prompt import PR_CREATOR_PROMPT, PR_CREATOR_SCHEMA
from haystack_integrations.tools.github.pr_creator_tool import GitHubPRCreatorTool


class TestGitHubPRCreatorTool:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubPRCreatorTool()
        assert tool.name == "pr_creator"
        assert tool.description == PR_CREATOR_PROMPT
        assert tool.parameters == PR_CREATOR_SCHEMA

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool_dict = {
            "type": "haystack_integrations.tools.github.pr_creator_tool.GitHubPRCreatorTool",
            "init_parameters": {
                "name": "pr_creator",
                "description": PR_CREATOR_PROMPT,
                "parameters": PR_CREATOR_SCHEMA,
                "github_token": {"env_vars": ["GITHUB_TOKEN"], "strict": True, "type": "env_var"},
                "raise_on_failure": True,
            },
        }
        tool = GitHubPRCreatorTool.from_dict(tool_dict)
        assert tool.name == "pr_creator"
        assert tool.description == PR_CREATOR_PROMPT
        assert tool.parameters == PR_CREATOR_SCHEMA
        assert tool.github_token == Secret.from_env_var("GITHUB_TOKEN")
        assert tool.raise_on_failure

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubPRCreatorTool()
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.pr_creator_tool.GitHubPRCreatorTool"
        assert tool_dict["init_parameters"]["name"] == "pr_creator"
        assert tool_dict["init_parameters"]["description"] == PR_CREATOR_PROMPT
        assert tool_dict["init_parameters"]["parameters"] == PR_CREATOR_SCHEMA
        assert tool_dict["init_parameters"]["github_token"] == {
            "env_vars": ["GITHUB_TOKEN"],
            "strict": True,
            "type": "env_var",
        }
        assert tool_dict["init_parameters"]["raise_on_failure"]
