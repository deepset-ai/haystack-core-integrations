# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.utils import Secret

from haystack_integrations.prompts.github.issue_commenter_prompt import ISSUE_COMMENTER_PROMPT, ISSUE_COMMENTER_SCHEMA
from haystack_integrations.tools.github.issue_commenter_tool import GitHubIssueCommenterTool


class TestGitHubIssueCommenterTool:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubIssueCommenterTool()
        assert tool.name == "issue_commenter"
        assert tool.description == ISSUE_COMMENTER_PROMPT
        assert tool.parameters == ISSUE_COMMENTER_SCHEMA
        assert tool.retry_attempts == 2

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool_dict = {
            "type": "haystack_integrations.tools.github.issue_commenter_tool.GitHubIssueCommenterTool",
            "init_parameters": {
                "name": "issue_commenter",
                "description": ISSUE_COMMENTER_PROMPT,
                "parameters": ISSUE_COMMENTER_SCHEMA,
                "github_token": {"env_vars": ["GITHUB_TOKEN"], "strict": True, "type": "env_var"},
                "raise_on_failure": True,
                "retry_attempts": 2,
            },
        }
        tool = GitHubIssueCommenterTool.from_dict(tool_dict)
        assert tool.name == "issue_commenter"
        assert tool.description == ISSUE_COMMENTER_PROMPT
        assert tool.parameters == ISSUE_COMMENTER_SCHEMA
        assert tool.github_token == Secret.from_env_var("GITHUB_TOKEN")
        assert tool.raise_on_failure
        assert tool.retry_attempts == 2

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubIssueCommenterTool()
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.issue_commenter_tool.GitHubIssueCommenterTool"
        assert tool_dict["init_parameters"]["name"] == "issue_commenter"
        assert tool_dict["init_parameters"]["description"] == ISSUE_COMMENTER_PROMPT
        assert tool_dict["init_parameters"]["parameters"] == ISSUE_COMMENTER_SCHEMA
        assert tool_dict["init_parameters"]["github_token"] == {
            "env_vars": ["GITHUB_TOKEN"],
            "strict": True,
            "type": "env_var",
        }
        assert tool_dict["init_parameters"]["raise_on_failure"]
        assert tool_dict["init_parameters"]["retry_attempts"] == 2
