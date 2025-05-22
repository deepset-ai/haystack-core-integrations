# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack_integrations.prompts.github.issue_viewer_prompt import ISSUE_VIEWER_PROMPT, ISSUE_VIEWER_SCHEMA
from haystack_integrations.tools.github.issue_viewer_tool import GitHubIssueViewerTool


class TestGitHubIssueViewerTool:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubIssueViewerTool()
        assert tool.name == "issue_viewer"
        assert tool.description == ISSUE_VIEWER_PROMPT
        assert tool.parameters == ISSUE_VIEWER_SCHEMA
        assert tool.retry_attempts == 2

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool_dict = {
            "type": "haystack_integrations.tools.github.issue_viewer_tool.GitHubIssueViewerTool",
            "init_parameters": {
                "name": "issue_viewer",
                "description": ISSUE_VIEWER_PROMPT,
                "parameters": ISSUE_VIEWER_SCHEMA,
                "github_token": None,
                "raise_on_failure": True,
                "retry_attempts": 2,
            },
        }
        tool = GitHubIssueViewerTool.from_dict(tool_dict)
        assert tool.name == "issue_viewer"
        assert tool.description == ISSUE_VIEWER_PROMPT
        assert tool.parameters == ISSUE_VIEWER_SCHEMA
        assert tool.github_token is None
        assert tool.raise_on_failure
        assert tool.retry_attempts == 2

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubIssueViewerTool()
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.issue_viewer_tool.GitHubIssueViewerTool"
        assert tool_dict["init_parameters"]["name"] == "issue_viewer"
        assert tool_dict["init_parameters"]["description"] == ISSUE_VIEWER_PROMPT
        assert tool_dict["init_parameters"]["parameters"] == ISSUE_VIEWER_SCHEMA
        assert tool_dict["init_parameters"]["github_token"] is None
        assert tool_dict["init_parameters"]["raise_on_failure"]
        assert tool_dict["init_parameters"]["retry_attempts"] == 2
