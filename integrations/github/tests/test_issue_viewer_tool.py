# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack_integrations.prompts.github.issue_viewer_prompt import ISSUE_VIEWER_PROMPT, ISSUE_VIEWER_SCHEMA
from haystack_integrations.tools.github.issue_viewer_tool import GitHubIssueViewerTool
from haystack_integrations.tools.github.utils import message_handler


class TestGitHubIssueViewerTool:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubIssueViewerTool()
        assert tool.name == "issue_viewer"
        assert tool.description == ISSUE_VIEWER_PROMPT
        assert tool.parameters == ISSUE_VIEWER_SCHEMA
        assert tool.github_token is None
        assert tool.raise_on_failure is True
        assert tool.retry_attempts == 2
        assert tool.outputs_to_string is None
        assert tool.inputs_from_state is None
        assert tool.outputs_to_state is None

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool_dict = {
            "type": "haystack_integrations.tools.github.issue_viewer_tool.GitHubIssueViewerTool",
            "data": {
                "name": "test_issue_viewer",
                "description": "Test description",
                "parameters": {"type": "object", "properties": {}},
                "github_token": None,
                "raise_on_failure": True,
                "retry_attempts": 2,
                "outputs_to_string": None,
                "inputs_from_state": None,
                "outputs_to_state": None,
            },
        }
        tool = GitHubIssueViewerTool.from_dict(tool_dict)
        assert tool.name == "test_issue_viewer"
        assert tool.description == "Test description"
        assert tool.parameters == {"type": "object", "properties": {}}
        assert tool.github_token is None
        assert tool.raise_on_failure
        assert tool.retry_attempts == 2
        assert tool.outputs_to_string is None
        assert tool.inputs_from_state is None
        assert tool.outputs_to_state is None

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubIssueViewerTool()
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.issue_viewer_tool.GitHubIssueViewerTool"
        assert tool_dict["data"]["name"] == "issue_viewer"
        assert tool_dict["data"]["description"] == ISSUE_VIEWER_PROMPT
        assert tool_dict["data"]["parameters"] == ISSUE_VIEWER_SCHEMA
        assert tool_dict["data"]["github_token"] is None
        assert tool_dict["data"]["raise_on_failure"] is True
        assert tool_dict["data"]["retry_attempts"] == 2
        assert tool_dict["data"]["outputs_to_string"] is None
        assert tool_dict["data"]["inputs_from_state"] is None
        assert tool_dict["data"]["outputs_to_state"] is None

    def test_to_dict_with_extra_params(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubIssueViewerTool(
            name="test_issue_viewer",
            description="Test description",
            parameters={"type": "object", "properties": {}},
            github_token=None,
            raise_on_failure=False,
            retry_attempts=3,
            outputs_to_string={"handler": message_handler},
            inputs_from_state={"repository": "repo"},
            outputs_to_state={"documents": {"source": "docs", "handler": message_handler}},
        )
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.issue_viewer_tool.GitHubIssueViewerTool"
        assert tool_dict["data"]["name"] == "test_issue_viewer"
        assert tool_dict["data"]["description"] == "Test description"
        assert tool_dict["data"]["parameters"] == {"type": "object", "properties": {}}
        assert tool_dict["data"]["github_token"] is None
        assert tool_dict["data"]["raise_on_failure"] is False
        assert tool_dict["data"]["retry_attempts"] == 3
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
            "type": "haystack_integrations.tools.github.issue_viewer_tool.GitHubIssueViewerTool",
            "data": {
                "name": "test_issue_viewer",
                "description": "Test description",
                "parameters": {"type": "object", "properties": {}},
                "github_token": None,
                "raise_on_failure": False,
                "retry_attempts": 3,
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
        tool = GitHubIssueViewerTool.from_dict(tool_dict)
        assert tool.name == "test_issue_viewer"
        assert tool.description == "Test description"
        assert tool.parameters == {"type": "object", "properties": {}}
        assert tool.github_token is None
        assert tool.raise_on_failure is False
        assert tool.retry_attempts == 3
        assert tool.outputs_to_string["handler"] == message_handler
        assert tool.inputs_from_state == {"repository": "repo"}
        assert tool.outputs_to_state["documents"]["source"] == "docs"
        assert tool.outputs_to_state["documents"]["handler"] == message_handler
