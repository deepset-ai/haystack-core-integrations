# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.utils import Secret

from haystack_integrations.prompts.github.issue_commenter_prompt import ISSUE_COMMENTER_PROMPT, ISSUE_COMMENTER_SCHEMA
from haystack_integrations.tools.github.issue_commenter_tool import GitHubIssueCommenterTool
from haystack_integrations.tools.github.utils import message_handler


class TestGitHubIssueCommenterTool:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubIssueCommenterTool()
        assert tool.name == "issue_commenter"
        assert tool.description == ISSUE_COMMENTER_PROMPT
        assert tool.parameters == ISSUE_COMMENTER_SCHEMA
        assert tool.retry_attempts == 2
        assert tool.outputs_to_string is None
        assert tool.inputs_from_state is None
        assert tool.outputs_to_state is None

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool_dict = {
            "type": "haystack_integrations.tools.github.issue_commenter_tool.GitHubIssueCommenterTool",
            "data": {
                "name": "issue_commenter",
                "description": ISSUE_COMMENTER_PROMPT,
                "parameters": ISSUE_COMMENTER_SCHEMA,
                "github_token": {"env_vars": ["GITHUB_TOKEN"], "strict": True, "type": "env_var"},
                "raise_on_failure": True,
                "retry_attempts": 2,
                "outputs_to_string": None,
                "inputs_from_state": None,
                "outputs_to_state": None,
            },
        }
        tool = GitHubIssueCommenterTool.from_dict(tool_dict)
        assert tool.name == "issue_commenter"
        assert tool.description == ISSUE_COMMENTER_PROMPT
        assert tool.parameters == ISSUE_COMMENTER_SCHEMA
        assert tool.github_token == Secret.from_env_var("GITHUB_TOKEN")
        assert tool.raise_on_failure
        assert tool.retry_attempts == 2
        assert tool.outputs_to_string is None
        assert tool.inputs_from_state is None
        assert tool.outputs_to_state is None

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubIssueCommenterTool()
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.issue_commenter_tool.GitHubIssueCommenterTool"
        assert tool_dict["data"]["name"] == "issue_commenter"
        assert tool_dict["data"]["description"] == ISSUE_COMMENTER_PROMPT
        assert tool_dict["data"]["parameters"] == ISSUE_COMMENTER_SCHEMA
        assert tool_dict["data"]["github_token"] == {
            "env_vars": ["GITHUB_TOKEN"],
            "strict": True,
            "type": "env_var",
        }
        assert tool_dict["data"]["raise_on_failure"]
        assert tool_dict["data"]["retry_attempts"] == 2
        assert tool_dict["data"]["outputs_to_string"] is None
        assert tool_dict["data"]["inputs_from_state"] is None
        assert tool_dict["data"]["outputs_to_state"] is None

    def test_to_dict_with_extra_params(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubIssueCommenterTool(
            name="test_issue_commenter",
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
        assert tool_dict["type"] == "haystack_integrations.tools.github.issue_commenter_tool.GitHubIssueCommenterTool"
        assert tool_dict["data"]["name"] == "test_issue_commenter"
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
            "type": "haystack_integrations.tools.github.issue_commenter_tool.GitHubIssueCommenterTool",
            "data": {
                "name": "test_issue_commenter",
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
        tool = GitHubIssueCommenterTool.from_dict(tool_dict)
        assert tool.name == "test_issue_commenter"
        assert tool.description == "Test description"
        assert tool.parameters == {"type": "object", "properties": {}}
        assert tool.github_token is None
        assert tool.raise_on_failure is False
        assert tool.retry_attempts == 3
        assert tool.outputs_to_string["handler"] == message_handler
        assert tool.inputs_from_state == {"repository": "repo"}
        assert tool.outputs_to_state["documents"]["source"] == "docs"
        assert tool.outputs_to_state["documents"]["handler"] == message_handler
