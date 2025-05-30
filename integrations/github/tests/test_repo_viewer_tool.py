# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack_integrations.prompts.github.repo_viewer_prompt import REPO_VIEWER_PROMPT, REPO_VIEWER_SCHEMA
from haystack_integrations.tools.github.repo_viewer_tool import GitHubRepoViewerTool
from haystack_integrations.tools.github.utils import message_handler


class TestGitHubRepoViewerTool:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubRepoViewerTool()
        assert tool.name == "repo_viewer"
        assert tool.description == REPO_VIEWER_PROMPT
        assert tool.parameters == REPO_VIEWER_SCHEMA
        assert tool.max_file_size == 1_000_000
        assert tool.github_token is None
        assert tool.repo is None
        assert tool.branch == "main"
        assert tool.raise_on_failure
        assert tool.outputs_to_string == {"source": "documents", "handler": message_handler}
        assert tool.inputs_from_state == {}
        assert tool.outputs_to_state == {"documents": {"source": "documents"}}

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool_dict = {
            "type": "haystack_integrations.tools.github.repo_viewer_tool.GitHubRepoViewerTool",
            "data": {
                "name": "repo_viewer",
                "description": REPO_VIEWER_PROMPT,
                "parameters": REPO_VIEWER_SCHEMA,
                "github_token": None,
                "repo": None,
                "branch": "main",
                "raise_on_failure": True,
                "max_file_size": 1_000_000,
                "outputs_to_string": {
                    "source": "documents",
                    "handler": "haystack_integrations.tools.github.utils.message_handler",
                },
                "inputs_from_state": {},
                "outputs_to_state": {"documents": {"source": "documents"}},
            },
        }
        tool = GitHubRepoViewerTool.from_dict(tool_dict)
        assert tool.name == "repo_viewer"
        assert tool.description == REPO_VIEWER_PROMPT
        assert tool.parameters == REPO_VIEWER_SCHEMA
        assert tool.github_token is None
        assert tool.repo is None
        assert tool.branch == "main"
        assert tool.raise_on_failure
        assert tool.max_file_size == 1_000_000
        assert tool.outputs_to_string["source"] == "documents"
        assert tool.outputs_to_string["handler"] == message_handler
        assert tool.inputs_from_state == {}
        assert tool.outputs_to_state == {"documents": {"source": "documents"}}

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubRepoViewerTool()
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.repo_viewer_tool.GitHubRepoViewerTool"
        assert tool_dict["data"]["name"] == "repo_viewer"
        assert tool_dict["data"]["description"] == REPO_VIEWER_PROMPT
        assert tool_dict["data"]["parameters"] == REPO_VIEWER_SCHEMA
        assert tool_dict["data"]["github_token"] is None
        assert tool_dict["data"]["repo"] is None
        assert tool_dict["data"]["branch"] == "main"
        assert tool_dict["data"]["raise_on_failure"]
        assert tool_dict["data"]["max_file_size"] == 1_000_000
        assert tool_dict["data"]["outputs_to_string"] == {
            "source": "documents",
            "handler": "haystack_integrations.tools.github.utils.message_handler",
        }
        assert tool_dict["data"]["inputs_from_state"] == {}
        assert tool_dict["data"]["outputs_to_state"] == {"documents": {"source": "documents"}}

    def test_to_dict_with_extra_params(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        tool = GitHubRepoViewerTool(
            outputs_to_string={"source": "result", "handler": message_handler},
            inputs_from_state={"repo_state": "repo"},
            outputs_to_state={"file_content": {"source": "content", "handler": message_handler}},
        )

        tool_dict = tool.to_dict()
        assert tool_dict["data"]["outputs_to_string"] == {
            "source": "result",
            "handler": "haystack_integrations.tools.github.utils.message_handler",
        }
        assert tool_dict["data"]["inputs_from_state"] == {"repo_state": "repo"}
        assert tool_dict["data"]["outputs_to_state"] == {
            "file_content": {
                "source": "content",
                "handler": "haystack_integrations.tools.github.utils.message_handler",
            },
        }

    def test_from_dict_with_extra_params(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        tool_dict = {
            "type": "haystack_integrations.tools.github.repo_viewer_tool.GitHubRepoViewerTool",
            "data": {
                "name": "repo_viewer",
                "description": REPO_VIEWER_PROMPT,
                "parameters": REPO_VIEWER_SCHEMA,
                "github_token": None,
                "repo": None,
                "branch": "main",
                "raise_on_failure": True,
                "max_file_size": 1_000_000,
                "outputs_to_string": {
                    "source": "result",
                    "handler": "haystack_integrations.tools.github.utils.message_handler",
                },
                "inputs_from_state": {"repo_state": "repo"},
                "outputs_to_state": {
                    "file_content": {
                        "source": "content",
                        "handler": "haystack_integrations.tools.github.utils.message_handler",
                    },
                },
            },
        }

        tool = GitHubRepoViewerTool.from_dict(tool_dict)
        assert tool.outputs_to_string["source"] == "result"
        assert tool.outputs_to_string["handler"] == message_handler
        assert tool.inputs_from_state == {"repo_state": "repo"}
        assert tool.outputs_to_state["file_content"]["source"] == "content"
        assert tool.outputs_to_state["file_content"]["handler"] == message_handler
