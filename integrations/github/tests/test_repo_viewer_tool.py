# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack_integrations.prompts.github.repo_viewer_prompt import REPO_VIEWER_PROMPT, REPO_VIEWER_SCHEMA
from haystack_integrations.tools.github.repo_viewer_tool import GitHubRepoViewerTool


def custom_handler(value):
    """A test handler function for serialization tests."""
    return f"Processed: {value}"


# Make the handler available at module level
__all__ = ["custom_handler"]


class TestGitHubRepoViewerTool:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubRepoViewerTool()
        assert tool.name == "repo_viewer"
        assert tool.description == REPO_VIEWER_PROMPT
        assert tool.parameters == REPO_VIEWER_SCHEMA
        assert tool.max_file_size == 1_000_000

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool_dict = {
            "type": "haystack_integrations.tools.github.repo_viewer_tool.GitHubRepoViewerTool",
            "init_parameters": {
                "name": "repo_viewer",
                "description": REPO_VIEWER_PROMPT,
                "parameters": REPO_VIEWER_SCHEMA,
                "github_token": None,
                "repo": None,
                "branch": "main",
                "raise_on_failure": True,
                "max_file_size": 1_000_000,
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

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubRepoViewerTool()
        tool_dict = tool.to_dict()
        assert tool_dict["type"] == "haystack_integrations.tools.github.repo_viewer_tool.GitHubRepoViewerTool"
        assert tool_dict["init_parameters"]["name"] == "repo_viewer"
        assert tool_dict["init_parameters"]["description"] == REPO_VIEWER_PROMPT
        assert tool_dict["init_parameters"]["parameters"] == REPO_VIEWER_SCHEMA
        assert tool_dict["init_parameters"]["github_token"] is None
        assert tool_dict["init_parameters"]["repo"] is None
        assert tool_dict["init_parameters"]["branch"] == "main"
        assert tool_dict["init_parameters"]["raise_on_failure"]
        assert tool_dict["init_parameters"]["max_file_size"] == 1_000_000

    def test_to_dict_with_extra_params(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        tool = GitHubRepoViewerTool(
            outputs_to_string={"source": "result", "handler": custom_handler},
            inputs_from_state={"repo_state": "repo"},
            outputs_to_state={"file_content": {"source": "content", "handler": custom_handler}},
        )

        tool_dict = tool.to_dict()
        assert tool_dict["init_parameters"]["outputs_to_string"] == {
            "source": "result",
            "handler": "tests.test_repo_viewer_tool.custom_handler",
        }
        assert tool_dict["init_parameters"]["inputs_from_state"] == {"repo_state": "repo"}
        assert tool_dict["init_parameters"]["outputs_to_state"] == {
            "file_content": {"source": "content", "handler": "tests.test_repo_viewer_tool.custom_handler"}
        }

    def test_from_dict_with_extra_params(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        tool_dict = {
            "type": "haystack_integrations.tools.github.repo_viewer_tool.GitHubRepoViewerTool",
            "init_parameters": {
                "name": "repo_viewer",
                "description": REPO_VIEWER_PROMPT,
                "parameters": REPO_VIEWER_SCHEMA,
                "github_token": None,
                "repo": None,
                "branch": "main",
                "raise_on_failure": True,
                "max_file_size": 1_000_000,
                "outputs_to_string": {"source": "result", "handler": "tests.test_repo_viewer_tool.custom_handler"},
                "inputs_from_state": {"repo_state": "repo"},
                "outputs_to_state": {
                    "file_content": {"source": "content", "handler": "tests.test_repo_viewer_tool.custom_handler"}
                },
            },
        }

        tool = GitHubRepoViewerTool.from_dict(tool_dict)
        assert tool.outputs_to_string["source"] == "result"
        assert tool.outputs_to_string["handler"] == custom_handler
        assert tool.inputs_from_state == {"repo_state": "repo"}
        assert tool.outputs_to_state["file_content"]["source"] == "content"
        assert tool.outputs_to_state["file_content"]["handler"] == custom_handler
