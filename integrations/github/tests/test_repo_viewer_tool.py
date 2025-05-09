# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack_integrations.prompts.github.repo_viewer_prompt import REPO_VIEWER_PROMPT, REPO_VIEWER_SCHEMA
from haystack_integrations.tools.github.repo_viewer_tool import GitHubRepoViewerTool


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
