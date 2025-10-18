# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import requests
from haystack.utils import Secret

from haystack_integrations.components.connectors.github.repo_viewer import GitHubRepoViewer


class TestGitHubRepoViewer:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        viewer = GitHubRepoViewer()
        assert viewer.github_token is None
        assert viewer.raise_on_failure is True
        assert viewer.max_file_size == 1_000_000
        assert viewer.repo is None
        assert viewer.branch == "main"

    def test_init_with_parameters(self):
        token = Secret.from_token("test-token")
        viewer = GitHubRepoViewer(
            github_token=token, raise_on_failure=False, max_file_size=500_000, repo="owner/repo", branch="test-branch"
        )
        assert viewer.github_token == token
        assert viewer.raise_on_failure is False
        assert viewer.max_file_size == 500_000
        assert viewer.repo == "owner/repo"
        assert viewer.branch == "test-branch"

        with pytest.raises(TypeError):
            GitHubRepoViewer(github_token="not_a_secret")

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")

        token = Secret.from_env_var("ENV_VAR")

        viewer = GitHubRepoViewer(
            github_token=token, raise_on_failure=False, max_file_size=500_000, repo="owner/repo", branch="test-branch"
        )

        data = viewer.to_dict()

        assert data == {
            "type": "haystack_integrations.components.connectors.github.repo_viewer.GitHubRepoViewer",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "raise_on_failure": False,
                "max_file_size": 500_000,
                "repo": "owner/repo",
                "branch": "test-branch",
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")

        data = {
            "type": "haystack_integrations.components.connectors.github.repo_viewer.GitHubRepoViewer",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "raise_on_failure": False,
                "max_file_size": 500_000,
                "repo": "owner/repo",
                "branch": "test-branch",
            },
        }

        viewer = GitHubRepoViewer.from_dict(data)

        assert viewer.github_token == Secret.from_env_var("ENV_VAR")
        assert viewer.raise_on_failure is False
        assert viewer.max_file_size == 500_000
        assert viewer.repo == "owner/repo"
        assert viewer.branch == "test-branch"

    @patch("requests.get")
    def test_run_file(self, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_get.return_value.json.return_value = {
            "name": "README.md",
            "path": "README.md",
            "size": 100,
            "html_url": "https://github.com/owner/repo/blob/main/README.md",
            "content": "SGVsbG8gV29ybGQ=",  # Base64 encoded "Hello World"
            "encoding": "base64",
        }
        mock_get.return_value.raise_for_status.return_value = None

        viewer = GitHubRepoViewer()

        result = viewer.run(repo="owner/repo", path="README.md", branch="main")

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Hello World"
        assert result["documents"][0].meta["type"] == "file_content"
        assert result["documents"][0].meta["path"] == "README.md"

        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/README.md?ref=main",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GitHubRepoViewer",
            },
            timeout=10,
        )

    @patch("requests.get")
    def test_run_directory(self, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_get.return_value.json.return_value = [
            {"name": "docs", "path": "docs", "type": "dir", "html_url": "https://github.com/owner/repo/tree/main/docs"},
            {
                "name": "README.md",
                "path": "README.md",
                "type": "file",
                "size": 100,
                "html_url": "https://github.com/owner/repo/blob/main/README.md",
            },
        ]
        mock_get.return_value.raise_for_status.return_value = None

        viewer = GitHubRepoViewer()

        result = viewer.run(repo="owner/repo", path="", branch="main")

        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "docs"
        assert result["documents"][0].meta["type"] == "dir"
        assert result["documents"][1].content == "README.md"
        assert result["documents"][1].meta["type"] == "file"

        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/?ref=main",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GitHubRepoViewer",
            },
            timeout=10,
        )

    @patch("requests.get")
    def test_run_error_handling(self, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_get.side_effect = requests.RequestException("API Error")

        viewer = GitHubRepoViewer(raise_on_failure=False)

        result = viewer.run(repo="owner/repo", path="README.md", branch="main")

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["type"] == "error"

        viewer = GitHubRepoViewer(raise_on_failure=True)
        with pytest.raises(requests.RequestException):
            viewer.run(repo="owner/repo", path="README.md", branch="main")

    def test_parse_repo(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        viewer = GitHubRepoViewer()

        owner, repo = viewer._parse_repo("owner/repo")
        assert owner == "owner"
        assert repo == "repo"

        with pytest.raises(ValueError):
            viewer._parse_repo("invalid_format")

    def test_get_request_headers_with_empty_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "")

        token = Secret.from_env_var("GITHUB_TOKEN")
        viewer = GitHubRepoViewer(github_token=token)

        headers = viewer._get_request_headers()

        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["User-Agent"] == "Haystack/GitHubRepoViewer"
