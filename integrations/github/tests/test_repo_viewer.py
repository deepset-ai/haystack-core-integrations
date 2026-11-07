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
            "https://api.github.com/repos/owner/repo/contents/README.md",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GitHubRepoViewer",
            },
            params={"ref": "main"},
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
            "https://api.github.com/repos/owner/repo/contents/",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GitHubRepoViewer",
            },
            params={"ref": "main"},
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

    @pytest.mark.parametrize("repo", ["../evil", "owner/..", "a/b/c", "."])
    @patch("requests.get")
    def test_run_rejects_invalid_repo(self, mock_get, repo, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        viewer = GitHubRepoViewer()
        with pytest.raises(ValueError):
            viewer.run(repo=repo, path="README.md")

        mock_get.assert_not_called()

    @pytest.mark.parametrize("path", ["../../other/repo/contents/x", "docs/../../../other/repo", "./x", "a//b"])
    @patch("requests.get")
    def test_run_rejects_invalid_path(self, mock_get, path, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        viewer = GitHubRepoViewer(raise_on_failure=False)
        result = viewer.run(repo="owner/repo", path=path)
        assert result["documents"][0].meta["type"] == "error"

        viewer = GitHubRepoViewer(raise_on_failure=True)
        with pytest.raises(ValueError, match="Invalid path"):
            viewer.run(repo="owner/repo", path=path)

        mock_get.assert_not_called()

    @pytest.mark.parametrize(
        "path,branch,expected_url",
        [
            # Same URLs as before ref was moved to `params`
            ("", "main", "https://api.github.com/repos/owner/repo/contents/?ref=main"),
            ("docs/", "main", "https://api.github.com/repos/owner/repo/contents/docs?ref=main"),
            (
                "docs/a b#1?.md",
                "feature/#1",
                "https://api.github.com/repos/owner/repo/contents/docs/a%20b%231%3F.md?ref=feature%2F%231",
            ),
        ],
    )
    @patch("requests.get")
    def test_run_sends_encoded_path_and_ref(self, mock_get, path, branch, expected_url, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        mock_get.return_value.json.return_value = []

        viewer = GitHubRepoViewer()
        viewer.run(repo="owner/repo", path=path, branch=branch)

        url = mock_get.call_args.args[0]
        params = mock_get.call_args.kwargs["params"]
        assert params == {"ref": branch}
        assert requests.Request("GET", url, params=params).prepare().url == expected_url

    def test_get_request_headers_with_empty_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "")

        token = Secret.from_env_var("GITHUB_TOKEN")
        viewer = GitHubRepoViewer(github_token=token)

        headers = viewer._get_request_headers()

        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["User-Agent"] == "Haystack/GitHubRepoViewer"

    def test_run_no_repo_raises(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        viewer = GitHubRepoViewer()
        with pytest.raises(ValueError, match="Repository not provided"):
            viewer.run(path="README.md")

    @pytest.mark.parametrize(
        "file_payload,error_substring",
        [
            (
                {
                    "name": "big.bin",
                    "path": "big.bin",
                    "size": 2_000_000,
                    "html_url": "https://x",
                    "content": "",
                    "encoding": "base64",
                },
                "exceeds limit",
            ),
            (
                {
                    "name": "raw.txt",
                    "path": "raw.txt",
                    "size": 5,
                    "html_url": "https://x",
                    "content": "Hello",
                    "encoding": "utf-8",
                },
                None,
            ),
        ],
    )
    @patch("requests.get")
    def test_run_file_size_and_encoding(self, mock_get, file_payload, error_substring, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        mock_get.return_value.json.return_value = file_payload
        mock_get.return_value.raise_for_status.return_value = None

        viewer = GitHubRepoViewer(raise_on_failure=False)
        result = viewer.run(repo="owner/repo", path=file_payload["path"])
        doc = result["documents"][0]
        if error_substring is None:
            assert doc.content == "Hello"
            assert doc.meta["type"] == "file_content"
        else:
            assert doc.meta["type"] == "error"
            assert error_substring in doc.content
