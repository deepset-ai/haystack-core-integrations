# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.connectors.github.repo_viewer import GithubRepositoryViewer


class TestGithubRepositoryViewer:
    def test_init_default(self):
        viewer = GithubRepositoryViewer()
        assert viewer.github_token is None
        assert viewer.raise_on_failure is True
        assert viewer.max_file_size == 1_000_000
        assert viewer.repo is None
        assert viewer.branch is None

    def test_init_with_parameters(self):
        token = Secret.from_token("test_token")
        viewer = GithubRepositoryViewer(
            github_token=token, raise_on_failure=False, max_file_size=500_000, repo="owner/repo", branch="main"
        )
        assert viewer.github_token == token
        assert viewer.raise_on_failure is False
        assert viewer.max_file_size == 500_000
        assert viewer.repo == "owner/repo"
        assert viewer.branch == "main"

        # Test with invalid token type
        with pytest.raises(TypeError):
            GithubRepositoryViewer(github_token="not_a_secret")

    def test_to_dict(self):
        token = Secret.from_token("test_token")
        viewer = GithubRepositoryViewer(github_token=token, raise_on_failure=False, max_file_size=500_000)

        result = viewer.to_dict()

        assert result["github_token"]["type"] == "haystack.utils.Secret"
        assert result["raise_on_failure"] is False
        assert result["max_file_size"] == 500_000

    def test_from_dict(self):
        data = {
            "github_token": {"type": "haystack.utils.Secret", "token": "test_token"},
            "raise_on_failure": False,
            "max_file_size": 500_000,
        }

        viewer = GithubRepositoryViewer.from_dict(data)

        assert isinstance(viewer.github_token, Secret)
        assert viewer.github_token.resolve_value() == "test_token"
        assert viewer.raise_on_failure is False
        assert viewer.max_file_size == 500_000

    @patch("requests.get")
    def test_run_file(self, mock_get):
        # Mock the file response
        mock_get.return_value.json.return_value = {
            "name": "README.md",
            "path": "README.md",
            "size": 100,
            "html_url": "https://github.com/owner/repo/blob/main/README.md",
            "content": "SGVsbG8gV29ybGQ=",  # Base64 encoded "Hello World"
            "encoding": "base64",
        }
        mock_get.return_value.raise_for_status.return_value = None

        token = Secret.from_token("test_token")
        viewer = GithubRepositoryViewer(github_token=token)

        result = viewer.run(repo="owner/repo", path="README.md", branch="main")

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Hello World"
        assert result["documents"][0].meta["type"] == "file_content"
        assert result["documents"][0].meta["path"] == "README.md"

        # Verify the API call
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/README.md?ref=main",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GithubRepositoryViewer",
                "Authorization": "Bearer test_token",
            },
        )

    @patch("requests.get")
    def test_run_directory(self, mock_get):
        # Mock the directory response
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

        token = Secret.from_token("test_token")
        viewer = GithubRepositoryViewer(github_token=token)

        result = viewer.run(repo="owner/repo", path="", branch="main")

        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "docs"
        assert result["documents"][0].meta["type"] == "dir"
        assert result["documents"][1].content == "README.md"
        assert result["documents"][1].meta["type"] == "file"

        # Verify the API call
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/?ref=main",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GithubRepositoryViewer",
                "Authorization": "Bearer test_token",
            },
        )

    @patch("requests.get")
    def test_run_error_handling(self, mock_get):
        # Mock an error response
        mock_get.side_effect = Exception("API Error")

        token = Secret.from_token("test_token")
        viewer = GithubRepositoryViewer(github_token=token, raise_on_failure=False)

        result = viewer.run(repo="owner/repo", path="README.md", branch="main")

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["type"] == "error"

        # Test with raise_on_failure=True
        viewer = GithubRepositoryViewer(github_token=token, raise_on_failure=True)
        with pytest.raises(Exception):
            viewer.run(repo="owner/repo", path="README.md", branch="main")

    def test_parse_repo(self):
        token = Secret.from_token("test_token")
        viewer = GithubRepositoryViewer(github_token=token)

        owner, repo = viewer._parse_repo("owner/repo")
        assert owner == "owner"
        assert repo == "repo"

        # Test with invalid format
        with pytest.raises(ValueError):
            viewer._parse_repo("invalid_format")
