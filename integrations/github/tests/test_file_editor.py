# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
from haystack.utils import Secret
import requests

from haystack_integrations.components.connectors.github.file_editor import Command, GithubFileEditor


class TestGithubFileEditor:
    def test_init_default(self):
        editor = GithubFileEditor()
        assert editor.github_token is None
        assert editor.default_repo is None
        assert editor.default_branch == "main"
        assert editor.raise_on_failure is True

    def test_init_with_parameters(self):
        token = Secret.from_token("test_token")
        editor = GithubFileEditor(github_token=token, repo="owner/repo", branch="feature", raise_on_failure=False)
        assert editor.github_token == token
        assert editor.default_repo == "owner/repo"
        assert editor.default_branch == "feature"
        assert editor.raise_on_failure is False

        # Test with invalid token type
        with pytest.raises(TypeError):
            GithubFileEditor(github_token="not_a_secret")

    def test_to_dict(self):
        token = Secret.from_token("test_token")
        editor = GithubFileEditor(github_token=token, repo="owner/repo", branch="feature", raise_on_failure=False)

        result = editor.to_dict()

        assert result["github_token"]["type"] == "haystack.utils.Secret"
        assert result["repo"] == "owner/repo"
        assert result["branch"] == "feature"
        assert result["raise_on_failure"] is False

    def test_from_dict(self):
        data = {
            "github_token": {"type": "haystack.utils.Secret", "token": "test_token"},
            "repo": "owner/repo",
            "branch": "feature",
            "raise_on_failure": False,
        }

        editor = GithubFileEditor.from_dict(data)

        assert isinstance(editor.github_token, Secret)
        assert editor.github_token.resolve_value() == "test_token"
        assert editor.default_repo == "owner/repo"
        assert editor.default_branch == "feature"
        assert editor.raise_on_failure is False

    @patch("requests.get")
    @patch("requests.put")
    def test_run_edit(self, mock_put, mock_get):
        # Mock the file content response
        mock_get.return_value.json.return_value = {
            "content": "SGVsbG8gV29ybGQ=",  # Base64 encoded "Hello World"
            "sha": "abc123",
        }
        mock_get.return_value.raise_for_status.return_value = None

        # Mock the update response
        mock_put.return_value.raise_for_status.return_value = None

        token = Secret.from_token("test_token")
        editor = GithubFileEditor(github_token=token)

        result = editor.run(
            command=Command.EDIT,
            payload={"path": "test.txt", "original": "Hello", "replacement": "Hi", "message": "Update greeting"},
            repo="owner/repo",
            branch="main",
        )

        assert result["result"] == "Edit successful"

        # Verify the API calls
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/test.txt",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GithubFileEditor",
                "Authorization": "Bearer test_token",
            },
            params={"ref": "main"},
        )

        mock_put.assert_called_once()
        put_call = mock_put.call_args
        assert put_call[0][0] == "https://api.github.com/repos/owner/repo/contents/test.txt"
        assert put_call[1]["json"]["message"] == "Update greeting"
        assert put_call[1]["json"]["sha"] == "abc123"
        assert put_call[1]["json"]["branch"] == "main"

    @patch("requests.get")
    @patch("requests.patch")
    def test_run_undo(self, mock_patch, mock_get):
        # Mock the user check response
        mock_get.return_value.json.return_value = {"login": "testuser"}
        mock_get.return_value.raise_for_status.return_value = None

        # Mock the commits response
        mock_get.side_effect = [
            type(
                "Response", (), {"json": lambda: [{"author": {"login": "testuser"}}], "raise_for_status": lambda: None}
            ),
            type(
                "Response",
                (),
                {"json": lambda: [{"sha": "abc123"}, {"sha": "def456"}], "raise_for_status": lambda: None},
            ),
        ]

        # Mock the update response
        mock_patch.return_value.raise_for_status.return_value = None

        token = Secret.from_token("test_token")
        editor = GithubFileEditor(github_token=token)

        result = editor.run(
            command=Command.UNDO, payload={"message": "Undo last change"}, repo="owner/repo", branch="main"
        )

        assert result["result"] == "Successfully undid last change"

        # Verify the API calls
        assert mock_get.call_count == 3
        mock_patch.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/git/refs/heads/main",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GithubFileEditor",
                "Authorization": "Bearer test_token",
            },
            json={"sha": "def456", "force": True},
        )

    @patch("requests.put")
    def test_run_create(self, mock_put):
        # Mock the create response
        mock_put.return_value.raise_for_status.return_value = None

        token = Secret.from_token("test_token")
        editor = GithubFileEditor(github_token=token)

        result = editor.run(
            command=Command.CREATE,
            payload={"path": "new.txt", "content": "New file content", "message": "Create new file"},
            repo="owner/repo",
            branch="main",
        )

        assert result["result"] == "File created successfully"

        # Verify the API call
        mock_put.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/new.txt",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GithubFileEditor",
                "Authorization": "Bearer test_token",
            },
            json={
                "message": "Create new file",
                "content": "TmV3IGZpbGUgY29udGVudA==",  # Base64 encoded "New file content"
                "branch": "main",
            },
        )

    @patch("requests.get")
    @patch("requests.delete")
    def test_run_delete(self, mock_delete, mock_get):
        # Mock the file content response
        mock_get.return_value.json.return_value = {"sha": "abc123"}
        mock_get.return_value.raise_for_status.return_value = None

        # Mock the delete response
        mock_delete.return_value.raise_for_status.return_value = None

        token = Secret.from_token("test_token")
        editor = GithubFileEditor(github_token=token)

        result = editor.run(
            command=Command.DELETE,
            payload={"path": "test.txt", "message": "Delete file"},
            repo="owner/repo",
            branch="main",
        )

        assert result["result"] == "File deleted successfully"

        # Verify the API calls
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/test.txt",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GithubFileEditor",
                "Authorization": "Bearer test_token",
            },
            params={"ref": "main"},
        )

        mock_delete.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/test.txt",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GithubFileEditor",
                "Authorization": "Bearer test_token",
            },
            json={"message": "Delete file", "sha": "abc123", "branch": "main"},
        )

    @patch("requests.get")
    def test_run_error_handling(self, mock_get):
        # Mock an error response
        mock_get.side_effect = requests.RequestException("API Error")

        token = Secret.from_token("test_token")
        editor = GithubFileEditor(github_token=token, raise_on_failure=False)

        result = editor.run(
            command=Command.EDIT,
            payload={"path": "test.txt", "original": "Hello", "replacement": "Hi", "message": "Update greeting"},
            repo="owner/repo",
            branch="main",
        )

        assert "Error: API Error" in result["result"]

        # Test with raise_on_failure=True
        editor = GithubFileEditor(github_token=token, raise_on_failure=True)
        with pytest.raises(requests.RequestException):
            editor.run(
                command=Command.EDIT,
                payload={"path": "test.txt", "original": "Hello", "replacement": "Hi", "message": "Update greeting"},
                repo="owner/repo",
                branch="main",
            )
