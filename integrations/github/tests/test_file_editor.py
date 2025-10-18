# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import requests
from haystack.utils import Secret

from haystack_integrations.components.connectors.github.file_editor import Command, GitHubFileEditor


class TestGitHubFileEditor:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        editor = GitHubFileEditor()
        assert editor.github_token is not None
        assert editor.github_token.resolve_value() == "test-token"
        assert editor.default_repo is None
        assert editor.default_branch == "main"
        assert editor.raise_on_failure is True

    def test_init_with_parameters(self):
        token = Secret.from_token("test-token")
        editor = GitHubFileEditor(github_token=token, repo="owner/repo", branch="feature", raise_on_failure=False)
        assert editor.github_token == token
        assert editor.default_repo == "owner/repo"
        assert editor.default_branch == "feature"
        assert editor.raise_on_failure is False

        with pytest.raises(TypeError):
            GitHubFileEditor(github_token="not_a_secret")

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")

        token = Secret.from_env_var("ENV_VAR")

        editor = GitHubFileEditor(github_token=token, repo="owner/repo", branch="feature", raise_on_failure=False)

        data = editor.to_dict()

        assert data == {
            "type": "haystack_integrations.components.connectors.github.file_editor.GitHubFileEditor",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "repo": "owner/repo",
                "branch": "feature",
                "raise_on_failure": False,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")
        data = {
            "type": "haystack_integrations.components.connectors.github.file_editor.GitHubFileEditor",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "repo": "owner/repo",
                "branch": "feature",
                "raise_on_failure": False,
            },
        }

        editor = GitHubFileEditor.from_dict(data)

        assert editor.github_token == Secret.from_env_var("ENV_VAR")
        assert editor.default_repo == "owner/repo"
        assert editor.default_branch == "feature"
        assert editor.raise_on_failure is False

    @patch("requests.get")
    @patch("requests.put")
    def test_run_edit(self, mock_put, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_get.return_value.json.return_value = {
            "content": "SGVsbG8gV29ybGQ=",  # Base64 encoded "Hello World"
            "sha": "abc123",
        }
        mock_get.return_value.raise_for_status.return_value = None
        mock_put.return_value.raise_for_status.return_value = None

        editor = GitHubFileEditor()

        result = editor.run(
            command=Command.EDIT,
            payload={"path": "test.txt", "original": "Hello", "replacement": "Hi", "message": "Update greeting"},
            repo="owner/repo",
            branch="main",
        )

        assert result["result"] == "Edit successful"

        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/test.txt",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GitHubFileEditor",
                "Authorization": "Bearer test-token",
            },
            params={"ref": "main"},
            timeout=10,
        )

        mock_put.assert_called_once()
        put_call = mock_put.call_args
        assert put_call[0][0] == "https://api.github.com/repos/owner/repo/contents/test.txt"
        assert put_call[1]["json"]["message"] == "Update greeting"
        assert put_call[1]["json"]["sha"] == "abc123"
        assert put_call[1]["json"]["branch"] == "main"

    @patch("requests.get")
    @patch("requests.patch")
    def test_run_undo(self, mock_patch, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        def create_mock_response(json_data, status_code=200):
            class MockResponse:
                def __init__(self, data, code):
                    self._data = data
                    self.status_code = code

                def json(self):
                    return self._data

                def raise_for_status(self):
                    if self.status_code >= 400:
                        error_message = f"HTTP {self.status_code}"
                        raise requests.RequestException(error_message)

            return MockResponse(json_data, status_code)

        get_responses = {
            "https://api.github.com/user": create_mock_response({"login": "testuser"}),
            "https://api.github.com/repos/owner/repo/commits": create_mock_response(
                [{"author": {"login": "testuser"}, "sha": "abc123"}, {"author": {"login": "testuser"}, "sha": "def456"}]
            ),
        }

        def get_side_effect(url, **_):
            return get_responses.get(url, create_mock_response({}))

        mock_get.side_effect = get_side_effect

        mock_patch.return_value.raise_for_status.return_value = None

        editor = GitHubFileEditor()

        result = editor.run(
            command=Command.UNDO, payload={"message": "Undo last change"}, repo="owner/repo", branch="main"
        )

        assert result["result"] == "Successfully undid last change"

        assert mock_get.call_count == 3  # One for commits, one for user info, one for last commit check
        mock_patch.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/git/refs/heads/main",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GitHubFileEditor",
                "Authorization": "Bearer test-token",
            },
            json={"sha": "def456", "force": True},
            timeout=10,
        )

    @patch("requests.put")
    def test_run_create(self, mock_put, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_put.return_value.raise_for_status.return_value = None

        editor = GitHubFileEditor()

        result = editor.run(
            command=Command.CREATE,
            payload={"path": "new.txt", "content": "New file content", "message": "Create new file"},
            repo="owner/repo",
            branch="main",
        )

        assert result["result"] == "File created successfully"

        mock_put.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/new.txt",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GitHubFileEditor",
                "Authorization": "Bearer test-token",
            },
            json={
                "message": "Create new file",
                "content": "TmV3IGZpbGUgY29udGVudA==",  # Base64 encoded "New file content"
                "branch": "main",
            },
            timeout=10,
        )

    @patch("requests.get")
    @patch("requests.delete")
    def test_run_delete(self, mock_delete, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_get.return_value.json.return_value = {
            "content": "SGVsbG8gV29ybGQ=",  # Base64 encoded "Hello World"
            "sha": "abc123",
        }
        mock_get.return_value.raise_for_status.return_value = None

        mock_delete.return_value.raise_for_status.return_value = None

        editor = GitHubFileEditor()

        result = editor.run(
            command=Command.DELETE,
            payload={"path": "test.txt", "message": "Delete file"},
            repo="owner/repo",
            branch="main",
        )

        assert result["result"] == "File deleted successfully"

        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/test.txt",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GitHubFileEditor",
                "Authorization": "Bearer test-token",
            },
            params={"ref": "main"},
            timeout=10,
        )

        mock_delete.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/test.txt",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GitHubFileEditor",
                "Authorization": "Bearer test-token",
            },
            json={"message": "Delete file", "sha": "abc123", "branch": "main"},
            timeout=10,
        )

    @patch("requests.get")
    def test_run_error_handling(self, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_get.side_effect = requests.RequestException("API Error")

        editor = GitHubFileEditor(raise_on_failure=False)

        result = editor.run(
            command=Command.EDIT,
            payload={"path": "test.txt", "original": "Hello", "replacement": "Hi", "message": "Update greeting"},
            repo="owner/repo",
            branch="main",
        )

        assert "Error: API Error" in result["result"]

        editor = GitHubFileEditor(raise_on_failure=True)
        with pytest.raises(requests.RequestException):
            editor.run(
                command=Command.EDIT,
                payload={"path": "test.txt", "original": "Hello", "replacement": "Hi", "message": "Update greeting"},
                repo="owner/repo",
                branch="main",
            )

    def test_get_request_headers_with_empty_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "")

        editor = GitHubFileEditor()

        headers = editor._get_request_headers()

        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["User-Agent"] == "Haystack/GitHubFileEditor"
