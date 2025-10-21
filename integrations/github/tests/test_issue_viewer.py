# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import requests
from haystack.utils import Secret

from haystack_integrations.components.connectors.github.issue_viewer import GitHubIssueViewer


class TestGitHubIssueViewer:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        viewer = GitHubIssueViewer()
        assert viewer.github_token is None
        assert viewer.raise_on_failure is True
        assert viewer.retry_attempts == 2

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        token = Secret.from_env_var("GITHUB_TOKEN")
        viewer = GitHubIssueViewer(github_token=token, raise_on_failure=False, retry_attempts=3)
        assert viewer.github_token == token
        assert viewer.raise_on_failure is False
        assert viewer.retry_attempts == 3

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")

        token = Secret.from_env_var("ENV_VAR")

        viewer = GitHubIssueViewer(github_token=token, raise_on_failure=False, retry_attempts=3)

        data = viewer.to_dict()

        assert data == {
            "type": "haystack_integrations.components.connectors.github.issue_viewer.GitHubIssueViewer",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "raise_on_failure": False,
                "retry_attempts": 3,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")

        data = {
            "type": "haystack_integrations.components.connectors.github.issue_viewer.GitHubIssueViewer",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "raise_on_failure": False,
                "retry_attempts": 3,
            },
        }

        viewer = GitHubIssueViewer.from_dict(data)

        assert viewer.github_token == Secret.from_env_var("ENV_VAR")
        assert viewer.raise_on_failure is False
        assert viewer.retry_attempts == 3

    @patch("requests.get")
    def test_run(self, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_get.return_value.json.return_value = {
            "body": "Issue body",
            "title": "Issue title",
            "number": 123,
            "state": "open",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-02T00:00:00Z",
            "user": {"login": "test_user"},
            "html_url": "https://github.com/owner/repo/issues/123",
            "comments": 2,
            "comments_url": "https://api.github.com/repos/owner/repo/issues/123/comments",
        }
        mock_get.return_value.raise_for_status.return_value = None

        mock_get.side_effect = [
            mock_get.return_value,  # First call for issue
            type(
                "Response",
                (),
                {
                    "json": lambda: [
                        {
                            "body": "Comment 1",
                            "created_at": "2023-01-01T01:00:00Z",
                            "updated_at": "2023-01-01T01:00:00Z",
                            "user": {"login": "commenter1"},
                            "html_url": "https://github.com/owner/repo/issues/123#issuecomment-1",
                        },
                        {
                            "body": "Comment 2",
                            "created_at": "2023-01-01T02:00:00Z",
                            "updated_at": "2023-01-01T02:00:00Z",
                            "user": {"login": "commenter2"},
                            "html_url": "https://github.com/owner/repo/issues/123#issuecomment-2",
                        },
                    ],
                    "raise_for_status": lambda: None,
                },
            ),
        ]

        viewer = GitHubIssueViewer()

        result = viewer.run(url="https://github.com/owner/repo/issues/123")

        assert len(result["documents"]) == 3  # 1 issue + 2 comments
        assert result["documents"][0].meta["type"] == "issue"
        assert result["documents"][1].meta["type"] == "comment"
        assert result["documents"][2].meta["type"] == "comment"

        assert mock_get.call_count == 2

    @patch("requests.get")
    def test_run_error_handling(self, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_get.side_effect = requests.RequestException("API Error")

        viewer = GitHubIssueViewer(raise_on_failure=False)

        result = viewer.run(url="https://github.com/owner/repo/issues/123")

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["type"] == "error"
        assert result["documents"][0].meta["error"] is True

        viewer = GitHubIssueViewer(raise_on_failure=True)
        with pytest.raises(requests.RequestException):
            viewer.run(url="https://github.com/owner/repo/issues/123")

    def test_parse_github_url(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        viewer = GitHubIssueViewer()

        owner, repo, issue_number = viewer._parse_github_url("https://github.com/owner/repo/issues/123")
        assert owner == "owner"
        assert repo == "repo"
        assert issue_number == 123

        with pytest.raises(ValueError):
            viewer._parse_github_url("https://github.com/invalid/url")

    def test_get_request_headers_with_valid_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token-value")

        token = Secret.from_env_var("GITHUB_TOKEN")
        viewer = GitHubIssueViewer(github_token=token)

        headers = viewer._get_request_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-token-value"
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["User-Agent"] == "Haystack/GitHubIssueViewer"

    def test_get_request_headers_without_token(self):
        viewer = GitHubIssueViewer(github_token=None)

        headers = viewer._get_request_headers()

        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["User-Agent"] == "Haystack/GitHubIssueViewer"

    def test_get_request_headers_with_empty_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "")

        token = Secret.from_env_var("GITHUB_TOKEN")
        viewer = GitHubIssueViewer(github_token=token)

        headers = viewer._get_request_headers()

        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["User-Agent"] == "Haystack/GitHubIssueViewer"
