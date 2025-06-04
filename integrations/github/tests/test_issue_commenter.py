# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import requests
from haystack.utils import Secret

from haystack_integrations.components.connectors.github.issue_commenter import GitHubIssueCommenter


class TestGitHubIssueCommenter:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        commenter = GitHubIssueCommenter()
        assert commenter.github_token is not None
        assert commenter.github_token.resolve_value() == "test-token"
        assert commenter.raise_on_failure is True
        assert commenter.retry_attempts == 2

    def test_init_with_parameters(self):
        token = Secret.from_token("test-token")
        commenter = GitHubIssueCommenter(github_token=token, raise_on_failure=False, retry_attempts=3)
        assert commenter.github_token == token
        assert commenter.raise_on_failure is False
        assert commenter.retry_attempts == 3

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")

        token = Secret.from_env_var("ENV_VAR")

        commenter = GitHubIssueCommenter(github_token=token, raise_on_failure=False, retry_attempts=3)

        data = commenter.to_dict()

        assert data == {
            "type": "haystack_integrations.components.connectors.github.issue_commenter.GitHubIssueCommenter",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "raise_on_failure": False,
                "retry_attempts": 3,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")

        data = {
            "type": "haystack_integrations.components.connectors.github.issue_commenter.GitHubIssueCommenter",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "raise_on_failure": False,
                "retry_attempts": 3,
            },
        }

        commenter = GitHubIssueCommenter.from_dict(data)

        assert commenter.github_token == Secret.from_env_var("ENV_VAR")
        assert commenter.raise_on_failure is False
        assert commenter.retry_attempts == 3

    @patch("requests.post")
    def test_run(self, mock_post, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_post.return_value.raise_for_status.return_value = None

        commenter = GitHubIssueCommenter()

        result = commenter.run(url="https://github.com/owner/repo/issues/123", comment="Test comment")

        assert result["success"] is True

        mock_post.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/issues/123/comments",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GitHubIssueCommenter",
                "Authorization": "Bearer test-token",
            },
            json={"body": "Test comment"},
            timeout=10,
        )

    @patch("requests.post")
    def test_run_error_handling(self, mock_post, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_post.side_effect = requests.RequestException("API Error")

        commenter = GitHubIssueCommenter(raise_on_failure=False)

        result = commenter.run(url="https://github.com/owner/repo/issues/123", comment="Test comment")

        assert result["success"] is False

        commenter = GitHubIssueCommenter(raise_on_failure=True)
        with pytest.raises(requests.RequestException):
            commenter.run(url="https://github.com/owner/repo/issues/123", comment="Test comment")

    def test_parse_github_url(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        commenter = GitHubIssueCommenter()

        owner, repo, issue_number = commenter._parse_github_url("https://github.com/owner/repo/issues/123")
        assert owner == "owner"
        assert repo == "repo"
        assert issue_number == 123

        with pytest.raises(ValueError):
            commenter._parse_github_url("https://github.com/invalid/url")
