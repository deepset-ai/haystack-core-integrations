# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import requests
from haystack.utils import Secret

from haystack_integrations.components.connectors.github.pr_creator import GitHubPRCreator


class TestGitHubPRCreator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        pr_creator = GitHubPRCreator()
        assert pr_creator.github_token is not None
        assert pr_creator.github_token.resolve_value() == "test-token"
        assert pr_creator.raise_on_failure is True

    def test_init_with_parameters(self):
        token = Secret.from_token("test-token")
        pr_creator = GitHubPRCreator(github_token=token, raise_on_failure=False)
        assert pr_creator.github_token == token
        assert pr_creator.raise_on_failure is False

        with pytest.raises(TypeError):
            GitHubPRCreator(github_token="not_a_secret")

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")

        token = Secret.from_env_var("ENV_VAR")

        pr_creator = GitHubPRCreator(github_token=token, raise_on_failure=False)

        data = pr_creator.to_dict()

        assert data == {
            "type": "haystack_integrations.components.connectors.github.pr_creator.GitHubPRCreator",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "raise_on_failure": False,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")

        data = {
            "type": "haystack_integrations.components.connectors.github.pr_creator.GitHubPRCreator",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "raise_on_failure": False,
            },
        }

        pr_creator = GitHubPRCreator.from_dict(data)

        assert pr_creator.github_token == Secret.from_env_var("ENV_VAR")
        assert pr_creator.raise_on_failure is False

    @patch("requests.get")
    @patch("requests.post")
    def test_run(self, mock_post, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_get.return_value.json.return_value = {"login": "test_user"}
        mock_get.return_value.raise_for_status.return_value = None

        mock_post.return_value.json.return_value = {"number": 123}
        mock_post.return_value.raise_for_status.return_value = None

        pr_creator = GitHubPRCreator()

        with patch.object(pr_creator, "_check_fork_exists", return_value=True):
            result = pr_creator.run(
                issue_url="https://github.com/owner/repo/issues/456",
                title="Test PR",
                branch="feature-branch",
                base="main",
                body="Test body",
                draft=False,
            )

            assert result["result"] == "Pull request #123 created successfully and linked to issue #456"

            mock_get.assert_called_once()
            mock_post.assert_called_once_with(
                "https://api.github.com/repos/owner/repo/pulls",
                headers={
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "Haystack/GitHubPRCreator",
                    "Authorization": "Bearer test-token",
                },
                json={
                    "title": "Test PR",
                    "body": "Test body",
                    "head": "test_user:feature-branch",
                    "base": "main",
                    "draft": False,
                    "maintainer_can_modify": True,
                },
                timeout=10,
            )

    @patch("requests.get")
    @patch("requests.post")
    def test_run_error_handling(self, _, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_get.side_effect = requests.RequestException("API Error")

        pr_creator = GitHubPRCreator(raise_on_failure=False)

        with patch.object(pr_creator, "_check_fork_exists", return_value=True):
            result = pr_creator.run(
                issue_url="https://github.com/owner/repo/issues/456",
                title="Test PR",
                branch="feature-branch",
                base="main",
            )

            assert "Error" in result["result"]

        pr_creator = GitHubPRCreator(raise_on_failure=True)
        with pytest.raises(requests.RequestException):
            pr_creator.run(
                issue_url="https://github.com/owner/repo/issues/456",
                title="Test PR",
                branch="feature-branch",
                base="main",
            )

    def test_get_request_headers_with_empty_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "")

        pr_creator = GitHubPRCreator()

        headers = pr_creator._get_request_headers()

        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["User-Agent"] == "Haystack/GitHubPRCreator"
