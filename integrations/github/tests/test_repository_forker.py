# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
from haystack.utils import Secret
import requests

from haystack_integrations.components.connectors.github.repository_forker import GithubRepoForker


class TestGithubRepoForker:
    def test_init_default(self):
        forker = GithubRepoForker()
        assert forker.github_token is not None
        assert forker.raise_on_failure is True
        assert forker.wait_for_completion is False
        assert forker.max_wait_seconds == 300
        assert forker.poll_interval == 2
        assert forker.auto_sync is True
        assert forker.create_branch is True

    def test_init_with_parameters(self):
        token = Secret.from_token("test_token")
        forker = GithubRepoForker(
            github_token=token,
            raise_on_failure=False,
            wait_for_completion=True,
            max_wait_seconds=60,
            poll_interval=1,
            auto_sync=False,
            create_branch=False,
        )
        assert forker.github_token == token
        assert forker.raise_on_failure is False
        assert forker.wait_for_completion is True
        assert forker.max_wait_seconds == 60
        assert forker.poll_interval == 1
        assert forker.auto_sync is False
        assert forker.create_branch is False

        # Test with invalid token type
        with pytest.raises(TypeError):
            GithubRepoForker(github_token="not_a_secret")

    def test_to_dict(self):
        token = Secret.from_token("test_token")
        forker = GithubRepoForker(
            github_token=token,
            raise_on_failure=False,
            wait_for_completion=True,
            max_wait_seconds=60,
            poll_interval=1,
            auto_sync=False,
            create_branch=False,
        )

        result = forker.to_dict()

        assert result["type"] == "haystack_integrations.components.connectors.github.repository_forker.GithubRepoForker"
        assert result["init_parameters"]["github_token"]["type"] == "haystack.utils.Secret"
        assert result["init_parameters"]["raise_on_failure"] is False
        assert result["init_parameters"]["wait_for_completion"] is True
        assert result["init_parameters"]["max_wait_seconds"] == 60
        assert result["init_parameters"]["poll_interval"] == 1
        assert result["init_parameters"]["auto_sync"] is False
        assert result["init_parameters"]["create_branch"] is False

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.connectors.github.repository_forker.GithubRepoForker",
            "init_parameters": {
                "github_token": {"type": "haystack.utils.Secret", "token": "test_token"},
                "raise_on_failure": False,
                "wait_for_completion": True,
                "max_wait_seconds": 60,
                "poll_interval": 1,
                "auto_sync": False,
                "create_branch": False,
            },
        }

        forker = GithubRepoForker.from_dict(data)

        assert isinstance(forker.github_token, Secret)
        assert forker.github_token.resolve_value() == "test_token"
        assert forker.raise_on_failure is False
        assert forker.wait_for_completion is True
        assert forker.max_wait_seconds == 60
        assert forker.poll_interval == 1
        assert forker.auto_sync is False
        assert forker.create_branch is False

    @patch("requests.get")
    @patch("requests.post")
    def test_run_create_fork(self, mock_post, mock_get):
        # Mock the authenticated user response
        mock_get.return_value.json.return_value = {"login": "test_user"}
        mock_get.return_value.raise_for_status.return_value = None

        # Mock the fork creation response
        mock_post.return_value.json.return_value = {"owner": {"login": "test_user"}, "name": "repo"}
        mock_post.return_value.raise_for_status.return_value = None

        token = Secret.from_token("test_token")
        forker = GithubRepoForker(github_token=token, create_branch=True, auto_sync=False)

        result = forker.run(url="https://github.com/owner/repo/issues/123")

        assert result["repo"] == "test_user/repo"
        assert result["issue_branch"] == "fix-123"

        # Verify the API calls
        mock_get.assert_called_once()
        assert mock_post.call_count == 2  # One for fork creation, one for branch creation

    @patch("requests.get")
    @patch("requests.post")
    def test_run_sync_existing_fork(self, mock_post, mock_get):
        # Mock the authenticated user response
        mock_get.side_effect = [
            type("Response", (), {"json": lambda: {"login": "test_user"}, "raise_for_status": lambda: None}),
            type(
                "Response", (), {"status_code": 200, "json": lambda: {"name": "repo"}, "raise_for_status": lambda: None}
            ),
        ]

        # Mock the sync response
        mock_post.return_value.raise_for_status.return_value = None

        token = Secret.from_token("test_token")
        forker = GithubRepoForker(github_token=token, create_branch=True, auto_sync=True)

        result = forker.run(url="https://github.com/owner/repo/issues/123")

        assert result["repo"] == "test_user/repo"
        assert result["issue_branch"] == "fix-123"

        # Verify the API calls
        assert mock_get.call_count == 2
        assert mock_post.call_count == 2  # One for sync, one for branch creation

    @patch("requests.get")
    @patch("requests.post")
    def test_run_error_handling(self, mock_post, mock_get):
        # Mock an error response
        mock_get.side_effect = requests.RequestException("API Error")

        token = Secret.from_token("test_token")
        forker = GithubRepoForker(github_token=token, raise_on_failure=False)

        result = forker.run(url="https://github.com/owner/repo/issues/123")

        assert result["repo"] == ""
        assert result["issue_branch"] is None

        # Test with raise_on_failure=True
        forker = GithubRepoForker(github_token=token, raise_on_failure=True)
        with pytest.raises(requests.RequestException):
            forker.run(url="https://github.com/owner/repo/issues/123")

    def test_parse_github_url(self):
        token = Secret.from_token("test_token")
        forker = GithubRepoForker(github_token=token)

        owner, repo, issue_number = forker._parse_github_url("https://github.com/owner/repo/issues/123")
        assert owner == "owner"
        assert repo == "repo"
        assert issue_number == "123"

        # Test with invalid URL
        with pytest.raises(ValueError):
            forker._parse_github_url("https://github.com/invalid/url")
