# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
from haystack.utils import Secret

from haystack_integrations.components.connectors.github.issue_commenter import GithubIssueCommenter


class TestGithubIssueCommenter:
    def test_init_default(self):
        commenter = GithubIssueCommenter()
        assert commenter.github_token is not None
        assert commenter.raise_on_failure is True
        assert commenter.retry_attempts == 2

    def test_init_with_parameters(self):
        token = Secret.from_token("test_token")
        commenter = GithubIssueCommenter(github_token=token, raise_on_failure=False, retry_attempts=3)
        assert commenter.github_token == token
        assert commenter.raise_on_failure is False
        assert commenter.retry_attempts == 3

    def test_to_dict(self):
        token = Secret.from_token("test_token")
        commenter = GithubIssueCommenter(github_token=token, raise_on_failure=False, retry_attempts=3)

        result = commenter.to_dict()

        assert (
            result["type"] == "haystack_integrations.components.connectors.github.issue_commenter.GithubIssueCommenter"
        )
        assert result["init_parameters"]["github_token"]["type"] == "haystack.utils.Secret"
        assert result["init_parameters"]["raise_on_failure"] is False
        assert result["init_parameters"]["retry_attempts"] == 3

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.connectors.github.issue_commenter.GithubIssueCommenter",
            "init_parameters": {
                "github_token": {"type": "haystack.utils.Secret", "token": "test_token"},
                "raise_on_failure": False,
                "retry_attempts": 3,
            },
        }

        commenter = GithubIssueCommenter.from_dict(data)

        assert isinstance(commenter.github_token, Secret)
        assert commenter.github_token.resolve_value() == "test_token"
        assert commenter.raise_on_failure is False
        assert commenter.retry_attempts == 3

    @patch("requests.post")
    def test_run(self, mock_post):
        """Test the run method."""
        # Mock the successful response
        mock_post.return_value.raise_for_status.return_value = None

        token = Secret.from_token("test_token")
        commenter = GithubIssueCommenter(github_token=token)

        result = commenter.run(url="https://github.com/owner/repo/issues/123", comment="Test comment")

        assert result["success"] is True

        # Verify the API call
        mock_post.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/issues/123/comments",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Haystack/GithubIssueCommenter",
                "Authorization": "Bearer test_token",
            },
            json={"body": "Test comment"},
        )

    @patch("requests.post")
    def test_run_error_handling(self, mock_post):
        # Mock an error response
        mock_post.side_effect = Exception("API Error")

        token = Secret.from_token("test_token")
        commenter = GithubIssueCommenter(github_token=token, raise_on_failure=False)

        result = commenter.run(url="https://github.com/owner/repo/issues/123", comment="Test comment")

        assert result["success"] is False

        # Test with raise_on_failure=True
        commenter = GithubIssueCommenter(github_token=token, raise_on_failure=True)
        with pytest.raises(Exception):
            commenter.run(url="https://github.com/owner/repo/issues/123", comment="Test comment")

    def test_parse_github_url(self):
        token = Secret.from_token("test_token")
        commenter = GithubIssueCommenter(github_token=token)

        owner, repo, issue_number = commenter._parse_github_url("https://github.com/owner/repo/issues/123")
        assert owner == "owner"
        assert repo == "repo"
        assert issue_number == 123

        # Test with invalid URL
        with pytest.raises(ValueError):
            commenter._parse_github_url("https://github.com/invalid/url")
