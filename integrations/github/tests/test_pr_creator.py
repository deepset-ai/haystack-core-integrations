# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
from haystack.utils import Secret

from haystack_integrations.components.connectors.github.pr_creator import GithubPRCreator


class TestGithubPRCreator:
    def test_init_default(self):
        pr_creator = GithubPRCreator()
        assert pr_creator.github_token is not None
        assert pr_creator.raise_on_failure is True

    def test_init_with_parameters(self):
        token = Secret.from_token("test_token")
        pr_creator = GithubPRCreator(github_token=token, raise_on_failure=False)
        assert pr_creator.github_token == token
        assert pr_creator.raise_on_failure is False

        # Test with invalid token type
        with pytest.raises(TypeError):
            GithubPRCreator(github_token="not_a_secret")

    def test_to_dict(self):
        token = Secret.from_token("test_token")
        pr_creator = GithubPRCreator(github_token=token, raise_on_failure=False)

        result = pr_creator.to_dict()

        assert result["type"] == "haystack_integrations.components.connectors.github.pr_creator.GithubPRCreator"
        assert result["init_parameters"]["github_token"]["type"] == "haystack.utils.Secret"
        assert result["init_parameters"]["raise_on_failure"] is False

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.connectors.github.pr_creator.GithubPRCreator",
            "init_parameters": {
                "github_token": {"type": "haystack.utils.Secret", "token": "test_token"},
                "raise_on_failure": False,
            },
        }

        pr_creator = GithubPRCreator.from_dict(data)

        assert isinstance(pr_creator.github_token, Secret)
        assert pr_creator.github_token.resolve_value() == "test_token"
        assert pr_creator.raise_on_failure is False

    @patch("requests.get")
    @patch("requests.post")
    def test_run(self, mock_post, mock_get):
        # Mock the authenticated user response
        mock_get.return_value.json.return_value = {"login": "test_user"}
        mock_get.return_value.raise_for_status.return_value = None

        # Mock the PR creation response
        mock_post.return_value.json.return_value = {"number": 123}
        mock_post.return_value.raise_for_status.return_value = None

        token = Secret.from_token("test_token")
        pr_creator = GithubPRCreator(github_token=token)

        result = pr_creator.run(
            issue_url="https://github.com/owner/repo/issues/456",
            title="Test PR",
            branch="feature-branch",
            base="main",
            body="Test body",
            draft=False,
        )

        assert result["result"] == "Pull request #123 created successfully and linked to issue #456"

        # Verify the API calls
        mock_get.assert_called_once()
        mock_post.assert_called_once()

    @patch("requests.get")
    @patch("requests.post")
    def test_run_error_handling(self, mock_post, mock_get):
        # Mock an error response
        mock_get.side_effect = Exception("API Error")

        token = Secret.from_token("test_token")
        pr_creator = GithubPRCreator(github_token=token, raise_on_failure=False)

        result = pr_creator.run(
            issue_url="https://github.com/owner/repo/issues/456", title="Test PR", branch="feature-branch", base="main"
        )

        assert "Error" in result["result"]

        # Test with raise_on_failure=True
        pr_creator = GithubPRCreator(github_token=token, raise_on_failure=True)
        with pytest.raises(Exception):
            pr_creator.run(
                issue_url="https://github.com/owner/repo/issues/456",
                title="Test PR",
                branch="feature-branch",
                base="main",
            )
