# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.connectors.github.issue_viewer import GithubIssueViewer

class TestGithubIssueViewer:
    def test_init_default(self):
        viewer = GithubIssueViewer()
        assert viewer.github_token is None
        assert viewer.raise_on_failure is True
        assert viewer.retry_attempts == 2

    def test_init_with_parameters(self):
        token = Secret.from_token("test_token")
        viewer = GithubIssueViewer(
            github_token=token,
            raise_on_failure=False,
            retry_attempts=3
        )
        assert viewer.github_token == token
        assert viewer.raise_on_failure is False
        assert viewer.retry_attempts == 3


    def test_to_dict(self):
        token = Secret.from_token("test_token")
        viewer = GithubIssueViewer(
            github_token=token,
            raise_on_failure=False,
            retry_attempts=3
        )
        
        result = viewer.to_dict()
        
        assert result["type"] == "haystack_integrations.components.connectors.github.issue_viewer.GithubIssueViewer"
        assert result["init_parameters"]["github_token"]["type"] == "haystack.utils.Secret"
        assert result["init_parameters"]["raise_on_failure"] is False
        assert result["init_parameters"]["retry_attempts"] == 3


    def test_from_dict():
        data = {
            "type": "haystack_integrations.components.connectors.github.issue_viewer.GithubIssueViewer",
            "init_parameters": {
                "github_token": {
                    "type": "haystack.utils.Secret",
                    "token": "test_token"
                },
                "raise_on_failure": False,
                "retry_attempts": 3
            }
        }
        
        viewer = GithubIssueViewer.from_dict(data)
        
        assert isinstance(viewer.github_token, Secret)
        assert viewer.github_token.resolve_value() == "test_token"
        assert viewer.raise_on_failure is False
        assert viewer.retry_attempts == 3


    @patch("requests.get")
    def test_run(mock_get):
        """Test the run method."""
        # Mock the issue response
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
            "comments_url": "https://api.github.com/repos/owner/repo/issues/123/comments"
        }
        mock_get.return_value.raise_for_status.return_value = None
        
        # Mock the comments response
        mock_get.side_effect = [
            mock_get.return_value,  # First call for issue
            type('Response', (), {
                'json': lambda: [
                    {
                        "body": "Comment 1",
                        "created_at": "2023-01-01T01:00:00Z",
                        "updated_at": "2023-01-01T01:00:00Z",
                        "user": {"login": "commenter1"},
                        "html_url": "https://github.com/owner/repo/issues/123#issuecomment-1"
                    },
                    {
                        "body": "Comment 2",
                        "created_at": "2023-01-01T02:00:00Z",
                        "updated_at": "2023-01-01T02:00:00Z",
                        "user": {"login": "commenter2"},
                        "html_url": "https://github.com/owner/repo/issues/123#issuecomment-2"
                    }
                ],
                'raise_for_status': lambda: None
            })
        ]
        
        token = Secret.from_token("test_token")
        viewer = GithubIssueViewer(github_token=token)
        
        result = viewer.run(url="https://github.com/owner/repo/issues/123")
        
        assert len(result["documents"]) == 3  # 1 issue + 2 comments
        assert result["documents"][0].meta["type"] == "issue"
        assert result["documents"][1].meta["type"] == "comment"
        assert result["documents"][2].meta["type"] == "comment"
        
        # Verify the API calls
        assert mock_get.call_count == 2


    @patch("requests.get")
    def test_run_error_handling(mock_get):
        # Mock an error response
        mock_get.side_effect = Exception("API Error")
        
        token = Secret.from_token("test_token")
        viewer = GithubIssueViewer(github_token=token, raise_on_failure=False)
        
        result = viewer.run(url="https://github.com/owner/repo/issues/123")
        
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["type"] == "error"
        assert result["documents"][0].meta["error"] is True
        
        # Test with raise_on_failure=True
        viewer = GithubIssueViewer(github_token=token, raise_on_failure=True)
        with pytest.raises(Exception):
            viewer.run(url="https://github.com/owner/repo/issues/123")


    def test_parse_github_url(self):
        token = Secret.from_token("test_token")
        viewer = GithubIssueViewer(github_token=token)
        
        owner, repo, issue_number = viewer._parse_github_url("https://github.com/owner/repo/issues/123")
        assert owner == "owner"
        assert repo == "repo"
        assert issue_number == 123
        
        # Test with invalid URL
        with pytest.raises(ValueError):
            viewer._parse_github_url("https://github.com/invalid/url")
