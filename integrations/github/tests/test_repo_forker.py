# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import requests
from haystack.utils import Secret

from haystack_integrations.components.connectors.github.repo_forker import GitHubRepoForker


class TestGitHubRepoForker:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        forker = GitHubRepoForker(github_token=Secret.from_env_var("GITHUB_TOKEN"))
        assert forker.github_token is not None
        assert forker.github_token.resolve_value() == "test-token"
        assert forker.raise_on_failure is True
        assert forker.wait_for_completion is False
        assert forker.max_wait_seconds == 300
        assert forker.poll_interval == 2
        assert forker.auto_sync is True
        assert forker.create_branch is True

    def test_init_with_parameters(self):
        token = Secret.from_token("test-token")
        forker = GitHubRepoForker(
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
            GitHubRepoForker(github_token="not_a_secret")

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")

        token = Secret.from_env_var("ENV_VAR")

        forker = GitHubRepoForker(
            github_token=token,
            raise_on_failure=False,
            wait_for_completion=True,
            max_wait_seconds=60,
            poll_interval=1,
            auto_sync=False,
            create_branch=False,
        )

        data = forker.to_dict()

        assert data == {
            "type": "haystack_integrations.components.connectors.github.repo_forker.GitHubRepoForker",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "raise_on_failure": False,
                "wait_for_completion": True,
                "max_wait_seconds": 60,
                "poll_interval": 1,
                "auto_sync": False,
                "create_branch": False,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")

        data = {
            "type": "haystack_integrations.components.connectors.github.repo_forker.GitHubRepoForker",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "raise_on_failure": False,
                "wait_for_completion": True,
                "max_wait_seconds": 60,
                "poll_interval": 1,
                "auto_sync": False,
                "create_branch": False,
            },
        }

        forker = GitHubRepoForker.from_dict(data)

        assert forker.github_token == Secret.from_env_var("ENV_VAR")
        assert forker.raise_on_failure is False
        assert forker.wait_for_completion is True
        assert forker.max_wait_seconds == 60
        assert forker.poll_interval == 1
        assert forker.auto_sync is False
        assert forker.create_branch is False

    @patch("requests.get")
    @patch("requests.post")
    def test_run_create_fork(self, mock_post, mock_get, monkeypatch):
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
            "https://api.github.com/user": create_mock_response({"login": "test_user"}),
            "https://api.github.com/repos/test_user/repo": create_mock_response(
                {}, status_code=404
            ),  # Fork doesn't exist
            "https://api.github.com/repos/test_user/repo/git/ref/heads/main": create_mock_response(
                {"object": {"sha": "abc123"}}
            ),
        }

        def get_side_effect(url, **_):
            if url == "https://api.github.com/repos/test_user/repo":
                if mock_get.call_count == 2:
                    return create_mock_response({}, status_code=404)  # Fork doesn't exist
                return create_mock_response({"default_branch": "main"})
            return get_responses.get(url, create_mock_response({"default_branch": "main"}))

        mock_get.side_effect = get_side_effect

        def post_side_effect(url, **_):
            if "forks" in url:
                return create_mock_response({"owner": {"login": "test_user"}, "name": "repo"})
            return create_mock_response({})

        mock_post.side_effect = post_side_effect

        forker = GitHubRepoForker(create_branch=True, auto_sync=False)

        result = forker.run(url="https://github.com/owner/repo/issues/123")

        assert result["repo"] == "test_user/repo"
        assert result["issue_branch"] == "fix-123"

        assert mock_get.call_count == 5  # user (2x), check fork status, get default branch, get SHA

        get_calls = [call[0][0] for call in mock_get.call_args_list]
        assert get_calls.count("https://api.github.com/user") == 2  # get user, check fork
        assert get_calls.count("https://api.github.com/repos/test_user/repo") == 2  # check status, get default branch
        assert "https://api.github.com/repos/test_user/repo/git/ref/heads/main" in get_calls

        post_calls = [call[0][0] for call in mock_post.call_args_list]
        assert "https://api.github.com/repos/owner/repo/forks" in post_calls
        assert "https://api.github.com/repos/test_user/repo/git/refs" in post_calls
        assert mock_post.call_count == 2  # One for fork creation, one for branch creation

    @patch("requests.get")
    @patch("requests.post")
    def test_run_sync_existing_fork(self, mock_post, mock_get, monkeypatch):
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
            "https://api.github.com/user": create_mock_response({"login": "test_user"}),
            "https://api.github.com/repos/test_user/repo": create_mock_response(
                {"name": "repo", "default_branch": "main"}
            ),
            "https://api.github.com/repos/test_user/repo/git/ref/heads/main": create_mock_response(
                {"object": {"sha": "abc123"}}
            ),
        }

        def get_side_effect(url, **_):
            return get_responses.get(url, create_mock_response({"default_branch": "main"}))

        mock_get.side_effect = get_side_effect

        def post_side_effect(url, **_):
            if "merge-upstream" in url:
                return create_mock_response({})
            return create_mock_response({})

        mock_post.side_effect = post_side_effect

        forker = GitHubRepoForker(create_branch=True, auto_sync=True)

        result = forker.run(url="https://github.com/owner/repo/issues/123")

        assert result["repo"] == "test_user/repo"
        assert result["issue_branch"] == "fix-123"

        assert mock_get.call_count == 5  # user, check fork, check fork status, get default branch, get SHA

        get_calls = [call[0][0] for call in mock_get.call_args_list]
        assert "https://api.github.com/user" in get_calls
        assert "https://api.github.com/repos/test_user/repo" in get_calls
        assert "https://api.github.com/repos/test_user/repo/git/ref/heads/main" in get_calls

        post_calls = [call[0][0] for call in mock_post.call_args_list]
        assert "https://api.github.com/repos/test_user/repo/merge-upstream" in post_calls
        assert "https://api.github.com/repos/test_user/repo/git/refs" in post_calls
        assert mock_post.call_count == 2  # One for sync, one for branch creation

    @patch("requests.get")
    @patch("requests.post")
    def test_run_error_handling(self, _, mock_get, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        mock_get.side_effect = requests.RequestException("API Error")

        forker = GitHubRepoForker(raise_on_failure=False)

        result = forker.run(url="https://github.com/owner/repo/issues/123")

        assert result["repo"] == ""
        assert result["issue_branch"] is None

        forker = GitHubRepoForker(raise_on_failure=True)
        with pytest.raises(requests.RequestException):
            forker.run(url="https://github.com/owner/repo/issues/123")

    def test_parse_github_url(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        forker = GitHubRepoForker()

        owner, repo, issue_number = forker._parse_github_url("https://github.com/owner/repo/issues/123")
        assert owner == "owner"
        assert repo == "repo"
        assert issue_number == "123"

        with pytest.raises(ValueError):
            forker._parse_github_url("https://github.com/invalid/url")

    def test_get_request_headers_with_empty_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "")

        forker = GitHubRepoForker()

        headers = forker._get_request_headers()

        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github.v3+json"
        assert headers["User-Agent"] == "Haystack/GitHubRepoForker"
