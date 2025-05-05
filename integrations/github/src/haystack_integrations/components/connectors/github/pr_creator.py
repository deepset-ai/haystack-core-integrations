# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import re
from typing import Any, Dict, Optional

import requests
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


@component
class GitHubPRCreator:
    """
    A Haystack component for creating pull requests from a fork back to the original repository.

    Uses the authenticated user's fork to create the PR and links it to an existing issue.

    ### Usage example
    ```python
    from haystack_integrations.components.connectors.github import GitHubPRCreator
    from haystack.utils import Secret

    pr_creator = GitHubPRCreator(
        github_token=Secret.from_env_var("GITHUB_TOKEN")  # Token from the fork owner
    )

    # Create a PR from your fork
    result = pr_creator.run(
        issue_url="https://github.com/owner/repo/issues/123",
        title="Fix issue #123",
        body="This PR addresses issue #123",
        branch="feature-branch",     # The branch in your fork with the changes
        base="main"                  # The branch in the original repo to merge into
    )
    ```
    """

    def __init__(self, *, github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"), raise_on_failure: bool = True):
        """
        Initialize the component.

        :param github_token: GitHub personal access token for authentication (from the fork owner)
        :param raise_on_failure: If True, raises exceptions on API errors
        """
        if not isinstance(github_token, Secret):
            msg = "github_token must be a Secret"
            raise TypeError(msg)

        self.github_token = github_token
        self.raise_on_failure = raise_on_failure

        self.base_headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Haystack/GitHubPRCreator",
        }

    def _get_request_headers(self) -> dict:
        """
        Get headers with resolved token for the request.

        :return: Dictionary of headers including authorization if token is present
        """
        headers = self.base_headers.copy()
        if self.github_token is not None:
            headers["Authorization"] = f"Bearer {self.github_token.resolve_value()}"
        return headers

    def _parse_issue_url(self, issue_url: str) -> tuple[str, str, str]:
        """
        Parse owner, repo name, and issue number from GitHub issue URL.

        :param issue_url: Full GitHub issue URL
        :return: Tuple of (owner, repo_name, issue_number)
        :raises ValueError: If URL format is invalid
        """
        pattern = r"https://github\.com/([^/]+)/([^/]+)/issues/(\d+)"
        match = re.match(pattern, issue_url)
        if not match:
            msg = "Invalid GitHub issue URL format"
            raise ValueError(msg)
        return match.group(1), match.group(2), match.group(3)

    def _get_authenticated_user(self) -> str:
        """Get the username of the authenticated user (fork owner)."""
        response = requests.get("https://api.github.com/user", headers=self._get_request_headers(), timeout=10)
        response.raise_for_status()
        return response.json()["login"]

    def _check_fork_exists(self, repo: str, fork_owner: str) -> bool:
        """Check if the fork exists."""
        url = f"https://api.github.com/repos/{fork_owner}/{repo}"
        try:
            response = requests.get(url, headers=self._get_request_headers(), timeout=10)
            response.raise_for_status()
            fork_data = response.json()
            return fork_data.get("fork", False)
        except requests.RequestException:
            return False

    def _create_fork(self, owner: str, repo: str) -> Optional[str]:
        """Create a fork of the repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/forks"
        try:
            response = requests.post(url, headers=self._get_request_headers(), timeout=10)
            response.raise_for_status()
            fork_data = response.json()
            return fork_data["owner"]["login"]
        except requests.RequestException as e:
            if self.raise_on_failure:
                msg = f"Failed to create fork: {e!s}"
                raise RuntimeError(msg) from e
            return None

    def _create_branch(self, owner: str, repo: str, branch_name: str, base_branch: str) -> bool:
        """Create a new branch in the repository."""
        # Get the SHA of the base branch
        url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{base_branch}"
        try:
            response = requests.get(url, headers=self._get_request_headers(), timeout=10)
            response.raise_for_status()
            base_sha = response.json()["object"]["sha"]

            # Create the new branch
            url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
            data = {"ref": f"refs/heads/{branch_name}", "sha": base_sha}
            response = requests.post(url, headers=self._get_request_headers(), json=data, timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            if self.raise_on_failure:
                msg = f"Failed to create branch: {e!s}"
                raise RuntimeError(msg) from e
            return False

    def _create_commit(
        self,
        owner: str,
        repo: str,
        branch_name: str,
        file_path: str,
        content: str,
        message: str,
    ) -> bool:
        """Create a commit with the file changes."""
        # Get the current commit SHA
        url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch_name}"
        try:
            response = requests.get(url, headers=self._get_request_headers(), timeout=10)
            response.raise_for_status()
            current_sha = response.json()["object"]["sha"]

            # Create a blob with the file content
            url = f"https://api.github.com/repos/{owner}/{repo}/git/blobs"
            data: dict[str, Any] = {"content": content, "encoding": "base64"}
            response = requests.post(url, headers=self._get_request_headers(), json=data, timeout=10)
            response.raise_for_status()
            blob_sha = response.json()["sha"]

            # Create a tree with the new file
            url = f"https://api.github.com/repos/{owner}/{repo}/git/trees"
            data = {
                "base_tree": current_sha,
                "tree": [{"path": file_path, "mode": "100644", "type": "blob", "sha": blob_sha}],
            }
            response = requests.post(url, headers=self._get_request_headers(), json=data, timeout=10)
            response.raise_for_status()
            tree_sha = response.json()["sha"]

            # Create the commit
            url = f"https://api.github.com/repos/{owner}/{repo}/git/commits"
            data = {"message": message, "tree": tree_sha, "parents": [current_sha]}
            response = requests.post(url, headers=self._get_request_headers(), json=data, timeout=10)
            response.raise_for_status()
            commit_sha = response.json()["sha"]

            # Update the branch reference
            url = f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch_name}"
            data = {"sha": commit_sha}
            response = requests.patch(url, headers=self._get_request_headers(), json=data, timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            if self.raise_on_failure:
                msg = f"Failed to create commit: {e!s}"
                raise RuntimeError(msg) from e
            return False

    def _create_pull_request(
        self,
        owner: str,
        repo: str,
        branch_name: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> bool:
        """Create a pull request."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        data = {"title": title, "body": body, "head": branch_name, "base": base_branch}
        try:
            response = requests.post(url, headers=self._get_request_headers(), json=data, timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            if self.raise_on_failure:
                msg = f"Failed to create pull request: {e!s}"
                raise RuntimeError(msg) from e
            return False

    @component.output_types(result=str)
    def run(
        self, issue_url: str, title: str, branch: str, base: str, body: str = "", draft: bool = False
    ) -> Dict[str, str]:
        """
        Create a new pull request from your fork to the original repository, linked to the specified issue.

        :param issue_url: URL of the GitHub issue to link the PR to
        :param title: Title of the pull request
        :param branch: Name of the branch in your fork where changes are implemented
        :param base: Name of the branch in the original repo you want to merge into
        :param body: Additional content for the pull request description
        :param draft: Whether to create a draft pull request
        :return: Dictionary containing operation result
        """
        try:
            # Parse repository information from issue URL
            owner, repo_name, issue_number = self._parse_issue_url(issue_url)

            # Get the authenticated user (fork owner)
            fork_owner = self._get_authenticated_user()

            # Check if the fork exists
            if not self._check_fork_exists(repo_name, fork_owner):
                return {"result": f"Error: Fork not found at {fork_owner}/{repo_name}"}

            url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls"

            # For cross-repository PRs, head must be in the format username:branch
            head = f"{fork_owner}:{branch}"

            pr_data = {
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": draft,
                "maintainer_can_modify": True,  # Allow maintainers to modify the PR
            }

            response = requests.post(url, headers=self._get_request_headers(), json=pr_data, timeout=10)
            response.raise_for_status()
            pr_number = response.json()["number"]

            return {"result": f"Pull request #{pr_number} created successfully and linked to issue #{issue_number}"}

        except (requests.RequestException, ValueError) as e:
            if self.raise_on_failure:
                raise
            return {"result": f"Error: {e!s}"}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            github_token=self.github_token.to_dict() if self.github_token else None,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubPRCreator":
        """Deserialize the component from a dictionary."""
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(init_params, keys=["github_token"])
        return default_from_dict(cls, data)
