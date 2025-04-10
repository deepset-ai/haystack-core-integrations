import re
from typing import Any, Dict

import requests
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


@component
class GithubPRCreator:
    """
    A Haystack component for creating pull requests from a fork back to the original repository.

    Uses the authenticated user's fork to create the PR and links it to an existing issue.

    ### Usage example
    ```python
    from haystack.components.actions import GithubPRCreator
    from haystack.utils import Secret

    pr_creator = GithubPRCreator(
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

    def __init__(
            self,
            github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
            raise_on_failure: bool = True
    ):
        """
        Initialize the component.

        :param github_token: GitHub personal access token for authentication (from the fork owner)
        :param raise_on_failure: If True, raises exceptions on API errors
        """
        if not isinstance(github_token, Secret):
            raise TypeError("github_token must be a Secret")

        self.github_token = github_token
        self.raise_on_failure = raise_on_failure

    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for GitHub API requests with resolved token.

        :return: Dictionary of request headers
        """
        return {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {self.github_token.resolve_value()}",
            "User-Agent": "Haystack/GithubPRCreator"
        }

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
            raise ValueError("Invalid GitHub issue URL format")
        return match.group(1), match.group(2), match.group(3)

    def _get_authenticated_user(self) -> str:
        """Get the username of the authenticated user (fork owner)."""
        response = requests.get(
            "https://api.github.com/user",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()["login"]

    def _check_fork_exists(self, owner: str, repo: str, fork_owner: str) -> bool:
        """Check if the fork exists."""
        url = f"https://api.github.com/repos/{fork_owner}/{repo}"
        try:
            response = requests.get(url, headers=self._get_headers())
            response.raise_for_status()
            fork_data = response.json()
            return fork_data.get("fork", False)
        except requests.RequestException:
            return False

    @component.output_types(result=str)
    def run(
            self,
            issue_url: str,
            title: str,
            branch: str,
            base: str,
            body: str = "",
            draft: bool = False
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
            if not self._check_fork_exists(owner, repo_name, fork_owner):
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

            response = requests.post(url, headers=self._get_headers(), json=pr_data)
            response.raise_for_status()
            pr_number = response.json()["number"]

            return {"result": f"Pull request #{pr_number} created successfully and linked to issue #{issue_number}"}

        except (requests.RequestException, ValueError) as e:
            if self.raise_on_failure:
                raise
            return {"result": f"Error: {str(e)}"}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            github_token=self.github_token.to_dict() if self.github_token else None,
            raise_on_failure=self.raise_on_failure
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GithubPRCreator":
        """Deserialize the component from a dictionary."""
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(init_params, keys=["github_token"])
        return default_from_dict(cls, data)
