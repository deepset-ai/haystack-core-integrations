# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import re
import time
from typing import Any, Dict, Optional

import requests
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


@component
class GitHubRepoForker:
    """
    Forks a GitHub repository from an issue URL.

    The component takes a GitHub issue URL, extracts the repository information,
    creates or syncs a fork of that repository, and optionally creates an issue-specific branch.

    ### Usage example
    ```python
    from haystack_integrations.components.connectors.github import GitHubRepoForker
    from haystack.utils import Secret

    # Using direct token with auto-sync and branch creation
    forker = GitHubRepoForker(
        github_token=Secret.from_env_var("GITHUB_TOKEN"),
        auto_sync=True,
        create_branch=True
    )

    result = forker.run(url="https://github.com/owner/repo/issues/123")
    print(result)
    # Will create or sync fork and create branch "fix-123"
    ```
    """

    def __init__(
        self,
        *,
        github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
        raise_on_failure: bool = True,
        wait_for_completion: bool = False,
        max_wait_seconds: int = 300,
        poll_interval: int = 2,
        auto_sync: bool = True,
        create_branch: bool = True,
    ):
        """
        Initialize the component.

        :param github_token: GitHub personal access token for API authentication
        :param raise_on_failure: If True, raises exceptions on API errors
        :param wait_for_completion: If True, waits until fork is fully created
        :param max_wait_seconds: Maximum time to wait for fork completion in seconds
        :param poll_interval: Time between status checks in seconds
        :param auto_sync: If True, syncs fork with original repository if it already exists
        :param create_branch: If True, creates a fix branch based on the issue number
        """
        error_message = "github_token must be a Secret"
        if not isinstance(github_token, Secret):
            raise TypeError(error_message)

        self.github_token = github_token
        self.raise_on_failure = raise_on_failure
        self.wait_for_completion = wait_for_completion
        self.max_wait_seconds = max_wait_seconds
        self.poll_interval = poll_interval
        self.auto_sync = auto_sync
        self.create_branch = create_branch

        self.base_headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Haystack/GitHubRepoForker",
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

    def _parse_github_url(self, url: str) -> tuple[str, str, str]:
        """
        Parse GitHub URL into owner, repo, and issue number.

        :param url: GitHub issue URL
        :return: Tuple of (owner, repo, issue_number)
        :raises ValueError: If URL format is invalid
        """
        pattern = r"https?://github\.com/([^/]+)/([^/]+)/issues/(\d+)"
        match = re.match(pattern, url)
        if not match:
            error_message = f"Invalid GitHub issue URL format: {url}"
            raise ValueError(error_message)

        owner, repo, issue_number = match.groups()
        return owner, repo, issue_number

    def _check_fork_status(self, fork_path: str) -> bool:
        """
        Check if a forked repository exists and is ready.

        :param fork_path: Repository path in owner/repo format
        :return: True if fork exists and is ready, False otherwise
        """
        url = f"https://api.github.com/repos/{fork_path}"
        try:
            response = requests.get(
                url,
                headers=self._get_request_headers(),
                timeout=10,
            )
            return response.status_code == 200  # noqa: PLR2004
        except requests.RequestException:
            return False

    def _get_authenticated_user(self) -> str:
        """
        Get the authenticated user's username.

        :return: Username of the authenticated user
        :raises requests.RequestException: If API call fails
        """
        url = "https://api.github.com/user"
        response = requests.get(url, headers=self._get_request_headers(), timeout=10)
        response.raise_for_status()
        return response.json()["login"]

    def _get_existing_repository(self, repo_name: str) -> Optional[str]:
        """
        Check if a repository with the given name already exists in the authenticated user's account.

        :param repo_name: Repository name to check
        :return: Full repository name if it exists, None otherwise
        """
        url = f"https://api.github.com/repos/{self._get_authenticated_user()}/{repo_name}"
        try:
            response = requests.get(
                url,
                headers=self._get_request_headers(),
                timeout=10,
            )
            if response.status_code == 200:  # noqa: PLR2004
                return repo_name
            return None
        except requests.RequestException as e:
            logger.warning(f"Failed to check repository existence: {e!s}")
            return None

    def _sync_fork(self, fork_path: str) -> None:
        """
        Sync a fork with its upstream repository.

        :param fork_path: Fork path in owner/repo format
        :raises requests.RequestException: If sync fails
        """
        url = f"https://api.github.com/repos/{fork_path}/merge-upstream"
        response = requests.post(
            url,
            headers=self._get_request_headers(),
            json={"branch": "main"},
            timeout=10,
        )
        response.raise_for_status()

    def _create_issue_branch(self, fork_path: str, issue_number: str) -> None:
        """
        Create a new branch for the issue.

        :param fork_path: Fork path in owner/repo format
        :param issue_number: Issue number to use in branch name
        :raises requests.RequestException: If branch creation fails
        """
        # First, get the default branch SHA
        url = f"https://api.github.com/repos/{fork_path}"
        response = requests.get(url, headers=self._get_request_headers(), timeout=10)
        response.raise_for_status()
        default_branch = response.json()["default_branch"]

        # Get the SHA of the default branch
        url = f"https://api.github.com/repos/{fork_path}/git/ref/heads/{default_branch}"
        response = requests.get(url, headers=self._get_request_headers(), timeout=10)
        response.raise_for_status()
        sha = response.json()["object"]["sha"]

        # Create the new branch
        branch_name = f"fix-{issue_number}"
        url = f"https://api.github.com/repos/{fork_path}/git/refs"
        response = requests.post(
            url,
            headers=self._get_request_headers(),
            json={"ref": f"refs/heads/{branch_name}", "sha": sha},
            timeout=10,
        )
        response.raise_for_status()

    def _create_fork(self, owner: str, repo: str) -> str:
        """
        Create a fork of the repository.

        :param owner: Original repository owner
        :param repo: Repository name
        :return: Fork path in owner/repo format
        :raises requests.RequestException: If fork creation fails
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/forks"
        response = requests.post(url, headers=self._get_request_headers(), timeout=10)
        response.raise_for_status()

        fork_data = response.json()
        return f"{fork_data['owner']['login']}/{fork_data['name']}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            github_token=self.github_token.to_dict() if self.github_token else None,
            raise_on_failure=self.raise_on_failure,
            wait_for_completion=self.wait_for_completion,
            max_wait_seconds=self.max_wait_seconds,
            poll_interval=self.poll_interval,
            auto_sync=self.auto_sync,
            create_branch=self.create_branch,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubRepoForker":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(init_params, keys=["github_token"])
        return default_from_dict(cls, data)

    @component.output_types(repo=str, issue_branch=str)
    def run(self, url: str) -> dict:
        """
        Process a GitHub issue URL and create or sync a fork of the repository.

        :param url: GitHub issue URL
        :return: Dictionary containing repository path in owner/repo format
        """
        try:
            # Extract repository information
            owner, repo, issue_number = self._parse_github_url(url)

            # Check if fork already exists
            user = self._get_authenticated_user()
            existing_fork = self._get_existing_repository(repo)

            if existing_fork and self.auto_sync:
                # If fork exists and auto_sync is enabled, sync with upstream
                fork_path = f"{user}/{repo}"
                logger.info("Fork already exists, syncing with upstream repository")
                self._sync_fork(fork_path)
            else:
                # Create new fork
                fork_path = self._create_fork(owner, repo)

            # Wait for fork completion if requested
            if self.wait_for_completion:
                start_time = time.time()

                while time.time() - start_time < self.max_wait_seconds:
                    if self._check_fork_status(fork_path):
                        logger.info("Fork creation completed successfully")
                        break
                    logger.debug("Waiting for fork creation to complete...")
                    time.sleep(self.poll_interval)
                else:
                    msg = f"Fork creation timed out after {self.max_wait_seconds} seconds"
                    if self.raise_on_failure:
                        raise TimeoutError(msg)
                    logger.warning(msg)

            # Create issue branch if enabled
            issue_branch = None
            if self.create_branch:
                issue_branch = f"fix-{issue_number}"
                logger.info(f"Creating branch for issue #{issue_number}")
                self._create_issue_branch(fork_path, issue_number)

            return {"repo": fork_path, "issue_branch": issue_branch}

        except Exception as e:
            if self.raise_on_failure:
                raise
            logger.warning("Error forking repository from {url}: {error}", url=url, error=str(e))
            return {"repo": "", "issue_branch": None}
