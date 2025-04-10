import re
from typing import Any, Dict, Optional

import requests
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import deserialize_secrets_inplace
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)


@component
class GithubIssueCommenter:
    """
    Posts comments to GitHub issues.

    The component takes a GitHub issue URL and comment text, then posts the comment
    to the specified issue using the GitHub API.

    ### Usage example
    ```python
    from haystack.components.writers import GithubIssueCommenter

    commenter = GithubIssueCommenter(github_token=Secret.from_env_var("GITHUB_TOKEN"))
    result = commenter.run(
        url="https://github.com/owner/repo/issues/123",
        comment="Thanks for reporting this issue! We'll look into it."
    )

    assert result["success"] is True
    ```
    """

    def __init__(
            self,
            github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
            raise_on_failure: bool = True,
            retry_attempts: int = 2,
    ):
        """
        Initialize the component.

        :param github_token: GitHub personal access token for API authentication as a Secret
        :param raise_on_failure: If True, raises exceptions on API errors
        :param retry_attempts: Number of retry attempts for failed requests
        """
        self.github_token = github_token
        self.raise_on_failure = raise_on_failure
        self.retry_attempts = retry_attempts

        # Set base headers during initialization
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Haystack/GithubIssueCommenter",
        }

    def _get_request_headers(self) -> dict:
        """
        Get headers with resolved token for the request.

        :return: Dictionary of headers including authorization if token is present
        """
        headers = self.headers.copy()
        if self.github_token is not None:
            headers["Authorization"] = f"Bearer {self.github_token.resolve_value()}"
        return headers

    def _parse_github_url(self, url: str) -> tuple[str, str, int]:
        """
        Parse GitHub URL into owner, repo and issue number.

        :param url: GitHub issue URL
        :return: Tuple of (owner, repo, issue_number)
        :raises ValueError: If URL format is invalid
        """
        pattern = r"https?://github\.com/([^/]+)/([^/]+)/issues/(\d+)"
        match = re.match(pattern, url)
        if not match:
            raise ValueError(f"Invalid GitHub issue URL format: {url}")

        owner, repo, issue_number = match.groups()
        return owner, repo, int(issue_number)

    def _post_comment(self, owner: str, repo: str, issue_number: int, comment: str) -> bool:
        """
        Post a comment to a GitHub issue.

        :param owner: Repository owner
        :param repo: Repository name
        :param issue_number: Issue number
        :param comment: Comment text to post
        :return: True if comment was posted successfully
        :raises requests.exceptions.RequestException: If the API request fails
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        data = {"body": comment}

        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(url, headers=self._get_request_headers(), json=data)
                response.raise_for_status()
                return True
            except requests.exceptions.RequestException as e:
                if attempt == self.retry_attempts - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")

        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            github_token=self.github_token.to_dict() if self.github_token else None,
            raise_on_failure=self.raise_on_failure,
            retry_attempts=self.retry_attempts,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GithubIssueCommenter":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(init_params, keys=["github_token"])
        return default_from_dict(cls, data)

    @component.output_types(success=bool)
    def run(self, url: str, comment: str) -> dict:
        """
        Post a comment to a GitHub issue.

        :param url: GitHub issue URL
        :param comment: Comment text to post
        :return: Dictionary containing success status
        """
        try:
            owner, repo, issue_number = self._parse_github_url(url)
            success = self._post_comment(owner, repo, issue_number, comment)
            return {"success": success}

        except Exception as e:
            if self.raise_on_failure:
                raise

            error_message = f"Error posting comment to GitHub issue {url}: {str(e)}"
            logger.warning(error_message)
            return {"success": False}
