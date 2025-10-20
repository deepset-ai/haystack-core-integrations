# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import re
from typing import Any, Dict, List, Optional

import requests
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import deserialize_secrets_inplace
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)


@component
class GitHubIssueViewer:
    """
    Fetches and parses GitHub issues into Haystack documents.

    The component takes a GitHub issue URL and returns a list of documents where:
    - First document contains the main issue content
    - Subsequent documents contain the issue comments

    ### Usage example
    ```python
    from haystack_integrations.components.connectors.github import GitHubIssueViewer

    viewer = GitHubIssueViewer()
    docs = viewer.run(
        url="https://github.com/owner/repo/issues/123"
    )["documents"]

    print(docs)
    ```
    """

    def __init__(
        self,
        *,
        github_token: Optional[Secret] = None,
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

        # Only set the basic headers during initialization
        self.base_headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Haystack/GitHubIssueViewer",
        }

    def _get_request_headers(self) -> dict:
        """
        Get headers with resolved token for the request.

        :return: Dictionary of headers including authorization if token is present
        """
        headers = self.base_headers.copy()
        if self.github_token:
            token_value = self.github_token.resolve_value()
            if token_value:
                headers["Authorization"] = f"Bearer {token_value}"
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
            msg = f"Invalid GitHub issue URL format: {url}"
            raise ValueError(msg)

        owner, repo, issue_number = match.groups()
        return owner, repo, int(issue_number)

    def _fetch_issue(self, owner: str, repo: str, issue_number: int) -> Any:
        """
        Fetch issue data from GitHub API.

        :param owner: Repository owner
        :param repo: Repository name
        :param issue_number: Issue number
        :return: Issue data dictionary
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        response = requests.get(url, headers=self._get_request_headers(), timeout=10)
        response.raise_for_status()
        return response.json()

    def _fetch_comments(self, comments_url: str) -> Any:
        """
        Fetch issue comments from GitHub API.

        :param comments_url: URL for issue comments
        :return: List of comment dictionaries
        """
        response = requests.get(comments_url, headers=self._get_request_headers(), timeout=10)
        response.raise_for_status()
        return response.json()

    def _create_issue_document(self, issue_data: dict) -> Document:
        """
        Create a Document from issue data.

        :param issue_data: Issue data from GitHub API
        :return: Haystack Document
        """
        return Document(  # type: ignore
            content=issue_data["body"],
            meta={
                "type": "issue",
                "title": issue_data["title"],
                "number": issue_data["number"],
                "state": issue_data["state"],
                "created_at": issue_data["created_at"],
                "updated_at": issue_data["updated_at"],
                "author": issue_data["user"]["login"],
                "url": issue_data["html_url"],
            },
        )

    def _create_comment_document(self, comment_data: dict, issue_number: int) -> Document:
        """
        Create a Document from comment data.

        :param comment_data: Comment data from GitHub API
        :param issue_number: Parent issue number
        :return: Haystack Document
        """
        return Document(
            content=comment_data["body"],
            meta={
                "type": "comment",
                "issue_number": issue_number,
                "created_at": comment_data["created_at"],
                "updated_at": comment_data["updated_at"],
                "author": comment_data["user"]["login"],
                "url": comment_data["html_url"],
            },
        )

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
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubIssueViewer":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(init_params, keys=["github_token"])
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, url: str) -> dict:
        """
        Process a GitHub issue URL and return documents.

        :param url: GitHub issue URL
        :return: Dictionary containing list of documents
        """
        try:
            owner, repo, issue_number = self._parse_github_url(url)

            # Fetch issue data
            issue_data = self._fetch_issue(owner, repo, issue_number)
            documents = [self._create_issue_document(issue_data)]

            # Fetch and process comments if they exist
            if issue_data["comments"] > 0:
                comments = self._fetch_comments(issue_data["comments_url"])
                documents.extend(self._create_comment_document(comment, issue_number) for comment in comments)

            return {"documents": documents}

        except Exception as e:
            if self.raise_on_failure:
                raise

            error_message = f"Error processing GitHub issue {url}: {e!s}"
            logger.warning(error_message)
            error_doc = Document(
                content=error_message,
                meta={
                    "error": True,
                    "type": "error",
                    "url": url,
                },
            )
            return {"documents": [error_doc]}
