import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from haystack import Document, component, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)


@dataclass
class GitHubItem:
    """Represents an item (file or directory) in a GitHub repository"""

    name: str
    type: str  # "file" or "dir"
    path: str
    size: int
    url: str
    content: Optional[str] = None


@component
class GithubRepositoryViewer:
    """
    Navigates and fetches content from GitHub repositories.

    For directories:
    - Returns a list of Documents, one for each item
    - Each Document's content is the item name
    - Full path and metadata in Document.meta

    For files:
    - Returns a single Document
    - Document's content is the file content
    - Full path and metadata in Document.meta

    For errors:
    - Returns a single Document
    - Document's content is the error message
    - Document's meta contains type="error"

    ### Usage example
    ```python
    from haystack.components.fetchers import GithubRepositoryViewer
    from haystack.utils import Secret

    # Using token directly
    viewer = GithubRepositoryViewer(github_token=Secret.from_token("your_token"))

    # Using environment variable
    viewer = GithubRepositoryViewer(github_token=Secret.from_env_var("GITHUB_TOKEN"))

    # List directory contents - returns multiple documents
    result = viewer.run(
        repo="owner/repository",
        path="docs/",
        ref="main"
    )

    # Get specific file - returns single document
    result = viewer.run(
        repo="owner/repository",
        path="README.md",
        ref="main"
    )
    ```
    """

    def __init__(
        self,
        github_token: Optional[Secret] = None,
        raise_on_failure: bool = True,
        max_file_size: int = 1_000_000,  # 1MB default limit
        repo: Optional[str] = None,
        branch: Optional[str] = None
    ):
        """
        Initialize the component.

        :param github_token: GitHub personal access token for API authentication
        :param raise_on_failure: If True, raises exceptions on API errors
        :param max_file_size: Maximum file size in bytes to fetch (default: 1MB)
        """
        if github_token is not None and not isinstance(github_token, Secret):
            raise TypeError("github_token must be a Secret")

        self.github_token = github_token
        self.raise_on_failure = raise_on_failure
        self.max_file_size = max_file_size
        self.repo = repo
        self.branch = branch

        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Haystack/GithubRepositoryViewer",
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return {
            "github_token": self.github_token.to_dict() if self.github_token else None,
            "raise_on_failure": self.raise_on_failure,
            "max_file_size": self.max_file_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GithubRepositoryViewer":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        init_params = data.copy()
        if init_params["github_token"]:
            init_params["github_token"] = Secret.from_dict(init_params["github_token"])
        return cls(**init_params)

    def _parse_repo(self, repo: str) -> tuple[str, str]:
        """Parse owner/repo string"""
        parts = repo.split("/")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid repository format. Expected 'owner/repo', got '{repo}'"
            )
        return parts[0], parts[1]

    def _normalize_path(self, path: str) -> str:
        """Normalize repository path"""
        return path.strip("/")

    def _fetch_contents(self, owner: str, repo: str, path: str, ref: str) -> Any:
        """Fetch repository contents from GitHub API"""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        if ref:
            url += f"?ref={ref}"

        headers = self.headers.copy()
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token.resolve_value()}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def _process_file_content(self, content: str, encoding: str) -> str:
        """Process file content based on encoding"""
        if encoding == "base64":
            return base64.b64decode(content).decode("utf-8")
        return content

    def _create_file_document(self, item: GitHubItem) -> Document:
        """Create a Document from a file"""
        return Document(
            content=item.content if item.content else item.name,
            meta={
                "path": item.path,
                "type": "file_content",
                "size": item.size,
                "url": item.url,
            },
        )

    def _create_directory_documents(self, items: List[GitHubItem]) -> List[Document]:
        """Create a list of Documents from directory contents"""
        return [
            Document(
                content=item.name,
                meta={
                    "path": item.path,
                    "type": item.type,
                    "size": item.size,
                    "url": item.url,
                },
            )
            for item in sorted(items, key=lambda x: (x.type != "dir", x.name.lower()))
        ]

    def _create_error_document(self, error: Exception, path: str) -> Document:
        """Create a Document from an error"""
        return Document(
            content=str(error),
            meta={
                "type": "error",
                "path": path,
            },
        )

    @component.output_types(documents=List[Document])
    def run(
        self, path: str, repo: Optional[str] = None, branch: Optional[str] = None
    ) -> Dict[str, List[Document]]:
        """
        Process a GitHub repository path and return documents.

        :param repo: Repository in format "owner/repo"
        :param path: Path within repository (default: root)
        :param ref: Git reference (branch, tag, commit) to use
        :return: Dictionary containing list of documents
        """
        if repo is None:
            repo = self.repo
        if branch is None:
            branch = self.branch

        try:
            owner, repo_name = self._parse_repo(repo)
            normalized_path = self._normalize_path(path)

            contents = self._fetch_contents(owner, repo_name, normalized_path, branch)

            # Handle single file response
            if not isinstance(contents, list):
                if contents.get("size", 0) > self.max_file_size:
                    raise ValueError(
                        f"File size {contents['size']} exceeds limit of {self.max_file_size}"
                    )

                item = GitHubItem(
                    name=contents["name"],
                    type="file",
                    path=contents["path"],
                    size=contents["size"],
                    url=contents["html_url"],
                    content=self._process_file_content(
                        contents["content"], contents["encoding"]
                    ),
                )
                return {"documents": [self._create_file_document(item)]}

            # Handle directory listing
            items = [
                GitHubItem(
                    name=item["name"],
                    type="dir" if item["type"] == "dir" else "file",
                    path=item["path"],
                    size=item.get("size", 0),
                    url=item["html_url"],
                )
                for item in contents
            ]

            return {"documents": self._create_directory_documents(items)}

        except Exception as e:
            error_doc = self._create_error_document(
                f"Error processing repository path {path}: {str(e)}. Seems like the file does not exist.", path
            )
            if self.raise_on_failure:
                raise
            logger.warning(
                "Error processing repository path {path}: {error}",
                path=path,
                error=str(e),
            )
            return {"documents": [error_doc]}

