# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

from haystack import default_from_dict, default_to_dict
from haystack.tools import ComponentTool
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.connectors.github.repo_viewer import GitHubRepoViewer
from haystack_integrations.prompts.github.repo_viewer_tool import REPO_VIEWER_PROMPT, REPO_VIEWER_SCHEMA


class GitHubRepoViewerTool(ComponentTool):
    """
    A tool for viewing files and directories in GitHub repositories.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = "repo_viewer",
        description: Optional[str] = REPO_VIEWER_PROMPT,
        parameters: Optional[Dict[str, Any]] = REPO_VIEWER_SCHEMA,
        github_token: Optional[Secret] = None,
        repo: Optional[str] = None,
        branch: str = "main",
        raise_on_failure: bool = True,
        max_file_size: int = 1_000_000,  # 1MB default limit
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.github_token = github_token
        self.repo = repo
        self.branch = branch
        self.raise_on_failure = raise_on_failure
        self.max_file_size = max_file_size

        repo_viewer = GitHubRepoViewer(
            github_token=github_token,
            repo=repo,
            branch=branch,
            raise_on_failure=raise_on_failure,
            max_file_size=max_file_size,
        )
        super().__init__(
            component=repo_viewer,
            name=name,
            description=description,
            parameters=parameters,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the tool to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            github_token=self.github_token.to_dict() if self.github_token else None,
            repo=self.repo,
            branch=self.branch,
            raise_on_failure=self.raise_on_failure,
            max_file_size=self.max_file_size,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubRepoViewerTool":
        """
        Deserializes the tool from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized tool.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["github_token"])
        return default_from_dict(cls, data)
