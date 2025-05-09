# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

from haystack import default_from_dict, default_to_dict
from haystack.tools import ComponentTool
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.connectors.github.issue_commenter import GitHubIssueCommenter
from haystack_integrations.prompts.github.issue_commenter_prompt import ISSUE_COMMENTER_PROMPT, ISSUE_COMMENTER_SCHEMA


class GitHubIssueCommenterTool(ComponentTool):
    """
    A tool for commenting on GitHub issues.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = "issue_commenter",
        description: Optional[str] = ISSUE_COMMENTER_PROMPT,
        parameters: Optional[Dict[str, Any]] = ISSUE_COMMENTER_SCHEMA,
        github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
        raise_on_failure: bool = True,
        retry_attempts: int = 2,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.github_token = github_token
        self.raise_on_failure = raise_on_failure
        self.retry_attempts = retry_attempts

        issue_commenter = GitHubIssueCommenter(
            github_token=github_token,
            raise_on_failure=raise_on_failure,
            retry_attempts=retry_attempts,
        )
        super().__init__(
            component=issue_commenter,
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
            raise_on_failure=self.raise_on_failure,
            retry_attempts=self.retry_attempts,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubIssueCommenterTool":
        """
        Deserializes the tool from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized tool.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["github_token"])
        return default_from_dict(cls, data)
