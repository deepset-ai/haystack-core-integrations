# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

from haystack import default_from_dict, default_to_dict
from haystack.tools import ComponentTool
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.connectors.github.pr_creator import GitHubPRCreator
from haystack_integrations.prompts.github.pr_system_prompt import PR_SYSTEM_PROMPT, PR_SCHEMA


class GitHubPRCreatorTool(ComponentTool):
    """
    A tool for creating pull requests in GitHub repositories.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = "pr_creator",
        description: Optional[str] = PR_SYSTEM_PROMPT,
        parameters: Optional[Dict[str, Any]] = PR_SCHEMA,
        github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
        raise_on_failure: bool = True,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.github_token = github_token
        self.raise_on_failure = raise_on_failure

        pr_creator = GitHubPRCreator(
            github_token=github_token,
            raise_on_failure=raise_on_failure,
        )
        super().__init__(
            component=pr_creator,
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
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubPRCreatorTool":
        """
        Deserializes the tool from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized tool.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["github_token"])
        return default_from_dict(cls, data)
