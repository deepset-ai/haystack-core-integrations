# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

from haystack import default_from_dict, default_to_dict
from haystack.tools import ComponentTool
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.connectors.github.file_editor import GitHubFileEditor
from haystack_integrations.prompts.github.file_editor_tool import FILE_EDITOR_PROMPT, FILE_EDITOR_SCHEMA


class GitHubFileEditorTool(ComponentTool):
    """
    A Haystack tool for editing files in GitHub repositories.
    """

    def __init__(
        self,
        name: Optional[str] = "file_editor",
        description: Optional[str] = FILE_EDITOR_PROMPT,
        parameters: Optional[Dict[str, Any]] = FILE_EDITOR_SCHEMA,
        github_token: Secret = Secret.from_env_var("GITHUB_TOKEN"),
        repo: Optional[str] = None,
        branch: str = "main",
        raise_on_failure: bool = True,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.github_token = github_token
        self.repo = repo
        self.branch = branch
        self.raise_on_failure = raise_on_failure

        file_editor = GitHubFileEditor(
            github_token=github_token,
            repo=repo,
            branch=branch,
            raise_on_failure=raise_on_failure,
        )
        super().__init__(
            component=file_editor,
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
            github_token=self.github_token.to_dict(),
            repo=self.repo,
            branch=self.branch,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubFileEditorTool":
        """
        Deserializes the tool from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized tool.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["github_token"])
        return default_from_dict(cls, data)
