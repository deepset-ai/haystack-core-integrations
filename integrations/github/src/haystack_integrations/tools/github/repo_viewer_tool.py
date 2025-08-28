# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Dict, Optional, Union

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import ComponentTool
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.connectors.github.repo_viewer import GitHubRepoViewer
from haystack_integrations.prompts.github.repo_viewer_prompt import REPO_VIEWER_PROMPT, REPO_VIEWER_SCHEMA
from haystack_integrations.tools.github.utils import deserialize_handlers, message_handler, serialize_handlers


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
        outputs_to_string: Optional[Dict[str, Union[str, Callable[[Any], str]]]] = None,
        inputs_from_state: Optional[Dict[str, str]] = None,
        outputs_to_state: Optional[Dict[str, Dict[str, Union[str, Callable]]]] = None,
    ):
        """
        Initialize the GitHub repository viewer tool.

        :param name: Optional name for the tool.
        :param description: Optional description.
        :param parameters: Optional JSON schema defining the parameters expected by the Tool.
        :param github_token: Optional GitHub personal access token for API authentication
        :param repo: Default repository in owner/repo format
        :param branch: Default branch to work with
        :param raise_on_failure: If True, raises exceptions on API errors
        :param max_file_size: Maximum file size in bytes to read
        :param outputs_to_string:
            Optional dictionary defining how a tool outputs should be converted into a string.
            By default, truncates the document.content of the viewed files to 150,000 characters each.
            If the source is provided only the specified output key is sent to the handler.
            If the source is omitted the whole tool result is sent to the handler.
            Example: {
                "source": "docs", "handler": format_documents
            }
        :param inputs_from_state:
            Optional dictionary mapping state keys to tool parameter names.
            By default, the tool does not use any inputs from state.
            Example: {"repository": "repo"} maps state's "repository" to tool's "repo" parameter.
        :param outputs_to_state:
            Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
            By default, outputs the viewed files as documents to the state.
            If the source is provided only the specified output key is sent to the handler.
            Example: {
                "documents": {"source": "docs", "handler": custom_handler}
            }
            If the source is omitted the whole tool result is sent to the handler.
            Example: {
                "documents": {"handler": custom_handler}
            }
        """
        self.github_token = github_token
        self.repo = repo
        self.branch = branch
        self.raise_on_failure = raise_on_failure
        self.max_file_size = max_file_size

        # Set default values for mutable parameters
        self.outputs_to_string = outputs_to_string or {"source": "documents", "handler": message_handler}
        self.inputs_from_state = inputs_from_state or {}
        self.outputs_to_state = outputs_to_state or {"documents": {"source": "documents"}}

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
            outputs_to_string=self.outputs_to_string,
            inputs_from_state=self.inputs_from_state,
            outputs_to_state=self.outputs_to_state,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the tool to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialized = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "github_token": self.github_token.to_dict() if self.github_token else None,
            "repo": self.repo,
            "branch": self.branch,
            "raise_on_failure": self.raise_on_failure,
            "max_file_size": self.max_file_size,
            "outputs_to_string": self.outputs_to_string,
            "inputs_from_state": self.inputs_from_state,
            "outputs_to_state": self.outputs_to_state,
        }

        serialize_handlers(serialized, self.outputs_to_state, self.outputs_to_string)
        return {"type": generate_qualified_class_name(type(self)), "data": serialized}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GitHubRepoViewerTool":
        """
        Deserializes the tool from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized tool.
        """
        inner_data = data["data"]
        deserialize_secrets_inplace(inner_data, keys=["github_token"])
        deserialize_handlers(inner_data)
        return cls(**inner_data)
