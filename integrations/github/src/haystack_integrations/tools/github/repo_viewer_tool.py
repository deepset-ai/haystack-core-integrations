# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Dict, Optional, Union

from haystack import default_from_dict, default_to_dict
from haystack.tools import ComponentTool
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

from haystack_integrations.components.connectors.github.repo_viewer import GitHubRepoViewer
from haystack_integrations.prompts.github.repo_viewer_prompt import REPO_VIEWER_PROMPT, REPO_VIEWER_SCHEMA


class GitHubRepoViewerTool(ComponentTool):
    """
    A tool for viewing files and directories in GitHub repositories.

    :param name: Optional name for the tool.
    :param description: Optional description.
    :param parameters: Optional JSON schema defining the parameters expected by the Tool.
    :param github_token: GitHub personal access token for API authentication
    :param repo: Default repository in owner/repo format
    :param branch: Default branch to work with
    :param raise_on_failure: If True, raises exceptions on API errors
    :param max_file_size: Maximum file size in bytes to read
    :param outputs_to_string:
        Optional dictionary defining how a tool outputs should be converted into a string.
        If the source is provided only the specified output key is sent to the handler.
        If the source is omitted the whole tool result is sent to the handler.
        Example: {
            "source": "docs", "handler": format_documents
        }
    :param inputs_from_state:
        Optional dictionary mapping state keys to tool parameter names.
        Example: {"repository": "repo"} maps state's "repository" to tool's "repo" parameter.
    :param outputs_to_state:
        Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
        If the source is provided only the specified output key is sent to the handler.
        Example: {
            "documents": {"source": "docs", "handler": custom_handler}
        }
        If the source is omitted the whole tool result is sent to the handler.
        Example: {
            "documents": {"handler": custom_handler}
        }
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
        self.name = name
        self.description = description
        self.parameters = parameters
        self.github_token = github_token
        self.repo = repo
        self.branch = branch
        self.raise_on_failure = raise_on_failure
        self.max_file_size = max_file_size
        self.outputs_to_string = outputs_to_string
        self.inputs_from_state = inputs_from_state
        self.outputs_to_state = outputs_to_state

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
            outputs_to_string=outputs_to_string,
            inputs_from_state=inputs_from_state,
            outputs_to_state=outputs_to_state,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the tool to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialized = default_to_dict(
            self,
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            github_token=self.github_token.to_dict() if self.github_token else None,
            repo=self.repo,
            branch=self.branch,
            raise_on_failure=self.raise_on_failure,
            max_file_size=self.max_file_size,
            outputs_to_string=self.outputs_to_string,
            inputs_from_state=self.inputs_from_state,
            outputs_to_state=self.outputs_to_state,
        )

        # Handle serialization of callable handlers based on the code in ComponentTool.to_dict
        if self.outputs_to_state is not None:
            serialized_outputs = {}
            for key, config in self.outputs_to_state.items():
                serialized_config = config.copy()
                if "handler" in config:
                    serialized_config["handler"] = serialize_callable(config["handler"])
                serialized_outputs[key] = serialized_config
            serialized["init_parameters"]["outputs_to_state"] = serialized_outputs

        if self.outputs_to_string is not None and self.outputs_to_string.get("handler") is not None:
            serialized["init_parameters"]["outputs_to_string"] = {
                **self.outputs_to_string,
                "handler": serialize_callable(self.outputs_to_string["handler"]),
            }

        return serialized

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

        # Handle deserialization of callable handlers based on the code in ComponentTool.from_dict
        if "outputs_to_state" in data["init_parameters"] and data["init_parameters"]["outputs_to_state"]:
            deserialized_outputs = {}
            for key, config in data["init_parameters"]["outputs_to_state"].items():
                deserialized_config = config.copy()
                if "handler" in config:
                    deserialized_config["handler"] = deserialize_callable(config["handler"])
                deserialized_outputs[key] = deserialized_config
            data["init_parameters"]["outputs_to_state"] = deserialized_outputs

        if (
            data["init_parameters"].get("outputs_to_string") is not None
            and data["init_parameters"]["outputs_to_string"].get("handler") is not None
        ):
            data["init_parameters"]["outputs_to_string"]["handler"] = deserialize_callable(
                data["init_parameters"]["outputs_to_string"]["handler"]
            )

        return default_from_dict(cls, data)
