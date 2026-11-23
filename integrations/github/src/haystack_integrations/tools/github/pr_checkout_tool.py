# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import ComponentTool
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.connectors.github.pr_checkout import GitHubPRCheckout
from haystack_integrations.prompts.github.pr_checkout_prompt import PR_CHECKOUT_PROMPT, PR_CHECKOUT_SCHEMA
from haystack_integrations.tools.github.utils import deserialize_handlers, serialize_handlers


class GitHubPRCheckoutTool(ComponentTool):
    """
    A tool that checks out a pull request's head commit and returns its guidelines and line-numbered diff.

    Requires the `git` command line tool, version 2.31 or later. See `GitHubPRCheckout`.
    """

    def __init__(
        self,
        *,
        name: str | None = "pr_checkout",
        description: str | None = PR_CHECKOUT_PROMPT,
        parameters: dict[str, Any] | None = PR_CHECKOUT_SCHEMA,
        github_token: Secret | None = Secret.from_env_var("GITHUB_TOKEN", strict=False),
        checkout_dir: str = "repo",
        guideline_files: list[str] | None = None,
        excluded_files: list[str] | None = None,
        max_diff_chars: int = 100_000,
        include_related_docs: bool = False,
        raise_on_failure: bool = False,
        outputs_to_string: dict[str, str | Callable[[Any], str]] | None = None,
        inputs_from_state: dict[str, str] | None = None,
        outputs_to_state: dict[str, dict[str, str | Callable]] | None = None,
    ) -> None:
        """
        Initialize the GitHub pull request checkout tool.

        :param name: Optional name for the tool.
        :param description: Optional description.
        :param parameters: Optional JSON schema defining the parameters expected by the Tool.
        :param github_token: GitHub token for the API and for fetching the code. Without a token, only public
            repositories can be read.
        :param checkout_dir: Directory, relative to the working directory, that the head commit is checked out into.
        :param guideline_files: Files read from the root of the base branch and returned before the diff.
            Defaults to `AGENTS.md`, `CLAUDE.md` and `CONTRIBUTING.md`.
        :param excluded_files: `fnmatch` patterns of changed files whose diff is left out.
            Defaults to common lockfiles and minified files.
        :param max_diff_chars: Maximum number of characters of diff to return.
        :param include_related_docs: If True, list the docs pages outside the pull request that mention a class
            defined in a changed source file.
        :param raise_on_failure: If True, errors from the GitHub API raise. If False (the default), the error message
            is returned to the Agent.
        :param outputs_to_string:
            Optional dictionary defining how a tool outputs should be converted into a string.
            Defaults to `{"source": "text"}`, so the Agent receives the `text` output.
        :param inputs_from_state:
            Optional dictionary mapping state keys to tool parameter names.
            By default, the tool does not use any inputs from state.
        :param outputs_to_state:
            Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
            By default, the tool does not write to state.
        """
        self.github_token = github_token
        self.checkout_dir = checkout_dir
        self.guideline_files = guideline_files
        self.excluded_files = excluded_files
        self.max_diff_chars = max_diff_chars
        self.include_related_docs = include_related_docs
        self.raise_on_failure = raise_on_failure

        self.outputs_to_string = outputs_to_string or {"source": "text"}
        self.inputs_from_state = inputs_from_state
        self.outputs_to_state = outputs_to_state

        pr_checkout = GitHubPRCheckout(
            github_token=github_token,
            checkout_dir=checkout_dir,
            guideline_files=guideline_files,
            excluded_files=excluded_files,
            max_diff_chars=max_diff_chars,
            include_related_docs=include_related_docs,
            raise_on_failure=raise_on_failure,
        )
        super().__init__(
            component=pr_checkout,
            name=name,
            description=description,
            parameters=parameters,
            outputs_to_string=self.outputs_to_string,
            inputs_from_state=self.inputs_from_state,
            outputs_to_state=self.outputs_to_state,
        )

    def to_dict(self) -> dict[str, Any]:
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
            "checkout_dir": self.checkout_dir,
            "guideline_files": self.guideline_files,
            "excluded_files": self.excluded_files,
            "max_diff_chars": self.max_diff_chars,
            "include_related_docs": self.include_related_docs,
            "raise_on_failure": self.raise_on_failure,
            "outputs_to_string": self.outputs_to_string,
            "inputs_from_state": self.inputs_from_state,
            "outputs_to_state": self.outputs_to_state,
        }

        serialize_handlers(serialized, self.outputs_to_state, self.outputs_to_string)
        return {"type": generate_qualified_class_name(type(self)), "data": serialized}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GitHubPRCheckoutTool":
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
