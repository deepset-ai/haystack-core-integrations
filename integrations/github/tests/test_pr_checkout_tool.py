# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import copy

from haystack.utils import Secret

from haystack_integrations.prompts.github.pr_checkout_prompt import PR_CHECKOUT_PROMPT, PR_CHECKOUT_SCHEMA
from haystack_integrations.tools.github.pr_checkout_tool import GitHubPRCheckoutTool


class TestGitHubPRCheckoutTool:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        tool = GitHubPRCheckoutTool()
        assert tool.name == "pr_checkout"
        assert tool.description == PR_CHECKOUT_PROMPT
        assert tool.parameters == PR_CHECKOUT_SCHEMA
        assert tool.github_token == Secret.from_env_var("GITHUB_TOKEN", strict=False)
        assert tool.raise_on_failure is False
        assert tool.outputs_to_string == {"source": "text"}

    def test_invoke_returns_the_text_output(self, monkeypatch):
        tool = GitHubPRCheckoutTool(github_token=None)
        monkeypatch.setattr(
            tool._component, "run", lambda repo, pull_number: {"text": f"{repo}#{pull_number}", "head_sha": "abc"}
        )

        assert tool.invoke(repo="owner/repo", pull_number=7) == {"text": "owner/repo#7", "head_sha": "abc"}

    def test_to_dict_from_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")
        tool = GitHubPRCheckoutTool(
            name="checkout_pr",
            github_token=Secret.from_env_var("ENV_VAR"),
            checkout_dir="code",
            guideline_files=["AGENTS.md"],
            excluded_files=["*.lock"],
            max_diff_chars=500,
            include_related_docs=True,
            raise_on_failure=True,
        )

        data = tool.to_dict()

        assert data == {
            "type": "haystack_integrations.tools.github.pr_checkout_tool.GitHubPRCheckoutTool",
            "data": {
                "name": "checkout_pr",
                "description": PR_CHECKOUT_PROMPT,
                "parameters": PR_CHECKOUT_SCHEMA,
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "checkout_dir": "code",
                "guideline_files": ["AGENTS.md"],
                "excluded_files": ["*.lock"],
                "max_diff_chars": 500,
                "include_related_docs": True,
                "raise_on_failure": True,
                "outputs_to_string": {"source": "text"},
                "inputs_from_state": {},
                "outputs_to_state": {},
            },
        }
        assert GitHubPRCheckoutTool.from_dict(copy.deepcopy(data)).to_dict() == data
