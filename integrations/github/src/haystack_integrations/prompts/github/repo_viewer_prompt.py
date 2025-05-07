# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
REPO_VIEWER_PROMPT = """Haystack-Agent uses this tool to browse GitHub repositories.
Haystack-Agent can view directories and files with this tool.

<usage>
Pass a `repo` string for the repository that you want to view.
It is required to pass `repo` to use this tool.
The structure is "owner/repo-name".

Pass a `path` string for the directory or file that you want to view.
If you pass an empty path, you will view the root directory of the repository.

Examples:

- {"repo": "pandas-dev/pandas", "path": ""}
    - will show you the root of the pandas repository
- {"repo": "pandas-dev/pandas", "path": "pyproject.toml"}
    - will show you the "pyproject.toml"-file of the pandas repository
- {"repo": "huggingface/transformers", "path": "src/transformers/models/albert"}
    - will show you the "albert"-directory in the transformers repository
- {"repo": "huggingface/transformers", "path": "src/transformers/models/albert/albert_modelling.py"}
    - will show you the source code for the albert model in the transformers repository
</usage>

Haystack-Agent uses the `github_repository_viewer` to view relevant code.
Haystack-Agent starts at the root of the repository.
Haystack-Agent navigates one level at a time using directory listings.
Haystack-Agent views all relevant code, testing, configuration, or documentation files on a level.
It never skips a directory level or guesses full paths.

Haystack-Agent thinks deeply about the content of a repository. Before Haystack-Agent uses the tool, it reasons about
next steps:

<thinking>
- What am I looking for in this location?
- Why is this path potentially relevant?
- What specific files might help solve the issue?
- What patterns or implementations should I look for?
 </thinking>

After viewing the contents of a file or directory, Haystack-Agent reflects on its observations before moving on:
<thinking>
- What did I learn from these files?
- What else might be related?
- Where should I look next and why?
</thinking>

IMPORTANT
Haystack-Agent views the content of relevant files, it knows that it is not enough to explore the directory structure.
Haystack-Agent needs to read the code to understand it properly.
To view a file, Haystack-Agent passes the full path of the file to the `github_repository_viewer`.
Haystack-Agent never guesses a file or directory path.

Haystack-Agent takes notes after viewing code:
<scratchpad>
- extract important code snippets
- document key functions, classes or configurations
- note key architecture patterns
- relate findings to the original issue
- relate findings to other code that was already viewed
- note down file paths as a reference
</scratchpad>
"""

REPO_VIEWER_SCHEMA = {
    "properties": {
        "repo": {"type": "string", "description": "The owner/repository_name that you want to view."},
        "path": {
            "type": "string",
            "description": "Path to directory or file to view. Defaults to repository root.",
        },
        "branch": {
            "type": "string",
            "description": "Branch to view. Defaults to 'main'.",
        },
    },
    "required": ["repo"],
    "type": "object",
}
