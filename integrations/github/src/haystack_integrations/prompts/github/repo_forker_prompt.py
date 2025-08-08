# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
REPO_FORKER_PROMPT = """Haystack-Agent uses this tool to fork GitHub repositories in order to contribute to issues.
Haystack-Agent initiates a fork so it can freely make changes for contributions.
A fork is required to open a pull request to the upstream repository.
Haystack-Agent works by forking the repository associated with a given issue.

<usage>
Pass a `url` string for the GitHub issue you want to work on in a fork.
It is REQUIRED to pass `url` to use this tool.
The structure must be "https://github.com/<repo-owner>/<repo-name>/issues/<issue-number>".

Examples:

- {"url": "https://github.com/deepset-ai/haystack/issues/9343"}
    - will fork the "deepset-ai/haystack" repository to work on issue 9343
- {"url": "https://github.com/deepset-ai/haystack-core-integrations/issues/1685"}
    - will fork the "deepset-ai/haystack-core-integrations" repository to work on issue 1685
</usage>

Haystack-Agent uses the `repo_forker` tool to create a copy (fork) of the target repository into its own account.
Haystack-Agent ensures the issue URL is valid and points to a real GitHub issue.
It parses the URL to identify the correct repository.

<thinking>
- Does this issue belong to the repository I need to work on?
- Can I extract the owner and repository name from the URL?
- Why am I forking this repository? (e.g., to implement a fix, to add a feature)
- Is there anything special about the branch or base state I should be aware of?
</thinking>

Haystack-Agent reflects on the results after forking:
<thinking>
- Did the fork succeed? Is the fork visible in my account?
- Can I access, clone, and push to my fork?
- Are there any permissions or fork-specific settings to configure before proceeding?
- Which branch will I be working on in the fork?
</thinking>

IMPORTANT
Haystack-Agent ONLY forks the repository mentioned in the given issue URL.
Haystack-Agent does NOT attempt to fork organizations, user profiles, or non-issue URLs.
Haystack-Agent knows that forking is a prerequisite to contributing changes and creating pull requests.

Haystack-Agent takes notes after the fork:
<scratchpad>
- Record the URL of the forked repository
- Note the original issue being worked on
- Document any post-fork steps (e.g., git cloning, installing dependencies)
- Make note of any errors or special setup requirements
</scratchpad>
"""

REPO_FORKER_SCHEMA = {
    "properties": {
        "url": {"type": "string", "description": "URL of the GitHub issue to work on in the fork."},
    },
    "required": ["url"],
    "type": "object",
}
