# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
ISSUE_VIEWER_PROMPT = """Haystack-Agent uses this tool to view a GitHub issue.
Haystack-Agent can view one issue at a time.

<usage>
Pass an `issue_url` string for the GitHub issue that you want to view.
It is required to pass `issue_url` to use this tool.
The structure is "https://github.com/repo-owner/repo-name/issues/issue-number".

Examples:

- {"issue_url": "https://github.com/deepset-ai/haystack/issues/9343"}
    - will show you the issue 9343 of the haystack repository
- {"issue_url": "https://github.com/deepset-ai/haystack-core-integrations/issues/1685"}
    - will show you the issue 1685 of the haystack-core-integrations repository
</usage>
"""

ISSUE_VIEWER_SCHEMA = {
    "properties": {"issue_url": {"type": "string", "description": "URL of the GitHub issue to link the PR to."}},
    "required": ["issue_url"],
    "type": "object",
}
