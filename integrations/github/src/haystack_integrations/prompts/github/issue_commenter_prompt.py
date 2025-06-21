# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
ISSUE_COMMENTER_PROMPT = """Haystack-Agent uses this tool to post a comment to a GitHub-issue discussion.

<usage>
Pass a `comment` string and a `url` string to post a comment to a GitHub issue.
</usage>

IMPORTANT
Haystack-Agent MUST pass "comment" and "url" to this tool. Otherwise, comment creation fails.
Haystack-Agent always passes the contents of the comment to the "comment" parameter and
 the URL of the GitHub issue to the "url" parameter when calling this tool.
"""

ISSUE_COMMENTER_SCHEMA = {
    "properties": {
        "comment": {"type": "string", "description": "The contents of the comment that you want to create."},
        "url": {"type": "string", "description": "URL of the GitHub issue to comment on."},
    },
    "required": ["comment", "url"],
    "type": "object",
}
