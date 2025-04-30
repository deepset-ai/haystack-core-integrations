# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
comment_prompt = """Haystack-Agent uses this tool to post a comment to a GitHub-issue discussion.

<usage>
Pass a `comment` string to post a comment.
</usage>

IMPORTANT
Haystack-Agent MUST pass "comment" to this tool. Otherwise, comment creation fails.
Haystack-Agent always passes the contents of the comment to the "comment" parameter when calling this tool.
"""

comment_schema = {
    "properties": {
        "comment": {"type": "string", "description": "The contents of the comment that you want to create."}
    },
    "required": ["comment"],
    "type": "object",
}
