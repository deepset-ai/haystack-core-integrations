# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from .comment_tool import COMMENT_PROMPT, COMMENT_SCHEMA
from .file_editor_tool import FILE_EDITOR_PROMPT, FILE_EDITOR_SCHEMA
from .pr_system_prompt import PR_SYSTEM_PROMPT
from .repo_viewer_tool import REPO_VIEWER_PROMPT, REPO_VIEWER_SCHEMA
from .system_prompt import ISSUE_PROMPT

__all__ = [
    "COMMENT_PROMPT",
    "COMMENT_SCHEMA",
    "FILE_EDITOR_PROMPT",
    "FILE_EDITOR_SCHEMA",
    "ISSUE_PROMPT",
    "PR_SYSTEM_PROMPT",
    "REPO_VIEWER_PROMPT",
    "REPO_VIEWER_SCHEMA",
]
