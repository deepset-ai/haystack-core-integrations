# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from .context_prompt import CONTEXT_PROMPT
from .file_editor_prompt import FILE_EDITOR_PROMPT, FILE_EDITOR_SCHEMA
from .issue_commenter_prompt import ISSUE_COMMENTER_PROMPT, ISSUE_COMMENTER_SCHEMA
from .pr_creator_prompt import PR_CREATOR_PROMPT, PR_CREATOR_SCHEMA
from .repo_viewer_prompt import REPO_VIEWER_PROMPT, REPO_VIEWER_SCHEMA
from .system_prompt import SYSTEM_PROMPT

__all__ = [
    "CONTEXT_PROMPT",
    "FILE_EDITOR_PROMPT",
    "FILE_EDITOR_SCHEMA",
    "ISSUE_COMMENTER_PROMPT",
    "ISSUE_COMMENTER_SCHEMA",
    "PR_CREATOR_PROMPT",
    "PR_CREATOR_SCHEMA",
    "REPO_VIEWER_PROMPT",
    "REPO_VIEWER_SCHEMA",
    "SYSTEM_PROMPT",
]
