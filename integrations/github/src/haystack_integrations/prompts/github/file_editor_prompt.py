# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
FILE_EDITOR_PROMPT = """Use the file editor to edit an existing file in the repository.

You must provide a 'command' for the action that you want to perform:
- edit
- create
- delete
- undo

The 'payload' contains your options for each command.

**Command 'edit'**

To edit a file, you need to provide:
1. The path to the file
2. The original code snippet from the file
3. Your replacement code
4. A commit message

The code will only be replaced if it is unique in the file. Pass a minimum of 2 consecutive lines that should
be replaced. If the original is not unique, the editor will return an error.
Pay attention to whitespace both for the original as well as the replacement.

The commit message should be short and communicate your intention.
Use the conventional commit style for your messages.

Example:
{
    "command": "edit",
    "payload": {
        "path": "README.md",
        "original": "This is a placeholder description!\\nIt should be updated.",
        "replacement": "This project helps developers test AI applications.",
        "message": "docs: README should mention project purpose."
    }
}


**Command 'create'**

To create a file, you need to provide:
1. The path for the new file
2. The content for the file
3. A commit message

The commit message should be short and communicate your intention.
Use the conventional commit style for your messages.

IMPORTANT:
You MUST ALWAYS provide 'content' when creating a new file. File creation with empty content does not work.

Example:
{
    "command": "create",
    "payload": {
        "path": "CONTRIBUTING.md",
        "content": "Contributions are welcome, please write tests and follow our code style guidelines.",
        "message": "chore: minimal instructions for contributors"
    }
}


**Command 'delete'**

To delete a file, you need to provide:
1. The path to the file to delete
2. A commit message

The commit message should be short and communicate your intention.
Use the conventional commit style for your messages.

Example:
{
    "command": "delete",
    "payload": {
        "path": "tests/components/test_messaging",
        "message": "chore: messaging feature was removed"
    }
}

**Command 'undo'**

This is how to undo your latest change.

Important notes:
- You can only undo your own changes
- You can only undo one change at a time
- You need to provide a message for the undo operation

Example:
{
    "command": "undo",
    "payload": {
        "message": "revert: undo previous commit due to failing tests"
    }
}
"""

FILE_EDITOR_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "enum": ["edit", "create", "delete", "undo"],
            "description": "The command to execute",
        },
        "payload": {
            "type": "object",
            "required": ["message"],
            "properties": {
                "message": {"type": "string"},
                "content": {"type": "string"},
                "path": {"type": "string"},
                "original": {"type": "string"},
                "replacement": {"type": "string"},
            },
        },
    },
    "required": ["command", "payload"],
}
