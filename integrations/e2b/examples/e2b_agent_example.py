# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Example: Haystack Agent with E2B sandbox tools.

Demonstrates that all four tools (run_bash_command, read_file, write_file,
list_directory) share the same sandbox instance, so the agent can write a
file in one step and read it back / execute it in the next.

Requirements:
    pip install e2b-haystack openai

Environment variables:
    E2B_API_KEY   - your E2B API key
    OPENAI_API_KEY - your OpenAI API key  (or swap the generator below)
"""

import sys

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.tools.e2b import (
    E2BSandbox,
    ListDirectoryTool,
    ReadFileTool,
    RunBashCommandTool,
    WriteFileTool,
)

# ---------------------------------------------------------------------------
# Example queries that exercise cross-tool state sharing:
#   1. The agent writes a Python script to the sandbox filesystem.
#   2. It executes the script via bash and captures stdout.
#   3. It reads the output file back (or lists a directory) to verify results.
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES = [
    # Simple: purely bash-based data wrangling
    ("Generate the first 10 Fibonacci numbers using a bash one-liner and show me the results."),
    # Cross-tool: write -> execute -> read
    (
        "Write a Python script to /tmp/primes.py that prints all prime numbers "
        "up to 50, run it, and then read the file back so I can see both the "
        "script and its output."
    ),
    # Multi-step: write -> list -> bash
    (
        "Create a directory /tmp/workspace, write three small text files into it "
        "with different content, list the directory to confirm they exist, and "
        "then use bash to count the total number of words across all three files."
    ),
]


def run(query: str, model: str = "gpt-4o-mini") -> None:
    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    # One sandbox passed to each tool class - they all share the same live sandbox process.
    sandbox = E2BSandbox()
    sandbox.warm_up()
    tools = [
        RunBashCommandTool(sandbox=sandbox),
        ReadFileTool(sandbox=sandbox),
        WriteFileTool(sandbox=sandbox),
        ListDirectoryTool(sandbox=sandbox),
    ]

    agent = Agent(
        chat_generator=OpenAIChatGenerator(model=model),
        tools=tools,
        system_prompt=(
            "You are a helpful coding assistant with access to a live Linux sandbox. "
            "Use the available tools freely to explore, write files, and run commands. "
            "All tools operate inside the same sandbox environment, so files written "
            "with write_file are immediately available to run_bash_command and read_file."
        ),
        max_agent_steps=15,
    )

    result = agent.run(messages=[ChatMessage.from_user(query)])
    print("\n--- Agent response ---")
    print(result["last_message"].text)


if __name__ == "__main__":
    # Run a specific query index (0/1/2) or all of them by default.
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        run(EXAMPLE_QUERIES[idx])
    else:
        for query in EXAMPLE_QUERIES:
            run(query)
