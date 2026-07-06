# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Give a Haystack Agent a shell over a Mirage virtual filesystem.

This example builds a tiny "log triage" agent. A directory of log files is mounted read-only, and the
agent is handed a single `MirageShellTool` it can drive with ordinary bash (ls / cat / grep / ...). It
then answers a question by exploring the files itself, instead of you pre-loading their contents.

The same pattern works for any Mirage backend (S3, Postgres, Google Drive, ...): swap the `MirageMount`
below for the backend you want. Here we use a local `disk` mount so the example is fully self-contained.

Prerequisites:
    pip install mirage-haystack
    export OPENAI_API_KEY=...      # this example uses OpenAIChatGenerator

Run:
    python examples/agent_with_mirage_tool.py
"""

import os
import tempfile

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.tools.mirage import MirageMount, MirageShellTool, MirageWorkspace

# --- 1. Create some sample data on disk -------------------------------------------------------------
# In a real setup this would already exist (a log directory, an S3 bucket, a Drive folder, ...).
SAMPLE_LOGS = {
    "api.log": (
        "INFO  request /health 200\n"
        "ERROR db connection timeout\n"
        "INFO  request /users 200\n"
        "ERROR db connection timeout\n"
        "WARN  slow query 1200ms\n"
    ),
    "worker.log": ("INFO  job 41 started\nINFO  job 41 done\nERROR job 42 failed: OutOfMemory\nINFO  job 43 started\n"),
}


def main() -> None:
    data_dir = tempfile.mkdtemp(prefix="mirage-example-logs-")
    for name, body in SAMPLE_LOGS.items():
        with open(os.path.join(data_dir, name), "w", encoding="utf-8") as fh:
            fh.write(body)

    # --- 2. Describe the workspace ------------------------------------------------------------------
    # The mount is read-only, which is the *authoritative* write boundary: Mirage refuses any write to
    # it regardless of the command the model chooses. Mirage never shells out to the host either, so
    # the agent's blast radius is confined to what you mount here.
    workspace = MirageWorkspace(
        mounts=[
            MirageMount(path="/logs", resource="disk", config={"root": data_dir}, read_only=True),
        ]
    )

    # --- 3. Wrap it as a Tool for the Agent ---------------------------------------------------------
    tool = MirageShellTool(workspace)

    # --- 4. Build the Agent -------------------------------------------------------------------------
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
        tools=[tool],
        system_prompt=(
            "You are a log-triage assistant. A virtual filesystem is available through the "
            "`mirage_shell` tool. Use bash commands (ls, cat, grep, wc, ...) to inspect the mounted "
            "files under /logs before answering. Base your answer only on what the files actually show."
        ),
    )
    agent.warm_up()

    # --- 5. Ask a question that requires exploring the filesystem -----------------------------------
    question = (
        "Across all files in /logs, what is the single most common ERROR message, and how many times does it occur?"
    )
    print(f"Q: {question}\n")

    result = agent.run(messages=[ChatMessage.from_user(question)])

    # Show the tool calls the agent made along the way, then its final answer.
    for message in result["messages"]:
        for call in message.tool_calls or []:
            print(f"  [tool call] {call.tool_name}: {call.arguments.get('command')!r}")

    print(f"\nA: {result['messages'][-1].text}")

    tool.close()


if __name__ == "__main__":
    main()
