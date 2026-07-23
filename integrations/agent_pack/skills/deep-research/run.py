# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
CLI wrapper around the Haystack deep research agent, invoked by the `deep-research` skill.

This runs the *real* agent from `agent-pack-haystack` (see
`haystack_integrations.agent_pack.create_deep_research_agent`). The host coding agent
(Claude Code / Codex) does not re-implement anything: it decides when to run this, phrases
the question, then reads the markdown report this script writes.

Progress (scope brief, notes collected) is logged to stderr; the final report goes to a
file (`--output`, default `report.md`) and is also echoed to stdout.
"""

import argparse
import logging
import os
import sys
from typing import NoReturn


def _fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    """Parse arguments, run the deep research agent, and write the report."""
    parser = argparse.ArgumentParser(
        prog="deep-research",
        description="Research a question on the web and write a cited markdown report.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="The research question. If omitted, the question is read from stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="report.md",
        help="Path to write the markdown report to (default: report.md).",
    )
    parser.add_argument(
        "--max-subtopics",
        type=int,
        default=None,
        help="Optional cap on how many sub-questions the orchestrator may delegate (breadth).",
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip() or sys.stdin.read().strip()
    if not question:
        _fail("no question provided (pass it as an argument or on stdin)")

    for key in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
        if not os.getenv(key):
            _fail(f"{key} is not set in the environment")

    # Surface the agent's own progress logging (scope brief, each note collected) on stderr,
    # so the host has something to watch during the long research run. stdout stays clean for
    # the report.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")

    # Imported here so `--help` and the argument/env checks above stay fast and don't require
    # the (heavy) Haystack install to be present just to see usage.
    from haystack.dataclasses import ChatMessage

    from haystack_integrations.agent_pack import create_deep_research_agent

    kwargs = {}
    if args.max_subtopics is not None:
        kwargs["max_subtopics"] = args.max_subtopics

    agent = create_deep_research_agent(**kwargs)
    result = agent.run(messages=[ChatMessage.from_user(question)])
    report = result["report"]

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"report written to {args.output}\n", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
