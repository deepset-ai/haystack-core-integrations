---
name: deep-research
description: >-
  Research a question on the web and produce a structured, cited markdown report,
  using the Haystack deep research agent (an orchestrator that splits the question into
  sub-topics, delegates each to an isolated sub-researcher that searches and reads pages,
  then writes a report with inline citations). Use when the user asks for a "deep research
  report", a thorough multi-source investigation of a topic, a literature/landscape review,
  or a well-cited written answer that a single web search would not cover. Not for quick
  factual lookups — this runs a multi-minute, multi-agent web research job.
---

# Deep Research

This skill runs the **real Haystack deep research agent** (`agent-pack-haystack`) as a
subprocess. You do not do the research yourself and you do not re-implement the agent — you
choose when to run it, phrase the question well, run the command, then read and present the
report it writes.

## When to use

Use this when the user wants a thorough, cited investigation of a topic — a "deep research
report", a landscape/literature review, "look into X and write it up with sources". Do **not**
use it for a quick fact you could answer with a single search; this is a multi-minute job that
spins up several web-searching sub-agents.

## Requirements

- `uv` available on PATH (used to run the script in an ephemeral, cached environment).
- Environment variables `OPENAI_API_KEY` and `TAVILY_API_KEY` set. If either is missing, the
  script exits with a clear error — relay it to the user and ask them to set the key rather
  than trying to work around it.

## How to run

From this skill's directory, run:

```bash
uv run \
  --with agent-pack-haystack \
  --with tavily-haystack \
  --with trafilatura \
  --with pypdf \
  --with arrow \
  run.py "<the research question>"
```

- Pass the question as a single quoted argument. Spend a moment making it a good, specific
  research question (scope, timeframe, angle) — the agent scopes from exactly this text.
- The report is written to `report.md` in the working directory and also printed to stdout.
  Progress (the scoped brief, each sub-topic note as it is collected) is logged to stderr, so
  you can see it is making progress during the run.
- Optional: `--max-subtopics N` caps how many sub-questions it investigates (breadth), and
  `-o/--output PATH` changes where the report is written.

The run can take several minutes. Let it finish; do not poll or re-run it.

## After it finishes

1. Read `report.md` (or the `--output` path you chose).
2. Present it to the user. For a short question, show the report directly; for a long one,
   lead with a brief summary and the key citations, and point them at the full `report.md`.
3. The report already contains inline `[text](url)` citations — preserve them; do not strip or
   fabricate sources.

## Notes

- This is a black box by design: the agent manages its own sub-researchers, web search
  (Tavily), page reading (HTML + PDF), and writing internally. You interact with it only
  through the question in and the report out.
- The same `run.py` works from any coding agent that can run a shell command (e.g. Codex via
  AGENTS.md), not just Claude Code.
- The first `uv run` builds and caches the environment, so it is slower than later runs.
