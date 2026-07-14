# Deep Research Agent

A small, working **deep research agent**: give it a question, and it researches the web and
produces a structured, cited markdown report. Built on Haystack v3.

## Setup

```bash
pip install agent-pack-haystack tavily-haystack trafilatura pypdf arrow
```

`tavily-haystack`, `trafilatura`, `pypdf` and `arrow` are optional runtime dependencies of the
deep research agent (web search, HTML/PDF parsing, and date rendering); install them alongside
`agent-pack-haystack` to use it.

Set `OPENAI_API_KEY` and `TAVILY_API_KEY` in the environment.

## Running the agent

```python
from haystack.dataclasses import ChatMessage
from agent_pack_haystack import create_deep_research_agent

agent = create_deep_research_agent()
result = agent.run(messages=[ChatMessage.from_user("your research question")])
print(result["report"])
```

## Configuration

Everything is configured through keyword arguments to
[`create_deep_research_agent`](agent.py) — see its docstring for the full
list, defaults, and per-argument notes.

## How it works (high level)

The whole thing is a single Haystack `Agent` — the **orchestrator** — with two **hooks** that
bracket its loop. The three logical phases are **Scope → Research → Write**:

```
  user query
      │
      ▼
┌──────────────┐
│  1. SCOPE    │   before_llm hook: rewrite the question into a focused research brief
└──────┬───────┘
       │ brief
       ▼
┌──────────────┐   the agent loop (see "The agents" below):
│ 2. RESEARCH  │   the orchestrator splits the brief into sub-questions,
│ (agent loop) │   delegates each to a sub-researcher, and collects their summaries
└──────┬───────┘
       │ brief + notes
       ▼
┌──────────────┐
│  3. WRITE    │   on_exit hook: brief + notes → cited markdown report
└──────┬───────┘
       │
       ▼
   report (markdown, with inline [text](url) citations)
```

**Scope** and **Write** are plain LLM calls (a `ChatPromptBuilder` + an `OpenAIResponsesChatGenerator`),
wrapped as serializable hook classes (`ScopeHook`, `WriteHook`):

- **Scope** runs as a `before_llm` hook: before the orchestrator's first step it turns the user
  query into a brief, stored on the agent's `State`.
- **Write** runs as an `on_exit` hook: when the orchestrator finishes, it turns the brief plus collected `notes` into 
  the final report.

`brief`, `notes` and `report` are declared in the agent's `state_schema`, so they come back as
outputs of a single `agent.run(...)` call.

---

## The agents

The Research phase is built from **two nested agents**. An agent here is a Haystack `Agent` — an
LLM that loops, calling tools, until it decides to answer.

### 1. Orchestrator (the lead agent)

The lead agent. It receives the research brief and coordinates the whole investigation.

- **Job:** split the brief into a few focused, non-overlapping sub-questions, delegate each one,
  check coverage, and stop when there's enough.
- **Parallelism:** it emits several delegation calls in a single turn, and they run **concurrently**
  (bounded by `max_concurrent_researchers`).
- **Memory:** the summaries returned by sub-researchers are appended to a shared `notes` list (the
  agent's `State`), which the writer later turns into the report.
- **Stops when:** it replies with plain text (research complete) or hits `max_orchestrator_steps`.

**Its tools:**

| Tool | What it is | What it does |
|---|---|---|
| `research_subtopic` | the **sub-researcher agent**, exposed as a tool (`ComponentTool`) | Researches ONE sub-question in an isolated context and returns a compressed, cited summary. Only that summary is shown to the orchestrator; the summary is also appended to `notes`. |
| `think_tool` | a no-op reflection tool | Lets the orchestrator pause to plan sub-questions and assess coverage between rounds. |

### 2. Sub-researcher

A reusable agent that answers ONE sub-question. The orchestrator runs it many times (in parallel),
each in its **own isolated context** — this is the key idea: each sub-researcher processes the raw
search results privately and returns only a concise summary, so the orchestrator's context stays
small and the final report stays coherent.

- **Job:** search the web, optionally read promising pages, reflect, then write a compressed
  summary with inline citations to the exact source URLs.
- **Returns:** its final text message *is* the summary (it exits as soon as it writes plain text).
- **Bounded by:** `max_researcher_steps`.

**Its tools:**

| Tool | What it is | What it does |
|---|---|---|
| `web_search` | `ComponentTool` over the Tavily integration (`TavilyWebSearch`) | Runs a web search; returns the top results as `title + exact URL + snippet`. |
| `read_url` | `PipelineTool` over a fetch→route→convert-to-text→summarize pipeline | Fetches a page (`LinkContentFetcher`), routes by MIME type (`FileTypeRouter`) to `HTMLToDocument` (Trafilatura) or `PyPDFToDocument` so **PDFs are parsed too**, and **summarizes the page toward a question the agent passes** — so only the relevant text enters the agent's context, not the full page. Used only when a search snippet is too shallow. |
| `think_tool` | a no-op reflection tool | "What did I learn? what's missing? stop or continue?" between searches. |

---

## Context management

The core challenge in a deep research agent is keeping each context window small and focused. Raw
web content (search results, full pages, PDFs) is large and noisy; if it all accumulated in a single
context, the model's output quality would degrade. We avoid that with **isolation + compression**:

- Each sub-researcher runs as its **own agent with its own `State`**, so all the messy intermediate
  content (every search result, every fetched page) stays in *its* private context.
- It finishes by writing one **short summary** (its final message). Only that summary leaves the
  sub-researcher — the raw content never reaches the orchestrator or the writer.

Two settings on the `research_subtopic` tool decide where that summary goes:

| Setting (on `research_subtopic`) | Controls | Effect |
|---|---|---|
| `outputs_to_string={"source": "last_message"}` | what the **orchestrator's LLM sees** as the tool result | Only the summary text comes back — not the sub-researcher's full message history. Keeps the orchestrator's context clean. |
| `outputs_to_state={"notes": {...}}` | what gets **saved for the writer** | The same summary is appended (as text) to the shared `notes` list, which becomes the writer's input. |

So each summary travels two ways — into the orchestrator's reasoning (so it can decide whether to
dig further) and into the `notes` accumulator (so the writer can use it) — while the bulky raw
research stays isolated and is thrown away:

```
sub-researcher  (private context: searches, pages, reflections)
      │  writes ONE short summary
      ├─ outputs_to_string → orchestrator's LLM   (decide: done, or dig more?)
      └─ outputs_to_state  → notes → writer        (final report)
```
