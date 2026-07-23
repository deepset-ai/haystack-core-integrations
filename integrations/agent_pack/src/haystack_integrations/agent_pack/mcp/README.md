# Deep Research MCP Server

Serve the Haystack [deep research agent](../deep_research/) over the
[Model Context Protocol](https://modelcontextprotocol.io) so any MCP client — Claude Code,
Claude Desktop, Codex, and others — can call it. The client sends a question; the **real**
Haystack agent does the research and returns a cited markdown report. Nothing about the agent is
re-implemented on the client side.

The server exposes one tool:

| Tool | Arguments | Returns |
|---|---|---|
| `deep_research` | `question: str`, `max_subtopics: int` (optional) | The final report as markdown, with inline `[text](url)` citations. |

## Requirements

- `OPENAI_API_KEY` and `TAVILY_API_KEY` in the server's environment.
- [`uv`](https://docs.astral.sh/uv/) if you use the zero-install invocation below (recommended).

## Running it

The `mcp` extra pulls in the MCP SDK plus the agent's runtime dependencies, and the package
ships an `agent-pack-deep-research-mcp` console script. So the whole server is one command with
nothing to pre-install — `uvx` builds and caches the environment on first run:

```bash
uvx --from "agent-pack-haystack[mcp]" agent-pack-deep-research-mcp
```

Equivalently, if you have installed the package yourself (`uv pip install "agent-pack-haystack[mcp]"`):

```bash
agent-pack-deep-research-mcp
# or, for local development from a checkout:
python -m haystack_integrations.agent_pack.mcp.deep_research
```

The server communicates over stdio (the default MCP transport), so you normally don't run it by
hand — you point an MCP client at that command and the client launches it.

## Client configuration

Everywhere below, the command is the same `uvx ...` line; only the config format differs. Put
your real keys in the `env` block (or drop `env` and let the server inherit them from the shell
that launches the client).

**Claude Code** (adds it to the current project):

```bash
claude mcp add deep-research \
  --env OPENAI_API_KEY=sk-... \
  --env TAVILY_API_KEY=tvly-... \
  -- uvx --from "agent-pack-haystack[mcp]" agent-pack-deep-research-mcp
```

**Claude Desktop / generic `mcpServers` JSON:**

```json
{
  "mcpServers": {
    "deep-research": {
      "command": "uvx",
      "args": ["--from", "agent-pack-haystack[mcp]", "agent-pack-deep-research-mcp"],
      "env": { "OPENAI_API_KEY": "sk-...", "TAVILY_API_KEY": "tvly-..." }
    }
  }
}
```

**Codex** (`~/.codex/config.toml`):

```toml
[mcp_servers.deep-research]
command = "uvx"
args = ["--from", "agent-pack-haystack[mcp]", "agent-pack-deep-research-mcp"]
env = { OPENAI_API_KEY = "sk-...", TAVILY_API_KEY = "tvly-..." }
```

Once configured, ask the client to "do deep research on ..." and it will call the `deep_research`
tool and present the returned report.

## Notes

- **Warm agent.** The server builds the agent once and reuses it across calls, so repeated
  queries in a session skip re-construction — the main advantage over the one-shot
  [`deep-research` skill](../../../../skills/deep-research/) that runs the agent as a fresh
  subprocess each time.
- **Long calls.** A research run takes minutes. The agent runs in a worker thread so the server
  stays responsive and can report progress via the MCP context; some clients enforce their own
  per-tool timeouts, so raise those if a run is cut short.
- **Shipping another agent as an MCP tool** follows the same recipe: wrap the agent's
  `create_*` + `run` in a `@mcp.tool()` function, add a console-script entry point in
  `pyproject.toml`, and list its runtime deps under the `mcp` extra. The advanced RAG agent is
  the exception — it needs a user-chosen document store and retriever, which is configuration the
  server can't supply generically.
```
