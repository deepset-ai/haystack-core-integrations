---
name: haystack-agent-building
description: Build and transform safe Haystack 3 Agent harnesses using typed Agent Pack optimization recipes.
---

# Haystack Agent building for harness optimization

Use this skill whenever proposing an optimized Haystack Agent harness.

## Required approach

1. Preserve the supplied reference Agent. Every candidate is created with `Agent.clone`.
2. Use only approved model IDs and tool names supplied in the task.
3. Prefer a model substitution before increasing harness complexity.
4. Use `AgentTool` when delegating work to a specialist Agent.
5. Keep specialist tool sets narrow and give each specialist a specific system prompt.
6. Never emit Python, arbitrary component dictionaries, import paths, credentials, or deployment operations.
7. Return only a JSON array containing the supported recipe objects below.

## Supported recipes

Model substitution:

```json
{"kind":"model_substitution","model_id":"approved-model-id"}
```

Prompt or generation settings:

```json
{
  "kind":"prompt_and_generation",
  "system_prompt":"replacement prompt",
  "generation_kwargs":{"temperature":0}
}
```

Tool selection:

```json
{"kind":"tool_selection","tool_names":["approved_tool"]}
```

Specialist delegation through `AgentTool`:

```json
{
  "kind":"specialist_delegation",
  "name":"retrieval_specialist",
  "description":"Find grounded evidence for an enterprise question.",
  "specialist_tool_names":["search_documents"],
  "specialist_system_prompt":"Retrieve relevant evidence and report it with document identifiers.",
  "coordinator_tool_names":[],
  "specialist_model_id":"approved-model-id"
}
```

Several compatible transformations may be combined explicitly:

```json
{"kind":"composite","recipes":[{"kind":"model_substitution","model_id":"approved-model-id"}]}
```

Only use `registered_structure` when the task explicitly lists the registered recipe name and its parameter schema.

## Selection principles

- Quality and sovereignty are hard gates.
- Cost is the primary optimization objective after the gates; latency is secondary.
- Avoid removing tools demonstrated as necessary by successful reference traces.
- For enterprise RAG, preserve retrieval grounding, citation behavior, metadata inspection, and absence handling.
- Make one interpretable change per candidate unless a composite is required to express a coherent topology.
