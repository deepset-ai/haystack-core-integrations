---
name: haystack-agent-building
description: Build and transform safe Haystack 3 Agent harnesses using typed Agent Pack optimization recipes.
---

# Haystack Agent building for harness optimization

Use this skill whenever proposing an optimized Haystack Agent harness.

## Required approach

1. Preserve the supplied reference Agent. Every candidate is created from it with `Agent.clone`.
2. Use only the model IDs listed in `approved_models` and the tool names listed in `approved_tools`. A recipe naming
   anything else is rejected when the candidate is materialized, which wastes a campaign slot.
3. Prefer a model substitution before increasing harness complexity.
4. Use `AgentTool` when delegating work to a specialist Agent.
5. Keep specialist tool sets narrow and give each specialist a specific system prompt.
6. Never emit Python, arbitrary component dictionaries, import paths, credentials, or deployment operations.
7. Return only a JSON array containing the supported recipe objects below. No prose, no code fences.

## Supported recipes

Model substitution:

```json
{"kind": "model_substitution", "model_id": "approved-model-id"}
```

Prompt or generation settings:

```json
{
  "kind": "prompt_and_generation",
  "system_prompt": "replacement prompt",
  "generation_kwargs": {"temperature": 0}
}
```

Tool selection. Names are sorted and de-duplicated, so two orderings of the same set are one candidate:

```json
{"kind": "tool_selection", "tool_names": ["approved_tool"]}
```

Specialist delegation through `AgentTool`. The specialist runs with its own prompt, no user prompt, and the text exit
condition. `coordinator_system_prompt` is optional: when omitted, the reference prompt is kept and a short
instruction naming the specialist tool is appended. Supply it when the coordinator's role changes materially:

```json
{
  "kind": "specialist_delegation",
  "name": "retrieval_specialist",
  "description": "Find grounded evidence for an enterprise question.",
  "specialist_tool_names": ["search_documents"],
  "specialist_system_prompt": "Retrieve relevant evidence and report it with document identifiers.",
  "coordinator_tool_names": [],
  "specialist_model_id": "approved-model-id",
  "coordinator_system_prompt": null
}
```

Several compatible transformations may be combined explicitly. Each recipe applies to the result of the previous one:

```json
{"kind": "composite", "recipes": [{"kind": "model_substitution", "model_id": "approved-model-id"}]}
```

Registered structural transformations are only available when the task's `registered_structural_recipes` field lists
the name and its parameter schema. A schema maps each parameter to `str`, `int`, `float`, `bool`, `list`, or `dict`,
with a trailing `?` marking it optional. Parameters outside the schema, or of the wrong type, are rejected:

```json
{"kind": "registered_structure", "name": "listed_name", "parameters": {"declared_parameter": "value"}}
```

## Selection principles

- Quality is a hard gate. A candidate that scores below the floor is discarded regardless of price.
- Cost is the primary optimization objective after the gate; latency is secondary. Check the `objectives` field:
  when `primary` is `latency`, rank on latency instead.
- Every model in `approved_models` is already cleared for use, so choose between them on cost, latency, and how
  likely they are to hold quality — not on provider or deployment.
- Avoid removing tools demonstrated as necessary by the successful reference traces.
- For enterprise RAG, preserve retrieval grounding, citation behavior, metadata inspection, and absence handling.
- Make one interpretable change per candidate unless a composite is required to express a coherent topology. A
  campaign compares candidates against each other, so a single-change candidate tells you what caused the difference.
