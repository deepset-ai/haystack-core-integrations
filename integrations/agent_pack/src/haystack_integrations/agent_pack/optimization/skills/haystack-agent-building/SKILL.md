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
3. Prefer a model substitution before changing the prompt or the tool set.
4. Make one interpretable change per candidate. A campaign compares candidates against each other, so a
   single-change candidate tells you what caused the difference. To combine two changes, propose them as two
   candidates.
5. Never emit Python, arbitrary component dictionaries, import paths, credentials, or deployment operations.
6. Return only a JSON array containing the supported recipe objects below. No prose, no code fences.

## Supported recipes

There are three. Anything else is rejected.

Model substitution — swap the coordinator model for an approved one:

```json
{"kind": "model_substitution", "model_id": "approved-model-id"}
```

Prompt or generation settings — either field may be omitted, but not both. `generation_kwargs` is merged over the
reference generator's own settings:

```json
{
  "kind": "prompt_and_generation",
  "system_prompt": "replacement prompt",
  "generation_kwargs": {"temperature": 0}
}
```

Tool selection — restrict the Agent to a subset of the tools it already has. Names are sorted and de-duplicated, so
two orderings of the same set are one candidate. Exit conditions naming a dropped tool are pruned automatically:

```json
{"kind": "tool_selection", "tool_names": ["approved_tool"]}
```

## Selection principles

- Quality is a hard gate. A candidate that scores below the floor is discarded regardless of price.
- Cost is the primary optimization objective after the gate; latency is secondary. Check the `objectives` field:
  when `primary` is `latency`, rank on latency instead.
- Every model in `approved_models` is already cleared for use, so choose between them on cost, latency, and how
  likely they are to hold quality — not on provider or deployment.
- Do not remove tools the successful reference traces show being used to reach the answer. Read
  `successful_trace_inputs` before proposing a tool selection.
- For enterprise RAG, preserve retrieval grounding, citation behavior, metadata inspection, and absence handling.
