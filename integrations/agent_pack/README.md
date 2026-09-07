# agent-pack-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/agent-pack-haystack.svg)](https://pypi.org/project/agent-pack-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/agent-pack-haystack.svg)](https://pypi.org/project/agent-pack-haystack)

- [Integration page](https://docs.haystack.deepset.ai/docs/agent-pack)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/agent_pack/CHANGELOG.md)

---

## Harness optimization

`HarnessOptimizationExperiment` wraps a reference Agent in a Pipeline and uses `Pipeline.dumps()` to create
`candidate.yaml`, with readable multiline prompts. The optimizer edits this file directly, validates it with
`Pipeline.loads()`, and submits a complete configuration for evaluation. Haystack's configured deserialization
allowlist applies throughout.

The editing tools accept no filesystem paths. Exact replacements require the latest revision and a unique match.
A failed validation can be repaired within the same optimizer turn without consuming an evaluation slot.
Submission requires the validated revision; repeated configurations are rejected even if their YAML formatting
differs. Changes build on the previous candidate, with a restore tool for the reference or any prior candidate.
Both editing steps and measured candidates are bounded.

Each invocation retains a reference YAML, immutable candidate snapshots, diffs, raw measurements, and a
`recommended.yaml` when a candidate beats the baseline and clears the quality gates, in its own artifact directory
and JSONL journal. Recommendations contain complete YAML and can be loaded with `optimization.load_agent()`.
In-memory document store contents are **not** embedded in YAML: loading in a new process requires rebuilding the
corpus with the same index, or using a persistent document store.

The Advanced RAG evaluator uses a small in-memory tracer for model usage, including `LLMRanker`'s internal generator
and the backup-answer hook. It owns global tracing during evaluation and disables it afterward. It does not log or
retain prompts, documents, or replies. Costs use model IDs reported in generator replies; supply prices for those
IDs. Missing usage on an observed model call makes cost unavailable. Components making untraced external calls need
instrumentation before their costs can be compared reliably.

Run the MultiHopRAG PoC from this integration directory with `OPENAI_API_KEY` configured and `datasets` installed:

```sh
hatch run test:python examples/harness_optimization_poc.py --max-cases 1 --max-iterations 1 --max-concurrent-cases 1
```

Use `--workspace` to separate experiment artifacts, `--config` to supply an editable YAML draft, and `--docs-mcp`
to enable documentation search. The lightweight checks measure retrieval, answer substrings and citation
references, rather than full semantic answer correctness. Metadata inspection remains available but is not
mandatory for the MultiHopRAG example.

Quality is the fraction of cases a candidate passes, and each case is measured once, so the smallest difference
the measurement can express is `1 / --max-cases`. Agent runs are not deterministic: the same configuration
re-measured over the same eight cases has scored two and three. Keep `--max-cases` well above the effect worth
detecting, and leave `--max-quality-loss` at no less than one case, or single-case noise gates out real
improvements.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to export the `OPENAI_API_KEY` and `TAVILY_API_KEY` environment variables.
