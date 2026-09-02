# agent-pack-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/agent-pack-haystack.svg)](https://pypi.org/project/agent-pack-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/agent-pack-haystack.svg)](https://pypi.org/project/agent-pack-haystack)

- [Integration page](https://docs.haystack.deepset.ai/docs/agent-pack)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/agent_pack/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to export the `OPENAI_API_KEY` and `TAVILY_API_KEY` environment variables.

## Experimental harness optimization

Agent Pack contains an experimental, local-first harness optimization API. It captures successful Agent runs,
evaluates a closed set of typed candidate transformations against them, checks every candidate against a catalog of
the models and tools it is allowed to use, applies the quality gate, and returns a recommendation for a human to
approve. Nothing is promoted or deployed automatically, and the optimizer never executes generated Python or
arbitrary serialized components.

Runs are captured by `haystack_integrations.agent_pack.tracing`, a standalone module that records Haystack spans as
data. It depends only on Haystack, and is a candidate for moving upstream once its shape settles.

```python
from haystack_integrations.agent_pack.optimization import HarnessOptimizationExperiment

result = HarnessOptimizationExperiment(
    reference=reference_agent,
    trace_source=trace_store,
    evaluator=evaluator,
    assets=approved_assets,
    objectives=objectives,
    journal=journal,
).run()
```

See `examples/harness_optimization_poc.py` for a runnable end-to-end walkthrough, and the
[API reference](https://docs.haystack.deepset.ai/reference/integrations-agent-pack) for the full surface. This API
may change without a deprecation period.
