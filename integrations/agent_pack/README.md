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

Agent Pack contains an experimental, local-first harness optimization API. It captures successful Agent runs in the
same `haystack-trace/v1` shape accepted by the deepset platform, evaluates typed candidate transformations, applies
approved-asset and quality gates, and returns a recommendation for explicit approval.

```python
from pathlib import Path

from haystack.dataclasses import ChatMessage
from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    CampaignJournal,
    HarnessOptimizationCampaign,
    LocalTraceStore,
    ModelAsset,
    OptimizationObjectives,
    ToolAsset,
    TraceCapturingAgentRunner,
)

trace_store = LocalTraceStore(Path(".agent-pack/traces"))
runner = TraceCapturingAgentRunner()
reference_run = runner.run(reference_agent, messages=[ChatMessage.from_user("A representative question")])
trace_store.add(reference_run.trace)

assets = ApprovedAssetCatalog(
    models=[
        ModelAsset("reference-model", "provider", "remote", input_cost_per_million=10),
        ModelAsset("approved-cheaper-model", "provider", "eu", sovereign=True, input_cost_per_million=2),
    ],
    tools=[ToolAsset(tool.name) for tool in approved_tools],
)

campaign = HarnessOptimizationCampaign(
    reference=reference_agent,
    trace_source=trace_store,
    evaluator=evaluator,
    assets=assets,
    objectives=OptimizationObjectives(min_quality=0.9, max_quality_loss=0.0),
    journal=CampaignJournal(".agent-pack/campaign.jsonl"),
)
result = campaign.run()

if result.recommendation is not None:
    candidate_for_review = result.recommendation.materialize(reference_agent, assets)
```

Candidate recipes are a closed set: approved model substitutions, prompt/generation changes, tool selection,
`AgentTool` specialist delegation, composites, and explicitly registered structural transformations. The optimizer
does not execute generated Python or arbitrary serialized components.

### Skill-guided optimizer Agent

`create_harness_optimizer_agent()` exposes a bundled Claude-format `haystack-agent-building` skill through Haystack's
`SkillToolset`. The skill teaches the optimizer to use `Agent.clone`, `AgentTool`, approved assets, and the typed recipe
language.

For current API details, the factory can also receive a read-only MCP toolset for Haystack's public documentation
server. Install `mcp-haystack`, then use `create_haystack_docs_toolset()`; it connects lazily to
`https://docs.haystack.deepset.ai/api/mcp` and exposes only `search_haystack_docs`. The MCP connection is optional and
is never used by local trace capture or deterministic model enumeration.

### Programmatic tool policy

`PolicyEnforcementStrategy` implements the existing Human-in-the-Loop `ConfirmationStrategy` protocol without
requiring human interaction. Register it in a `ConfirmationHook` at `before_tool`; the initial implementation only
allows the original invocation or rejects it with safe feedback. Unknown tools, provider failures, missing policies,
and indeterminate decisions fail closed. The existing confirmation strategy spans record the allow/reject decision.
