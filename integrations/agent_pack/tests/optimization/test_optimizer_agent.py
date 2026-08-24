from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import SkillToolset, Toolset, tool

from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    HarnessOptimizerAgentProposer,
    ModelAsset,
    ModelSubstitutionRecipe,
    OptimizationObjectives,
    TraceArtifact,
    bundled_agent_building_skills_path,
    create_harness_optimizer_agent,
)


def test_optimizer_agent_exposes_bundled_agent_building_skill():
    agent = create_harness_optimizer_agent(chat_generator=MockChatGenerator("[]"))
    assert isinstance(agent.tools[0], SkillToolset)
    assert set(agent.tools[0].skills) == {"haystack-agent-building"}
    body, _ = agent.tools[0]._store.load_skill("haystack-agent-building")
    assert "Agent.clone" in body
    assert "AgentTool" in body
    assert bundled_agent_building_skills_path().is_dir()


def test_optimizer_agent_accepts_optional_read_only_docs_toolset():
    @tool
    def search_haystack_docs(query: str) -> str:
        """Search official Haystack documentation."""
        return query

    docs = Toolset([search_haystack_docs])
    agent = create_harness_optimizer_agent(chat_generator=MockChatGenerator("[]"), docs_toolset=docs)
    assert agent.tools == [agent.tools[0], docs]


def test_agent_proposer_only_accepts_typed_recipe_json():
    optimizer = create_harness_optimizer_agent(
        chat_generator=MockChatGenerator('[{"kind":"model_substitution","model_id":"cheap"}]')
    )
    proposer = HarnessOptimizerAgentProposer(optimizer)
    reference = Agent(chat_generator=MockChatGenerator(model="reference"))
    assets = ApprovedAssetCatalog(models=[ModelAsset("reference", "p", "d"), ModelAsset("cheap", "p", "d")], tools=[])
    trace = TraceArtifact(
        run_id="run",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_ms=1000,
        status="success",
        traces=(
            {
                "operation_name": "haystack.agent.run",
                "parent_span_id": None,
                "tags": {"haystack.agent.input": {"messages": [ChatMessage.from_user("q").to_dict()]}},
            },
        ),
    )
    proposals = proposer.propose(
        reference=reference,
        reference_traces=[trace],
        assets=assets,
        objectives=OptimizationObjectives(),
    )
    assert proposals == [ModelSubstitutionRecipe("cheap")]
