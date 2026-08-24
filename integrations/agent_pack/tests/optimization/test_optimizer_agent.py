import pytest
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
    StructuralRecipeRegistry,
    TraceArtifact,
    create_harness_optimizer_agent,
    create_haystack_docs_toolset,
)
from haystack_integrations.agent_pack.optimization.optimizer_agent import bundled_agent_building_skills_path

SKILL_NAME = "haystack-agent-building"


def reference_trace():
    return TraceArtifact(
        run_id="run",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        duration_ms=1000,
        status="success",
        traces=(
            {
                "span_id": "root",
                "operation_name": "haystack.agent.run",
                "parent_span_id": None,
                "tags": {"haystack.agent.input": {"messages": [ChatMessage.from_user("q").to_dict()]}},
            },
        ),
    )


def proposer_for(responses, **kwargs):
    return HarnessOptimizerAgentProposer(
        optimizer_agent=create_harness_optimizer_agent(chat_generator=MockChatGenerator(responses)), **kwargs
    )


def propose_with(proposer, assets=None, objectives=None):
    return proposer.propose(
        reference=Agent(chat_generator=MockChatGenerator(model="reference")),
        reference_traces=[reference_trace()],
        assets=assets
        or ApprovedAssetCatalog(
            models=[
                ModelAsset(model_id="reference", provider="p", deployment="d"),
                ModelAsset(model_id="cheap", provider="p", deployment="d"),
            ],
            tools=[],
        ),
        objectives=objectives or OptimizationObjectives(),
    )


def test_bundled_skill_ships_with_the_package():
    """The skill is a data file inside the wheel, so a packaging change must not silently drop it."""
    skill = bundled_agent_building_skills_path() / SKILL_NAME / "SKILL.md"
    assert skill.is_file()
    body = skill.read_text(encoding="utf-8")
    assert f"name: {SKILL_NAME}" in body
    assert "Agent.clone" in body
    assert "AgentTool" in body
    assert "registered_structural_recipes" in body


def test_optimizer_agent_exposes_bundled_agent_building_skill():
    agent = create_harness_optimizer_agent(chat_generator=MockChatGenerator("[]"))
    assert isinstance(agent.tools[0], SkillToolset)
    assert set(agent.tools[0].skills) == {SKILL_NAME}
    assert agent.exit_conditions == ["text"]


def test_optimizer_agent_accepts_optional_read_only_docs_toolset():
    @tool
    def search_haystack_docs(query: str) -> str:
        """Search official Haystack documentation."""
        return query

    docs = Toolset([search_haystack_docs])
    agent = create_harness_optimizer_agent(chat_generator=MockChatGenerator("[]"), docs_toolset=docs)
    assert agent.tools == [agent.tools[0], docs]


def test_haystack_docs_toolset_is_read_only_and_lazily_connected():
    pytest.importorskip("haystack_integrations.tools.mcp", reason="mcp-haystack is optional")
    toolset = create_haystack_docs_toolset()
    assert toolset.tool_names == ["search_haystack_docs"]
    assert toolset.server_info.url == "https://docs.haystack.deepset.ai/api/mcp"
    assert toolset.eager_connect is False


def test_agent_proposer_only_accepts_typed_recipe_json():
    proposals = propose_with(proposer_for('[{"kind":"model_substitution","model_id":"cheap"}]'))
    assert proposals == [ModelSubstitutionRecipe(model_id="cheap")]


def test_agent_proposer_tolerates_fenced_or_prefixed_json():
    """A code fence or a sentence of preamble is a formatting slip, not a reason to abort a campaign."""
    response = 'Here is my proposal:\n```json\n[{"kind": "model_substitution", "model_id": "cheap"}]\n```'
    assert propose_with(proposer_for(response)) == [ModelSubstitutionRecipe(model_id="cheap")]


def test_agent_proposer_retries_once_with_corrective_feedback():
    proposer = proposer_for(["not json at all", '[{"kind":"model_substitution","model_id":"cheap"}]'])
    assert propose_with(proposer) == [ModelSubstitutionRecipe(model_id="cheap")]


def test_agent_proposer_gives_up_after_max_attempts():
    with pytest.raises(ValueError, match="did not return a valid typed recipe array"):
        propose_with(proposer_for("still not json"))


def test_agent_proposer_rejects_untyped_and_oversized_responses():
    with pytest.raises(ValueError, match="did not return a valid typed recipe array"):
        propose_with(proposer_for('[{"kind": "python", "code": "dangerous()"}]'))
    with pytest.raises(ValueError, match="did not return a valid typed recipe array"):
        propose_with(
            proposer_for(
                '[{"kind":"model_substitution","model_id":"cheap"}, '
                '{"kind":"model_substitution","model_id":"reference"}]',
                max_recipes=1,
            )
        )


def test_registered_structural_recipes_are_offered_to_the_optimizer():
    registry = StructuralRecipeRegistry()
    registry.register("short-run", lambda agent, _assets, _params: agent, parameters_schema={"steps": "int"})
    proposer = proposer_for(
        '[{"kind":"registered_structure","name":"short-run","parameters":{"steps":3}}]', registry=registry
    )
    request = proposer.build_request(
        reference=Agent(chat_generator=MockChatGenerator(model="reference")),
        reference_traces=[reference_trace()],
        assets=ApprovedAssetCatalog(models=[ModelAsset(model_id="reference", provider="p", deployment="d")], tools=[]),
        objectives=OptimizationObjectives(),
    )
    assert request["registered_structural_recipes"] == {"short-run": {"steps": "int"}}
    assert propose_with(proposer)[0].name == "short-run"
