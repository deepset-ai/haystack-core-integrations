import json

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator, OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Toolset, tool

from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    EvaluationMetrics,
    HarnessOptimizerAgentProposer,
    HarnessPatch,
    ModelAsset,
    ModelSubstitutionRecipe,
    OptimizationObjectives,
    create_harness_optimizer_agent,
    create_haystack_documentation_mcp_toolset,
)
from haystack_integrations.agent_pack.optimization.proposer import HARNESS_OPTIMIZER_SYSTEM_PROMPT
from haystack_integrations.agent_pack.optimization.recipes import RECIPE_KINDS, proposal_json_schema
from haystack_integrations.agent_pack.runs import AgentRunRecord


def reference_run():
    return AgentRunRecord(
        run_id="run",
        inputs={"messages": [ChatMessage.from_user("q")]},
        outputs={"last_message": ChatMessage.from_assistant("a")},
    )


def catalog():
    return ApprovedAssetCatalog(
        models=[ModelAsset(model_id="reference"), ModelAsset(model_id="cheap")],
        patches=[HarnessPatch(name="reasoning-high", patch={"a.b": 1})],
    )


def proposer_for(responses, **kwargs):
    return HarnessOptimizerAgentProposer(
        optimizer_agent=create_harness_optimizer_agent(chat_generator=MockChatGenerator(responses)), **kwargs
    )


def propose_with(proposer, *, history=None):
    return proposer.propose(
        reference=Agent(chat_generator=MockChatGenerator(model="reference")),
        reference_runs=[reference_run()],
        assets=catalog(),
        objectives=OptimizationObjectives(),
        baseline=EvaluationMetrics(quality=1.0, cost=10.0, latency_ms=100),
        history=history or [],
    )


def test_system_prompt_documents_exactly_the_supported_recipe_kinds():
    for kind in RECIPE_KINDS:
        assert kind in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    assert "one JSON object" in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    assert "complete history" in HARNESS_OPTIMIZER_SYSTEM_PROMPT


def test_optimizer_agent_defaults_and_optional_docs_toolset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    default = create_harness_optimizer_agent()
    assert isinstance(default.chat_generator, OpenAIResponsesChatGenerator)
    assert default.chat_generator.model == "gpt-5.6-sol"
    assert default.system_prompt == HARNESS_OPTIMIZER_SYSTEM_PROMPT

    @tool
    def search_haystack_docs(query: str) -> str:
        """Search official Haystack documentation."""
        return query

    docs = Toolset([search_haystack_docs])
    with_docs = create_harness_optimizer_agent(chat_generator=MockChatGenerator("{}"), docs_toolset=docs)
    assert with_docs.tools == [docs]


def test_haystack_documentation_mcp_server_is_read_only_and_lazy():
    pytest.importorskip("haystack_integrations.tools.mcp", reason="mcp-haystack is optional")
    toolset = create_haystack_documentation_mcp_toolset()
    assert toolset.tool_names == ["search_haystack_docs"]
    assert toolset.eager_connect is False


def test_proposal_schema_closes_over_catalog_choices():
    schema = proposal_json_schema(assets=catalog())
    definitions = schema["$defs"]
    assert definitions["ModelSubstitutionRecipe"]["properties"]["model_id"]["enum"] == ["cheap", "reference"]
    assert definitions["ApplyPatchRecipe"]["properties"]["patch"]["enum"] == ["reasoning-high"]
    recipe_definitions = (definition for definition in definitions.values() if "kind" in definition["properties"])
    assert all("kind" in definition["required"] for definition in recipe_definitions)


def test_agent_proposer_returns_one_typed_recipe_or_stops():
    proposer = proposer_for('{"recipe": {"kind": "model_substitution", "model_id": "cheap"}}')
    assert propose_with(proposer) == ModelSubstitutionRecipe(model_id="cheap")
    assert propose_with(proposer_for('{"recipe": null}')) is None


def test_agent_proposer_receives_baseline_and_prior_measurements():
    seen = []

    def capture(messages):
        seen.extend(messages)
        return '{"recipe": null}'

    proposer = HarnessOptimizerAgentProposer(
        create_harness_optimizer_agent(chat_generator=MockChatGenerator(response_fn=capture))
    )
    history = [{"recipe": {"kind": "apply_patch", "patch": "reasoning-high"}, "status": "measured"}]
    assert propose_with(proposer, history=history) is None
    request = json.loads(next(message.text for message in seen if message.is_from("user")))
    assert request["baseline"]["cost"] == 10.0
    assert request["history"] == history
    assert request["approved_patches"][0]["changes"] == {"a.b": 1}
    assert request["successful_run_inputs"][0]["messages"][0]["text"] == "q"


def test_agent_proposer_configures_structured_output_and_recovers_formatting_slips():
    response = 'Here is the decision:\n```json\n{"recipe": {"kind": "model_substitution", "model_id": "cheap"}}\n```'
    proposer = proposer_for(response, structured_output_key="text")
    assert propose_with(proposer) == ModelSubstitutionRecipe(model_id="cheap")
    configured = proposer._structured_output(catalog())
    assert configured["text"]["format"]["schema"]["$defs"]["ModelSubstitutionRecipe"]


def test_agent_proposer_retries_invalid_decisions_then_gives_up():
    proposer = proposer_for(["not json", '{"recipe": {"kind": "model_substitution", "model_id": "cheap"}}'])
    assert propose_with(proposer) == ModelSubstitutionRecipe(model_id="cheap")
    with pytest.raises(ValueError, match="did not return a valid decision"):
        propose_with(proposer_for('{"recipe": {"kind": "python"}}'))
