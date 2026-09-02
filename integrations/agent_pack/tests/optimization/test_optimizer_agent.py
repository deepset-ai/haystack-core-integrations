import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator, OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Toolset, tool

from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    HarnessOptimizerAgentProposer,
    HarnessPatch,
    ModelAsset,
    ModelSubstitutionRecipe,
    OptimizationObjectives,
    create_harness_optimizer_agent,
    create_haystack_documentation_mcp_toolset,
)
from haystack_integrations.agent_pack.optimization.prompts import HARNESS_OPTIMIZER_SYSTEM_PROMPT
from haystack_integrations.agent_pack.optimization.recipes import RECIPE_KINDS, proposal_json_schema
from haystack_integrations.agent_pack.tracing import (
    TraceArtifact,
)


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
                ModelAsset(model_id="reference"),
                ModelAsset(model_id="cheap"),
            ],
        ),
        objectives=objectives or OptimizationObjectives(),
    )


def test_system_prompt_documents_exactly_the_kinds_the_parser_accepts():
    """The prompt is the only place the Agent learns the recipe language, so a proposal is never spent on a kind
    the parser would reject."""
    for kind in RECIPE_KINDS:
        assert kind in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    for removed in ("specialist_delegation", "composite", "registered_structure"):
        assert removed not in HARNESS_OPTIMIZER_SYSTEM_PROMPT


def test_optimizer_agent_bakes_the_guidance_into_its_system_prompt():
    agent = create_harness_optimizer_agent(chat_generator=MockChatGenerator("[]"))
    assert agent.system_prompt == HARNESS_OPTIMIZER_SYSTEM_PROMPT
    # No tools are needed to know the recipe language.
    assert agent.tools == []
    assert agent.exit_conditions == ["text"]


def test_optimizer_agent_defaults_its_generator(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    agent = create_harness_optimizer_agent()
    assert isinstance(agent.chat_generator, OpenAIResponsesChatGenerator)
    # Proposing a change decides what a whole experiment measures, so the default is the most capable tier.
    assert agent.chat_generator.model == "gpt-5.6-sol"


def test_optimizer_agent_accepts_optional_read_only_docs_toolset():
    @tool
    def search_haystack_docs(query: str) -> str:
        """Search official Haystack documentation."""
        return query

    docs = Toolset([search_haystack_docs])
    agent = create_harness_optimizer_agent(chat_generator=MockChatGenerator("[]"), docs_toolset=docs)
    assert agent.tools == [docs]


def test_haystack_documentation_mcp_server_is_read_only_and_lazily_connected():
    pytest.importorskip("haystack_integrations.tools.mcp", reason="mcp-haystack is optional")
    toolset = create_haystack_documentation_mcp_toolset()
    assert toolset.tool_names == ["search_haystack_docs"]
    assert toolset.server_info.url == "https://docs.haystack.deepset.ai/api/mcp"
    assert toolset.eager_connect is False


def test_agent_proposer_only_accepts_typed_recipe_json():
    proposals = propose_with(proposer_for('{"recipes": [{"kind":"model_substitution","model_id":"cheap"}]}'))
    assert proposals == [ModelSubstitutionRecipe(model_id="cheap")]


def test_proposal_schema_closes_over_the_catalog():
    """The schema handed to a generator offers exactly the catalog's choices, and only the supported kinds."""
    catalog = ApprovedAssetCatalog(
        models=[
            ModelAsset(model_id="reference"),
            ModelAsset(model_id="cheap"),
        ],
        patches=[
            HarnessPatch(name="reasoning-high", patch={"a.b": 1}),
            HarnessPatch(name="retrieval-top-10", patch={"c.d": 10}),
        ],
    )
    schema = proposal_json_schema(assets=catalog)
    definitions = schema["$defs"]

    assert definitions["ModelSubstitutionProposal"]["properties"]["model_id"]["enum"] == ["cheap", "reference"]
    assert definitions["ApplyPatchProposal"]["properties"]["patch"]["enum"] == [
        "reasoning-high",
        "retrieval-top-10",
    ]
    kinds = {
        definition["properties"]["kind"]["const"]
        for name, definition in definitions.items()
        if name.endswith("Proposal")
    }
    assert kinds == set(RECIPE_KINDS)


def test_agent_proposer_reads_a_schema_shaped_response():
    response = '{"recipes": [{"kind": "model_substitution", "model_id": "cheap"}]}'
    assert propose_with(proposer_for(response)) == [ModelSubstitutionRecipe(model_id="cheap")]


def test_agent_proposer_configures_structured_output_from_the_catalog():
    proposer = proposer_for(
        '{"recipes": [{"kind": "model_substitution", "model_id": "cheap"}]}', structured_output_key="text"
    )
    catalog = ApprovedAssetCatalog(
        models=[
            ModelAsset(model_id="reference"),
            ModelAsset(model_id="cheap"),
        ],
    )
    configured = proposer._structured_output(assets=catalog)
    assert configured is not None
    schema = configured["text"]["format"]["schema"]
    assert schema["$defs"]["ModelSubstitutionProposal"]["properties"]["model_id"]["enum"] == ["cheap", "reference"]


def test_agent_proposer_tolerates_fenced_or_prefixed_json():
    """A code fence or a sentence of preamble is a formatting slip, not a reason to abort an experiment."""
    response = 'Here is my proposal:\n```json\n{"recipes": [{"kind": "model_substitution", "model_id": "cheap"}]}\n```'
    assert propose_with(proposer_for(response)) == [ModelSubstitutionRecipe(model_id="cheap")]


def test_agent_proposer_retries_once_with_corrective_feedback():
    proposer = proposer_for(["not json at all", '{"recipes": [{"kind":"model_substitution","model_id":"cheap"}]}'])
    assert propose_with(proposer) == [ModelSubstitutionRecipe(model_id="cheap")]


def test_agent_proposer_gives_up_after_max_attempts():
    with pytest.raises(ValueError, match="did not return a valid proposal"):
        propose_with(proposer_for("still not json"))


def test_agent_proposer_rejects_untyped_and_oversized_responses():
    with pytest.raises(ValueError, match="did not return a valid proposal"):
        propose_with(proposer_for('{"recipes": [{"kind": "python", "code": "dangerous()"}]}'))
    with pytest.raises(ValueError, match="did not return a valid proposal"):
        propose_with(
            proposer_for(
                '{"recipes": [{"kind":"model_substitution","model_id":"cheap"}, '
                '{"kind":"model_substitution","model_id":"reference"}]}',
                max_recipes=1,
            )
        )
