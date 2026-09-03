import json

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator, OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Toolset, tool
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics
from haystack_integrations.agent_pack.optimization import (
    AgentMutation,
    HarnessOptimizerAgentProposer,
    ModelPrice,
    ModelPriceCatalog,
    MutationOperation,
    OptimizationObjectives,
    OptimizerDecision,
    create_harness_optimizer_agent,
    create_haystack_documentation_mcp_toolset,
)
from haystack_integrations.agent_pack.optimization.proposer import HARNESS_OPTIMIZER_SYSTEM_PROMPT


def reference_run():
    """Create one successful input/output example for optimizer context."""
    return AgentRunRecord(
        run_id="run",
        inputs={"messages": [ChatMessage.from_user("q")]},
        outputs={"last_message": ChatMessage.from_assistant("a")},
    )


def pricing():
    """Create optimizer price context that is deliberately not an allowlist."""
    return ModelPriceCatalog(prices=[ModelPrice(model_id="reference"), ModelPrice(model_id="cheap")])


def proposer_for(response):
    """Build a real optimizer Agent around a deterministic mock generator."""
    return HarnessOptimizerAgentProposer(
        optimizer_agent=create_harness_optimizer_agent(chat_generator=MockChatGenerator(response))
    )


def propose_with(proposer, history=None):
    """Call a proposer with complete minimal experiment context."""
    return proposer.propose(
        reference=Agent(chat_generator=MockChatGenerator(model="reference"), system_prompt="reference prompt"),
        reference_runs=[reference_run()],
        pricing=pricing(),
        objectives=OptimizationObjectives(),
        baseline=EvaluationMetrics(quality=1.0, cost=10.0, latency_ms=100),
        history=history or [],
    )


def test_system_prompt_grants_full_configuration_control_and_explains_mutations():
    """The optimizer is guided toward evidence-based arbitrary edits rather than named patches."""
    assert "complete serialized reference Agent configuration" in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    assert "may change any part" in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    assert "RFC 6901" in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    assert "rather than an allowlist" in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    assert "documentation tools" in HARNESS_OPTIMIZER_SYSTEM_PROMPT


def test_optimizer_agent_defaults_and_optional_docs_toolset(monkeypatch):
    """The factory keeps provider and optional documentation setup compact."""
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
    """The exposed public MCP integration contains only documentation search."""
    pytest.importorskip("haystack_integrations.tools.mcp", reason="mcp-haystack is optional")
    toolset = create_haystack_documentation_mcp_toolset()
    assert toolset.tool_names == ["search_haystack_docs"]
    assert toolset.eager_connect is False


def test_optimizer_decision_converts_to_a_provider_strict_schema():
    """The fixed decision model is valid provider-native structured output."""
    schema = to_strict_json_schema(OptimizerDecision)
    assert schema["additionalProperties"] is False
    operation_schema = schema["$defs"]["MutationOperation"]
    assert "oneOf" not in operation_schema
    assert operation_schema["properties"]["op"]["enum"] == [
        "set",
        "create_object",
        "create_array",
        "remove",
        "copy",
    ]


def test_agent_proposer_returns_one_typed_mutation_or_stops():
    """No manual JSON extraction sits between provider output and Pydantic validation."""
    response = json.dumps(
        {
            "mutation": {
                "operations": [
                    {
                        "op": "set",
                        "path": "/init_parameters/chat_generator/init_parameters/model",
                        "value": "any-model",
                    }
                ]
            }
        }
    )
    assert propose_with(proposer=proposer_for(response=response)) == AgentMutation(
        operations=(
            MutationOperation(
                op="set", path="/init_parameters/chat_generator/init_parameters/model", value="any-model"
            ),
        )
    )
    assert propose_with(proposer=proposer_for(response='{"mutation": null}')) is None


def test_agent_proposer_sends_full_configuration_runs_and_history():
    """The optimizer can reason from all editable state plus measured input/output outcomes."""
    seen = []

    def capture(messages):
        """Capture generator messages and return a stop decision."""
        seen.extend(messages)
        return '{"mutation": null}'

    proposer = HarnessOptimizerAgentProposer(
        optimizer_agent=create_harness_optimizer_agent(chat_generator=MockChatGenerator(response_fn=capture))
    )
    history = [{"mutation": {"operations": []}, "status": "failed"}]
    assert propose_with(proposer=proposer, history=history) is None
    request = json.loads(next(message.text for message in seen if message.is_from("user")))
    assert request["reference_agent_configuration"]["init_parameters"]["system_prompt"] == "reference prompt"
    assert request["baseline"]["cost"] == 10.0
    assert request["history"] == history
    assert request["successful_reference_runs"][0]["inputs"]["messages"][0]["text"] == "q"
    assert request["successful_reference_runs"][0]["outputs"]["last_message"]["text"] == "a"


def test_agent_proposer_always_passes_the_pydantic_text_format(monkeypatch):
    """Structured output is mandatory rather than an optional provider-specific switch."""
    proposer = proposer_for(response='{"mutation": null}')
    captured = {}
    original = proposer.optimizer_agent.run

    def spy(**kwargs):
        """Record Agent invocation arguments before delegating to the real implementation."""
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(proposer.optimizer_agent, "run", spy)
    assert propose_with(proposer=proposer) is None
    assert captured["generation_kwargs"] == {"text_format": OptimizerDecision}


def test_invalid_structured_text_fails_at_one_validation_boundary():
    """Malformed output is not recovered through brace scanning or ad-hoc retries."""
    with pytest.raises(ValidationError):
        propose_with(proposer=proposer_for(response="not json"))
