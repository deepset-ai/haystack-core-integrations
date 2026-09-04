import json

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator, OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Toolset, tool
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord, EvaluationMetrics
from haystack_integrations.agent_pack.optimization import (
    AgentMutation,
    ModelPrice,
    ModelPriceCatalog,
    MutationOperation,
    OptimizationObjectives,
    OptimizerDecision,
    create_harness_optimizer_agent,
    create_haystack_documentation_mcp_toolset,
    propose_mutation,
)
from haystack_integrations.agent_pack.optimization.agent import (
    HARNESS_OPTIMIZER_SYSTEM_PROMPT,
    OPTIMIZER_PROMPT_CACHE_KEY,
)


@tool
def search_documents(query: str) -> str:
    """Retrieve documents matching a query."""
    return query


def reference_run():
    """Create one successful input/output example carrying the tool behavior an optimizer reasons from."""
    call = ToolCall(tool_name="search_documents", arguments={"query": "q", "filters": {"field": "meta.year"}}, id="c1")
    messages = [
        ChatMessage.from_user("q"),
        ChatMessage.from_assistant(tool_calls=[call]),
        ChatMessage.from_tool("one document", origin=call),
        ChatMessage.from_assistant("a"),
    ]
    return AgentRunRecord(
        run_id="run",
        inputs={"messages": [ChatMessage.from_user("q")]},
        outputs={
            "messages": messages,
            "last_message": messages[-1],
            "exit_reason": "text",
            "step_count": 2,
            "tool_call_counts": {"search_documents": 1, "fetch_documents_by_filter": 0},
        },
    )


def pricing():
    """Create optimizer price context that is deliberately not an allowlist."""
    return ModelPriceCatalog(prices=[ModelPrice(model_id="reference"), ModelPrice(model_id="cheap")])


def optimizer_agent_for(response):
    """Build an optimizer Agent around a deterministic mock generator."""
    return create_harness_optimizer_agent(chat_generator=MockChatGenerator(response))


def propose_with(optimizer_agent, history=None):
    """Request a mutation with complete minimal experiment context."""
    return propose_mutation(
        optimizer_agent=optimizer_agent,
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
    assert "outcome is attributable" in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    assert "vary one thing at a time" in HARNESS_OPTIMIZER_SYSTEM_PROMPT
    assert "removing one withdraws the tool entirely" in HARNESS_OPTIMIZER_SYSTEM_PROMPT


def test_optimizer_agent_defaults_and_optional_docs_toolset(monkeypatch):
    """The factory keeps provider and optional documentation setup compact."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    default = create_harness_optimizer_agent()
    assert isinstance(default.chat_generator, OpenAIResponsesChatGenerator)
    assert default.chat_generator.model == "gpt-5.6-terra"
    assert default.system_prompt == HARNESS_OPTIMIZER_SYSTEM_PROMPT

    @tool
    def search_haystack_docs(query: str) -> str:
        """Search official Haystack documentation."""
        return query

    docs = Toolset([search_haystack_docs])
    with_docs = create_harness_optimizer_agent(chat_generator=MockChatGenerator("{}"), docs_toolset=docs)
    assert with_docs.tools == [docs]


def test_domain_guidance_extends_rather_than_replaces_the_optimizer_instructions():
    """A harness can teach the optimizer about its own Agent without rewriting the mutation instructions."""
    guided = create_harness_optimizer_agent(
        chat_generator=MockChatGenerator("{}"), additional_instructions="  Keep top_k above the case minimum.  "
    )
    assert guided.system_prompt is not None
    assert guided.system_prompt.startswith(HARNESS_OPTIMIZER_SYSTEM_PROMPT)
    assert guided.system_prompt.endswith("\n\nKeep top_k above the case minimum.")

    replaced = create_harness_optimizer_agent(
        chat_generator=MockChatGenerator("{}"),
        system_prompt="Only these rules apply.",
        additional_instructions="Keep top_k above the case minimum.",
    )
    assert replaced.system_prompt == "Only these rules apply.\n\nKeep top_k above the case minimum."


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


def test_propose_mutation_returns_one_typed_mutation_or_stops():
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
    assert propose_with(optimizer_agent=optimizer_agent_for(response=response)) == AgentMutation(
        operations=(
            MutationOperation(
                op="set", path="/init_parameters/chat_generator/init_parameters/model", value="any-model"
            ),
        )
    )
    assert propose_with(optimizer_agent=optimizer_agent_for(response='{"mutation": null}')) is None


def test_propose_mutation_sends_full_configuration_runs_and_history():
    """The optimizer can reason from all editable state plus measured input/output outcomes."""
    seen = []

    def capture(messages):
        """Capture generator messages and return a stop decision."""
        seen.extend(messages)
        return '{"mutation": null}'

    optimizer_agent = create_harness_optimizer_agent(chat_generator=MockChatGenerator(response_fn=capture))
    history = [{"mutation": {"operations": []}, "status": "failed"}]
    assert propose_with(optimizer_agent=optimizer_agent, history=history) is None
    sent = [json.loads(message.text) for message in seen if message.is_from("user")]
    # Three messages ordered by how often each changes, so a cache breakpoint can sit between them and everything
    # ahead of the churn stays identical from turn to turn.
    context, record, detail = sent
    assert "history" not in context
    # The record carries every outcome without its tool traces, and never revises an entry once written.
    assert record == {"outcomes": history}
    assert detail == {"recent_outcomes_in_detail": history}
    request = context
    assert request["reference_agent_configuration"]["init_parameters"]["system_prompt"] == "reference prompt"
    assert request["baseline"]["cost"] == 10.0
    assert request["successful_reference_runs"][0]["inputs"]["messages"][0]["content"] == [{"text": "q"}]

    digest = request["successful_reference_runs"][0]["outputs"]
    assert digest["exit_reason"] == "text"
    assert digest["tool_call_counts"] == {"search_documents": 1, "fetch_documents_by_filter": 0}
    assert digest["tool_steps"] == [
        {
            "tool": "search_documents",
            "arguments": '{"query": "q", "filters": {"field": "meta.year"}}',
            "result": "one document",
            "error": False,
        }
    ]
    assert digest["answer_excerpt"] == "a"
    # The verbatim transcript is not sent: provider response metadata dominates it, and `last_message` only
    # repeats the final message.
    assert "messages" not in digest
    assert "last_message" not in digest


def test_available_tools_names_what_the_serialized_configuration_cannot():
    """A ComponentTool serializes a null parameter schema, so the request has to carry the specs separately."""
    seen = []

    def capture(messages):
        """Capture generator messages and return a stop decision."""
        seen.extend(messages)
        return '{"mutation": null}'

    optimizer_agent = create_harness_optimizer_agent(chat_generator=MockChatGenerator(response_fn=capture))
    reference = Agent(chat_generator=MockChatGenerator(model="reference"), tools=[search_documents])
    assert (
        propose_mutation(
            optimizer_agent=optimizer_agent,
            reference=reference,
            reference_runs=[reference_run()],
            pricing=pricing(),
            objectives=OptimizationObjectives(),
            baseline=EvaluationMetrics(quality=1.0, cost=10.0, latency_ms=100),
            history=[],
        )
        is None
    )
    request = json.loads(next(message.text for message in seen if message.is_from("user")))
    assert request["available_tools"] == [
        {
            "name": "search_documents",
            "description": "Retrieve documents matching a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]


def test_the_default_optimizer_carries_a_stable_cache_routing_key(monkeypatch):
    """Reuse of the prefix depends on requests for it being routed together."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    default = create_harness_optimizer_agent()

    assert default.chat_generator.generation_kwargs["prompt_cache_key"] == OPTIMIZER_PROMPT_CACHE_KEY
    # Reasoning is billed as output on the most expensive model in the experiment, so the effort behind one
    # decision per turn is chosen here rather than left to the provider's heavier default.
    assert default.chat_generator.generation_kwargs["reasoning"] == {"effort": "low"}

    # A caller-supplied generator is left alone: the key is provider-specific, and a generator that has no such
    # setting must not acquire one.
    supplied = create_harness_optimizer_agent(chat_generator=MockChatGenerator("{}"))
    assert not hasattr(supplied.chat_generator, "generation_kwargs")


def test_a_reference_whose_tools_cannot_be_read_still_produces_a_proposal(monkeypatch):
    """Losing the tool specs degrades the evidence; it must not end the experiment."""
    seen = []

    def capture(messages):
        """Capture generator messages and return a stop decision."""
        seen.extend(messages)
        return '{"mutation": null}'

    def explode(tools):  # noqa: ARG001
        """Fail the way a lazily-loaded toolset that cannot connect would."""
        msg = "toolset unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr("haystack_integrations.agent_pack.optimization.agent.warm_up_tools", explode)
    optimizer_agent = create_harness_optimizer_agent(chat_generator=MockChatGenerator(response_fn=capture))
    assert propose_with(optimizer_agent=optimizer_agent) is None
    request = json.loads(next(message.text for message in seen if message.is_from("user")))
    assert request["available_tools"] == []


def test_propose_mutation_always_passes_the_pydantic_text_format(monkeypatch):
    """Structured output is mandatory rather than an optional provider-specific switch."""
    optimizer_agent = optimizer_agent_for(response='{"mutation": null}')
    captured = {}
    original = optimizer_agent.run

    def spy(**kwargs):
        """Record Agent invocation arguments before delegating to the real implementation."""
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(optimizer_agent, "run", spy)
    assert propose_with(optimizer_agent=optimizer_agent) is None
    assert captured["generation_kwargs"] == {"text_format": OptimizerDecision}


def test_invalid_structured_text_fails_at_one_validation_boundary():
    """Malformed output is not recovered through brace scanning or ad-hoc retries."""
    with pytest.raises(ValidationError):
        propose_with(optimizer_agent=optimizer_agent_for(response="not json"))
