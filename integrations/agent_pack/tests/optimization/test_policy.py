import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.hooks.human_in_the_loop import ConfirmationHook
from haystack.tools import AgentTool, tool

from haystack_integrations.agent_pack.advanced_rag.hooks import BackupAnswerHook
from haystack_integrations.agent_pack.optimization import (
    POLICY_DECISIONS_CONTEXT_KEY,
    ApprovedAssetCatalog,
    LocalTraceCollector,
    ModelAsset,
    PolicyEnforcementStrategy,
    StaticPolicyProvider,
    ToolAsset,
    TraceCapturingAgentRunner,
)


def modelled_generator(model, responses=None):
    generator = MockChatGenerator(responses or [ChatMessage.from_assistant("done")])
    generator.model = model
    return generator


@tool
def nested_search_documents(query: str) -> str:
    """Search documents from a delegated Agent."""
    return query


@pytest.mark.parametrize("allowed,expected_calls", [(True, ["payload"]), (False, [])])
def test_policy_strategy_runs_programmatically_through_confirmation_hook(allowed, expected_calls):
    calls = []

    @tool
    def guarded_tool(value: str) -> str:
        """Perform a guarded action."""
        calls.append(value)
        return value

    responses = [
        ChatMessage.from_assistant(tool_calls=[ToolCall("guarded_tool", {"value": "payload"}, id="call")]),
        ChatMessage.from_assistant("done"),
    ]
    strategy = PolicyEnforcementStrategy(
        StaticPolicyProvider(allowed_tools=["guarded_tool"] if allowed else [], policy_version="test/v1")
    )
    agent = Agent(
        chat_generator=MockChatGenerator(responses),
        tools=[guarded_tool],
        hooks={"before_tool": [ConfirmationHook(confirmation_strategies={"*": strategy})]},
    )
    decisions = []

    result = agent.run(messages=[ChatMessage.from_user("go")], hook_context={POLICY_DECISIONS_CONTEXT_KEY: decisions})

    assert result["last_message"].text == "done"
    assert calls == expected_calls
    assert decisions[0]["decision"] == ("allow" if allowed else "deny")
    assert decisions[0]["policy_version"] == "test/v1"
    if not allowed:
        assert "tool_not_allowed" in str(result["messages"])
        assert "payload" not in decisions[0]


@pytest.mark.asyncio
async def test_policy_strategy_async_path_delegates_to_same_logic():
    strategy = PolicyEnforcementStrategy(StaticPolicyProvider(allowed_tools=[]))
    decision = await strategy.run_async(
        tool_name="unknown", tool_description="unknown", tool_params={"secret": "not-recorded"}, tool_call_id="1"
    )
    assert decision.execute is False
    assert decision.feedback == "Tool call rejected by policy (tool_not_allowed)."


def test_policy_strategy_fails_closed_on_provider_error_and_roundtrips():
    class BrokenProvider:
        def evaluate(self, **_kwargs):
            message = "backend unavailable"
            raise RuntimeError(message)

    decision = PolicyEnforcementStrategy(BrokenProvider()).run(
        tool_name="tool", tool_description="d", tool_params={}, tool_call_id="1"
    )
    assert decision.execute is False
    assert decision.feedback == "Tool call rejected by policy (policy_provider_error)."

    original = PolicyEnforcementStrategy(StaticPolicyProvider(allowed_tools=["tool"], policy_version="v2"))
    restored = PolicyEnforcementStrategy.from_dict(original.to_dict())
    assert restored.run(tool_name="tool", tool_description="d", tool_params={}).execute is True


def test_existing_hitl_strategy_trace_contains_policy_decision():
    @tool
    def guarded_tool() -> str:
        """Perform a guarded action."""
        return "done"

    strategy = PolicyEnforcementStrategy(StaticPolicyProvider(allowed_tools=[]))
    agent = Agent(
        chat_generator=MockChatGenerator(
            [
                ChatMessage.from_assistant(tool_calls=[ToolCall("guarded_tool", {}, id="call")]),
                ChatMessage.from_assistant("done"),
            ]
        ),
        tools=[guarded_tool],
        hooks={"before_tool": [ConfirmationHook(confirmation_strategies={"*": strategy})]},
    )

    captured = TraceCapturingAgentRunner(LocalTraceCollector()).run(agent, messages=[ChatMessage.from_user("go")])
    strategy_spans = [
        span
        for span in captured.trace.traces
        if span["operation_name"] == "haystack.agent.hook.human_in_the_loop.strategy"
    ]
    assert strategy_spans[0]["tags"]["haystack.agent.hook.human_in_the_loop.strategy.decision"] == "reject"
    assert strategy_spans[0]["tags"]["haystack.tool.name"] == "guarded_tool"


def test_asset_catalog_validates_nested_agent_tools():
    specialist = Agent(chat_generator=modelled_generator("sovereign"), tools=[nested_search_documents])
    coordinator = Agent(
        chat_generator=modelled_generator("reference"),
        tools=[AgentTool(agent=specialist, name="retrieval_specialist", description="Retrieve evidence")],
    )
    catalog = ApprovedAssetCatalog(
        models=[
            ModelAsset("reference", "provider", "remote"),
            ModelAsset("sovereign", "provider", "local", sovereign=True),
        ],
        tools=[ToolAsset("retrieval_specialist"), ToolAsset("nested_search_documents")],
    )

    validation = catalog.require_valid_agent(coordinator)
    assert validation.model_ids == ("reference", "sovereign")
    assert validation.tool_names == ("nested_search_documents", "retrieval_specialist")

    invalid = ApprovedAssetCatalog(
        models=[ModelAsset("reference", "provider", "remote")],
        tools=[ToolAsset("retrieval_specialist")],
    ).validate_agent(coordinator)
    assert invalid.allowed is False
    assert "model_not_approved:sovereign" in invalid.reason_codes
    assert "tool_not_approved:nested_search_documents" in invalid.reason_codes


def test_asset_catalog_validates_models_configured_on_hooks():
    agent = Agent(
        chat_generator=modelled_generator("reference"),
        hooks={"after_run": [BackupAnswerHook(chat_generator=modelled_generator("backup"))]},
    )

    validation = ApprovedAssetCatalog(
        models=[ModelAsset("reference", "provider", "remote")],
        tools=[],
    ).validate_agent(agent)

    assert validation.allowed is False
    assert validation.model_ids == ("backup", "reference")
    assert "model_not_approved:backup" in validation.reason_codes
