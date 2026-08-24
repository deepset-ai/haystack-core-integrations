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
    PolicyEvaluation,
    StaticPolicyProvider,
    ToolAsset,
    TraceCapturingAgentRunner,
    span_tag,
)
from haystack_integrations.agent_pack.optimization.policy.model_identity import generator_model_id

MOCK_GENERATOR_TYPE = "haystack.components.generators.chat.mock.MockChatGenerator"


def modelled_generator(model, responses=None):
    return MockChatGenerator(responses or [ChatMessage.from_assistant("done")], model=model)


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
        provider=StaticPolicyProvider(allowed_tools=["guarded_tool"] if allowed else [], policy_version="test/v1")
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
    assert decisions[0]["tool_name"] == "guarded_tool"
    if not allowed:
        assert "tool_not_allowed" in str(result["messages"])
        assert "payload" not in decisions[0]


def test_policy_decisions_are_recorded_without_a_pre_seeded_list():
    """Governance capture must not depend on the caller seeding the context first."""
    context = {}
    PolicyEnforcementStrategy(provider=StaticPolicyProvider(allowed_tools=["known"])).run(
        tool_name="known", tool_description="d", tool_params={}, confirmation_strategy_context=context
    )
    assert [record["reason_code"] for record in context[POLICY_DECISIONS_CONTEXT_KEY]] == ["tool_allowed"]


@pytest.mark.asyncio
async def test_policy_strategy_async_path_delegates_to_same_logic():
    strategy = PolicyEnforcementStrategy(provider=StaticPolicyProvider(allowed_tools=[]))
    decision = await strategy.run_async(
        tool_name="unknown", tool_description="unknown", tool_params={"secret": "not-recorded"}, tool_call_id="1"
    )
    assert decision.execute is False
    assert decision.feedback == "Tool call rejected by policy (tool_not_allowed)."
    assert decision.final_tool_params is None


def test_policy_strategy_fails_closed_on_provider_error_and_roundtrips():
    class BrokenProvider:
        def evaluate(self, **_kwargs):
            message = "backend unavailable"
            raise RuntimeError(message)

    decision = PolicyEnforcementStrategy(provider=BrokenProvider()).run(
        tool_name="tool", tool_description="d", tool_params={}, tool_call_id="1"
    )
    assert decision.execute is False
    assert decision.feedback == "Tool call rejected by policy (policy_provider_error)."

    original = PolicyEnforcementStrategy(provider=StaticPolicyProvider(allowed_tools=["tool"], policy_version="v2"))
    restored = PolicyEnforcementStrategy.from_dict(original.to_dict())
    assert restored.run(tool_name="tool", tool_description="d", tool_params={}).execute is True


def test_indeterminate_provider_decisions_fail_closed():
    class VagueProvider:
        def evaluate(self, **_kwargs):
            return PolicyEvaluation(
                decision="indeterminate", policy_version="v1", rule_id="unclear", reason_code="unclear"
            )

    decision = PolicyEnforcementStrategy(provider=VagueProvider()).run(
        tool_name="tool", tool_description="d", tool_params={}
    )
    assert decision.execute is False
    assert decision.feedback == "Tool call rejected by policy (indeterminate_decision)."


def test_existing_hitl_strategy_trace_contains_policy_decision():
    @tool
    def guarded_tool() -> str:
        """Perform a guarded action."""
        return "done"

    strategy = PolicyEnforcementStrategy(provider=StaticPolicyProvider(allowed_tools=[]))
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

    captured = TraceCapturingAgentRunner(collector=LocalTraceCollector()).run(
        agent, messages=[ChatMessage.from_user("go")]
    )
    strategy_spans = [
        span
        for span in captured.trace.traces
        if span["operation_name"] == "haystack.agent.hook.human_in_the_loop.strategy"
    ]
    decision_tag = "haystack.agent.hook.human_in_the_loop.strategy.decision"
    assert span_tag(span=strategy_spans[0], key=decision_tag) == "reject"
    assert span_tag(span=strategy_spans[0], key="haystack.tool.name") == "guarded_tool"


def test_asset_catalog_validates_nested_agent_tools():
    specialist = Agent(chat_generator=modelled_generator("specialist"), tools=[nested_search_documents])
    coordinator = Agent(
        chat_generator=modelled_generator("reference"),
        tools=[AgentTool(agent=specialist, name="retrieval_specialist", description="Retrieve evidence")],
    )
    catalog = ApprovedAssetCatalog(
        models=[
            ModelAsset(model_id="reference", provider="provider", deployment="remote"),
            ModelAsset(model_id="specialist", provider="provider", deployment="local"),
        ],
        tools=[ToolAsset(name="retrieval_specialist"), ToolAsset(name="nested_search_documents")],
    )

    validation = catalog.require_valid_agent(coordinator)
    assert validation.model_ids == ("reference", "specialist")
    assert validation.tool_names == ("nested_search_documents", "retrieval_specialist")

    invalid = ApprovedAssetCatalog(
        models=[ModelAsset(model_id="reference", provider="provider", deployment="remote")],
        tools=[ToolAsset(name="retrieval_specialist")],
    ).validate_agent(coordinator)
    assert invalid.allowed is False
    assert "model_not_approved:specialist" in invalid.violations
    assert "tool_not_approved:nested_search_documents" in invalid.violations


def test_asset_catalog_validates_models_configured_on_hooks():
    agent = Agent(
        chat_generator=modelled_generator("reference"),
        hooks={"after_run": [BackupAnswerHook(chat_generator=modelled_generator("backup"))]},
    )

    validation = ApprovedAssetCatalog(
        models=[ModelAsset(model_id="reference", provider="provider", deployment="remote")],
        tools=[],
    ).validate_agent(agent)

    assert validation.allowed is False
    assert validation.model_ids == ("backup", "reference")
    assert "model_not_approved:backup" in validation.violations


class UnnamedGenerator:
    """A chat generator that exposes no model identifier at all."""

    def run(self, messages, tools=None, **kwargs):  # noqa: ARG002 - signature only, the reply is fixed
        return {"replies": []}

    def to_dict(self):
        return {"type": "tests.UnnamedGenerator", "init_parameters": {}}


def test_unidentifiable_assets_fail_closed_but_can_be_downgraded_to_warnings():
    agent = Agent(chat_generator=UnnamedGenerator())
    catalog_arguments = {
        "models": [ModelAsset(model_id="reference", provider="provider", deployment="remote")],
        "tools": [],
    }

    strict = ApprovedAssetCatalog(**catalog_arguments).validate_agent(agent)
    assert strict.allowed is False
    assert strict.violations == ("model_not_identifiable",)

    lenient = ApprovedAssetCatalog(**catalog_arguments, strict_identification=False).validate_agent(agent)
    assert lenient.allowed is True
    assert lenient.warnings == ("model_not_identifiable",)


class AzureStyleGenerator:
    def __init__(self):
        self.azure_deployment = "eu-gpt-deployment"


class HuggingFaceStyleGenerator:
    def __init__(self):
        self.api_params = {"model": "local/llama"}


@pytest.mark.parametrize(
    "generator,expected",
    [
        (MockChatGenerator(model="plain"), "plain"),
        (AzureStyleGenerator(), "eu-gpt-deployment"),
        (HuggingFaceStyleGenerator(), "local/llama"),
        (UnnamedGenerator(), None),
    ],
)
def test_model_identity_is_resolved_across_generator_conventions(generator, expected):
    """Azure deployments and Hugging Face API generators must not be rejected as unidentifiable."""
    assert generator_model_id(generator) == expected


def test_model_asset_substitutes_the_model_on_the_reference_generator_class():
    asset = ModelAsset(model_id="cheaper", provider="provider", deployment="local")
    built = asset.build_generator(MockChatGenerator([ChatMessage.from_assistant("hi")], model="reference"))
    assert built.model == "cheaper"
    assert isinstance(built, MockChatGenerator)


def test_model_asset_can_declare_its_own_generator_for_cross_provider_substitution():
    asset = ModelAsset(
        model_id="declared",
        provider="other-provider",
        deployment="eu",
        generator={"type": MOCK_GENERATOR_TYPE, "init_parameters": {"model": "declared"}},
    )
    built = asset.build_generator(UnnamedGenerator())
    assert isinstance(built, MockChatGenerator)
    assert built.model == "declared"


def test_declared_generator_must_agree_with_the_catalog_identifier():
    asset = ModelAsset(
        model_id="declared",
        provider="other-provider",
        deployment="eu",
        generator={"type": MOCK_GENERATOR_TYPE, "init_parameters": {"model": "something-else"}},
    )
    with pytest.raises(ValueError, match="must match the generator"):
        asset.build_generator(MockChatGenerator(model="reference"))


def test_substitution_requires_a_declared_generator_when_the_reference_hides_its_model():
    with pytest.raises(ValueError, match="needs an explicit 'generator' configuration"):
        ModelAsset(model_id="cheaper", provider="provider", deployment="local").build_generator(UnnamedGenerator())
