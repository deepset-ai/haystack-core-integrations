import pytest
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.hooks import hook
from haystack.tools import tool

from haystack_integrations.agent_pack.plugins import AgentPlugin, apply_plugins


@tool
def base_tool(value: str) -> str:
    """Return a value."""
    return value


@tool
def plugin_tool(value: str) -> str:
    """Return a plugin value."""
    return value


@hook
def base_hook(state: State) -> None:  # noqa: ARG001
    return None


@hook
def plugin_hook(state: State) -> None:  # noqa: ARG001
    return None


def _agent(**kwargs):
    return Agent(chat_generator=MockChatGenerator("done"), **kwargs)


def test_applies_contributions_in_order_without_mutating_input_agent():
    agent = _agent(
        tools=[base_tool],
        system_prompt="Base prompt",
        state_schema={"base": {"type": str}},
        hooks={"before_llm": [base_hook]},
    )
    plugin = AgentPlugin(
        name="example",
        tools=[plugin_tool],
        state_schema={"plugin_state": {"type": int}},
        hooks={"before_llm": [plugin_hook]},
        prompt_instructions="Use the plugin tool when needed.",
    )

    configured = apply_plugins(agent, [plugin])

    assert configured is not agent
    assert [item.name for item in configured.tools] == ["base_tool", "plugin_tool"]
    assert configured.state_schema == {"base": {"type": str}, "plugin_state": {"type": int}}
    assert configured.hooks["before_llm"] == [base_hook, plugin_hook]
    assert configured.system_prompt == "Base prompt\n\n## Plugin: example\nUse the plugin tool when needed."
    assert [item.name for item in agent.tools] == ["base_tool"]
    assert agent.state_schema == {"base": {"type": str}}
    assert agent.hooks == {"before_llm": [base_hook]}
    assert agent.system_prompt == "Base prompt"


def test_appends_prompt_instructions_inside_jinja_message_block():
    agent = _agent(system_prompt='{% message role="system" %}Base prompt{% endmessage %}')

    configured = apply_plugins(
        agent,
        [
            AgentPlugin(name="first", prompt_instructions="First instructions."),
            AgentPlugin(name="second", prompt_instructions="Second instructions."),
        ],
    )

    assert configured.system_prompt == (
        '{% message role="system" %}Base prompt\n\n'
        "## Plugin: first\nFirst instructions.\n\n"
        "## Plugin: second\nSecond instructions.\n"
        "{% endmessage %}"
    )
    result = configured.run(messages=[ChatMessage.from_user("hello")])
    assert result["messages"][0].text.endswith("Second instructions.")


def test_deduplicates_identical_state_definitions():
    configured = apply_plugins(
        _agent(state_schema={"user_id": {"type": str}}),
        [
            AgentPlugin(name="one", state_schema={"user_id": {"type": str}}),
            AgentPlugin(name="two", state_schema={"user_id": {"type": str}}),
        ],
    )

    assert configured.state_schema == {"user_id": {"type": str}}


@pytest.mark.parametrize(
    ("plugins", "match"),
    [
        ([AgentPlugin(name="")], "must not be empty"),
        ([AgentPlugin(name="same"), AgentPlugin(name="same")], "Duplicate AgentPlugin name"),
        ([AgentPlugin(name="duplicate", tools=[base_tool])], "duplicate tool name 'base_tool'"),
        ([AgentPlugin(name="state", state_schema={"value": {"type": int}})], "incompatible definition"),
        (
            [AgentPlugin(name="prompt", prompt_instructions='{% message role="system" %}bad{% endmessage %}')],
            "must not contain Jinja message blocks",
        ),
    ],
)
def test_rejects_conflicting_plugins_atomically(plugins, match):
    agent = _agent(tools=[base_tool], system_prompt="original", state_schema={"value": {"type": str}})

    with pytest.raises(ValueError, match=match):
        apply_plugins(agent, plugins)

    assert [item.name for item in agent.tools] == ["base_tool"]
    assert agent.system_prompt == "original"
    assert agent.state_schema == {"value": {"type": str}}


def test_rejects_duplicate_tools_already_on_agent():
    agent = _agent(tools=[base_tool, base_tool])

    with pytest.raises(ValueError, match="Agent already contains duplicate known tool name 'base_tool'"):
        apply_plugins(agent, [])


def test_serialization_roundtrip_of_configured_agent():
    configured = apply_plugins(
        _agent(system_prompt="Base"),
        [AgentPlugin(name="state", state_schema={"user_id": {"type": str}}, prompt_instructions="Remember users.")],
    )

    restored = Agent.from_dict(configured.to_dict())

    assert restored.state_schema == {"user_id": {"type": str}}
    assert restored.system_prompt == "Base\n\n## Plugin: state\nRemember users."
