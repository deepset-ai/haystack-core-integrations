import pytest
from haystack.components.agents.state import State
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import MockChatGenerator, OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.agent_pack.deep_research.hooks import ScopeHook, WriteHook

SCOPE_SCHEMA = {"brief": {"type": str}}
WRITE_SCHEMA = {"notes": {"type": list}, "brief": {"type": str}, "report": {"type": str}}


def _echo_prompt(messages):
    return messages[-1].text


def _subtopic_message(*subtopics):
    tool_calls = [
        ToolCall(tool_name="research_subtopic", arguments={"messages": [{"role": "user", "content": s}]})
        for s in subtopics
    ]
    return ChatMessage.from_assistant(tool_calls=tool_calls)


def test_scope_hook_sets_brief_and_replaces_messages():
    hook = ScopeHook(
        generator=MockChatGenerator("a focused brief"),
        prompt_builder=ChatPromptBuilder(template='{% message role="user" %}{{ query }}{% endmessage %}'),
    )
    state = State(schema=SCOPE_SCHEMA)
    state.set("messages", [ChatMessage.from_user("original question")])

    hook.run(state)

    assert state.get("brief") == "a focused brief"
    assert state.get("messages")[-1].text == "a focused brief"


def test_write_hook_writes_report_from_brief_and_notes():
    hook = WriteHook(
        generator=MockChatGenerator(response_fn=_echo_prompt),
        prompt_builder=ChatPromptBuilder(
            template='{% message role="user" %}{{ replies[0].text }} {{ notes|join(" ") }}{% endmessage %}'
        ),
    )
    state = State(schema=WRITE_SCHEMA)
    state.set("brief", "the brief")
    state.set("notes", ["note one", "note two"])
    state.set("messages", [_subtopic_message("subtopic a", "subtopic b")])

    hook.run(state)

    report = state.get("report")
    assert "the brief" in report
    assert "note one" in report and "note two" in report
    assert state.get("messages")[-1].text == report


def test_write_hook_drops_empty_notes():
    hook = WriteHook(
        generator=MockChatGenerator(response_fn=_echo_prompt),
        prompt_builder=ChatPromptBuilder(template='{% message role="user" %}{{ notes|length }}{% endmessage %}'),
    )
    state = State(schema=WRITE_SCHEMA)
    state.set("notes", ["kept", "", None])
    state.set("messages", [])

    hook.run(state)

    assert state.get("report") == "1"


@pytest.mark.parametrize("hook_cls", [ScopeHook, WriteHook])
def test_hook_serialization_roundtrip(hook_cls):
    generator = OpenAIChatGenerator(model="gpt-5.4")
    hook = hook_cls(generator=generator, prompt_builder=ChatPromptBuilder(template="{{ x }}"))
    data = hook.to_dict()
    assert data["type"].endswith(hook_cls.__name__)

    restored = hook_cls.from_dict(data)
    assert isinstance(restored, hook_cls)
    assert restored.generator.model == "gpt-5.4"
    assert restored.prompt_builder.template == "{{ x }}"
