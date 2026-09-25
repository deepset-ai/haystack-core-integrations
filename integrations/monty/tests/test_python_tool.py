# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from haystack import component
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from haystack_integrations.tools.monty import MontyPythonTool
from haystack_integrations.tools.monty.python_tool import _DEFAULT_DESCRIPTION


@pytest.fixture
def tool():
    tool = MontyPythonTool()
    yield tool
    tool.close()


@component
class _ScriptedChatGenerator:
    """Asks for one `run_python` call, then answers with the tool result it received."""

    def __init__(self, code: str) -> None:
        self.code = code

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None) -> dict[str, Any]:  # noqa: ARG002
        tool_result = messages[-1].tool_call_result
        if tool_result is not None:
            return {"replies": [ChatMessage.from_assistant(f"The sandbox said: {tool_result.result}")]}
        tool_call = ToolCall(tool_name="run_python", arguments={"code": self.code})
        return {"replies": [ChatMessage.from_assistant(tool_calls=[tool_call])]}


class TestInit:
    def test_defaults(self):
        tool = MontyPythonTool()

        assert isinstance(tool, Tool)
        assert tool.name == "run_python"
        assert tool.description == _DEFAULT_DESCRIPTION
        assert "fresh interpreter" in tool.description
        assert "no file system, network" in tool.description
        assert tool.parameters["properties"]["code"]["type"] == "string"
        assert tool.parameters["required"] == ["code"]
        assert tool._resource_limits == {"max_feed_duration_secs": 30.0, "max_memory": 256 * 1024 * 1024}

    def test_custom_name_and_description(self):
        tool = MontyPythonTool(name="python", description="Run some Python.")

        assert tool.name == "python"
        assert tool.description == "Run some Python."

    def test_resource_limits_merged_over_defaults(self):
        tool = MontyPythonTool(resource_limits={"max_memory": None, "max_recursion_depth": 200})

        assert tool._resource_limits == {
            "max_feed_duration_secs": 30.0,
            "max_memory": None,
            "max_recursion_depth": 200,
        }

    def test_unknown_resource_limit_raises(self):
        with pytest.raises(ValueError, match="Unknown resource limits: \\['max_duration_secs'\\]"):
            MontyPythonTool(resource_limits={"max_duration_secs": 5.0})

    def test_invalid_max_output_chars_raises(self):
        with pytest.raises(ValueError, match="max_output_chars must be at least 1"):
            MontyPythonTool(max_output_chars=0)


class TestInvoke:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("1 + 2", "result:\n3"),
            ("print('hello')", "output:\nhello"),
            ("x = [i * i for i in range(4)]\nprint(len(x))\nx", "output:\n4\n\nresult:\n[0, 1, 4, 9]"),
            ("'text'", "result:\n'text'"),
            ("x = 1", "The code ran successfully and produced no output."),
        ],
    )
    def test_success(self, tool, code, expected):
        assert tool.invoke(code=code) == expected

    def test_syntax_error(self, tool):
        output = tool.invoke(code="def")

        assert output.startswith("error:\nTraceback (most recent call last):")
        assert "SyntaxError" in output

    def test_runtime_error_keeps_output_before_it(self, tool):
        output = tool.invoke(code="print('before')\n1 / 0")

        assert output.startswith("output:\nbefore\n\nerror:\nTraceback (most recent call last):")
        assert output.endswith("ZeroDivisionError: division by zero")

    @pytest.mark.parametrize(
        ("code", "expected_error"),
        [
            ("import socket", "ModuleNotFoundError: No module named 'socket'"),
            ("open('/etc/passwd').read()", "PermissionError"),
            ("import os\nos.getenv('HOME')", "'os.getenv' is not supported"),
        ],
    )
    def test_no_host_access(self, tool, code, expected_error):
        assert expected_error in tool.invoke(code=code)

    def test_time_limit(self):
        tool = MontyPythonTool(resource_limits={"max_feed_duration_secs": 0.5})

        output = tool.invoke(code="while True:\n    pass")
        tool.close()

        assert output.startswith("error:\nTimeoutError: feed time limit exceeded")

    def test_time_limit_inside_builtin_stops_worker(self):
        # The sandbox can't interrupt a huge integer power, so the pool kills and replaces the worker instead
        tool = MontyPythonTool(resource_limits={"max_feed_duration_secs": 0.2})

        output = tool.invoke(code="7 ** 50_000_000")
        follow_up = tool.invoke(code="1 + 1")
        tool.close()

        assert output.startswith("error:\nTimeoutError: the code exceeded the time limit and the sandbox was stopped")
        assert follow_up == "result:\n2"

    def test_memory_limit(self):
        tool = MontyPythonTool(resource_limits={"max_memory": 16 * 1024 * 1024})

        output = tool.invoke(code="'a' * 32 * 1024 * 1024")
        tool.close()

        assert output.startswith("error:\nTraceback (most recent call last):")
        assert "MemoryError: memory limit exceeded" in output

    def test_type_check(self):
        tool = MontyPythonTool(type_check=True)

        output = tool.invoke(code="x: int = 'not an int'")
        tool.close()

        assert output.startswith("error:\n")
        assert "invalid-assignment" in output

    def test_no_state_between_calls(self, tool):
        tool.invoke(code="leaked = 1")

        assert "NameError: name 'leaked' is not defined" in tool.invoke(code="leaked")

    def test_output_truncated(self):
        tool = MontyPythonTool(max_output_chars=10)

        output = tool.invoke(code="print('a' * 25)\n'b' * 25")
        tool.close()

        assert output == (
            f"output:\n{'a' * 10}\n[... 15 characters truncated]\n\nresult:\n'{'b' * 9}\n[... 17 characters truncated]"
        )

    def test_concurrent_calls(self, tool):
        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(executor.map(lambda n: tool.invoke(code=f"sum(range({n}))"), range(10, 18)))

        assert outputs == [f"result:\n{sum(range(n))}" for n in range(10, 18)]

    @pytest.mark.asyncio
    async def test_invoke_async(self, tool):
        assert await tool.invoke_async(code="2 ** 10") == "result:\n1024"


class TestLifecycle:
    def test_warm_up_is_idempotent(self, tool):
        tool.warm_up()
        pool = tool._pool
        tool.warm_up()

        assert pool is not None
        assert tool._pool is pool

    def test_invoke_warms_up_lazily(self, tool):
        assert tool._pool is None

        assert tool.invoke(code="1") == "result:\n1"
        assert tool._pool is not None

    def test_close_is_idempotent_and_invoke_restarts_pool(self, tool):
        tool.warm_up()
        tool.close()
        tool.close()

        assert tool._pool is None
        assert tool.invoke(code="1") == "result:\n1"


class TestSerialization:
    def test_to_dict_defaults(self):
        assert MontyPythonTool().to_dict() == {
            "type": "haystack_integrations.tools.monty.python_tool.MontyPythonTool",
            "data": {
                "name": "run_python",
                "description": _DEFAULT_DESCRIPTION,
                "resource_limits": {"max_feed_duration_secs": 30.0, "max_memory": 256 * 1024 * 1024},
                "type_check": False,
                "max_output_chars": 20_000,
            },
        }

    def test_round_trip_with_custom_params(self):
        tool = MontyPythonTool(
            name="python",
            description="Run some Python.",
            resource_limits={"max_feed_duration_secs": 5.0, "max_recursion_depth": 200},
            type_check=True,
            max_output_chars=500,
        )

        restored = MontyPythonTool.from_dict(tool.to_dict())

        assert restored.to_dict() == tool.to_dict()
        assert restored.name == "python"
        assert restored.description == "Run some Python."
        assert restored._resource_limits == {
            "max_feed_duration_secs": 5.0,
            "max_memory": 256 * 1024 * 1024,
            "max_recursion_depth": 200,
        }
        assert restored._type_check is True
        assert restored._max_output_chars == 500

    def test_agent_round_trip(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        agent = Agent(
            chat_generator=OpenAIChatGenerator(),
            tools=[MontyPythonTool(resource_limits={"max_feed_duration_secs": 5.0})],
        )

        restored = Agent.from_dict(agent.to_dict())

        assert isinstance(restored.tools[0], MontyPythonTool)
        assert restored.tools[0].to_dict() == agent.tools[0].to_dict()


def test_agent_runs_code(tool):
    agent = Agent(chat_generator=_ScriptedChatGenerator(code="print(sum(range(101)))"), tools=[tool])
    agent.warm_up()

    result = agent.run(messages=[ChatMessage.from_user("What is the sum of the numbers from 0 to 100?")])

    tool_message = result["messages"][-2]
    assert tool_message.tool_call_result.result == "output:\n5050"
    assert tool_message.tool_call_result.error is False
    assert result["last_message"].text == "The sandbox said: output:\n5050"
