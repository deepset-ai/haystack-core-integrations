# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool
from pydantic_monty import (
    CollectString,
    Monty,
    MontyCrashedError,
    MontyRuntimeError,
    MontySyntaxError,
    MontyTypingError,
    ResourceLimits,
)

_DEFAULT_RESOURCE_LIMITS: ResourceLimits = {"max_feed_duration_secs": 30.0, "max_memory": 256 * 1024 * 1024}

_DEFAULT_DESCRIPTION = (
    "Run Python code in an isolated sandbox and get back what it printed and the value of its last expression. "
    "Use it for calculations, data processing, and anything else that is more reliable to compute than to guess.\n"
    "- Every call starts a fresh interpreter: variables, functions, and imports from earlier calls are gone, so each "
    "snippet must be self-contained.\n"
    "- The sandbox has no file system, network, environment variables, or subprocesses, and no third-party packages. "
    "Only these standard library modules can be imported, some of them partially: asyncio, base64, binascii, "
    "collections, copy, dataclasses, datetime, functools, itertools, json, math, random, re, sys, time, typing, "
    "unicodedata.\n"
    "- The interpreter supports a subset of Python: functions, lambdas, closures, comprehensions, simple classes, "
    "dataclasses, try/except, f-strings, and async/await. Class inheritance (including custom exception classes), "
    "generators (`yield`), `match`, `del`, and method decorators such as `@property` or `@staticmethod` are not "
    "supported.\n"
    "- Use `print()` for intermediate output. Errors come back as a traceback, so you can fix the code and retry."
)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[... {len(text) - max_chars} characters truncated]"


def _format_output(*, output: str, result: Any, error: str | None, max_chars: int) -> str:
    """Render the printed output, the final expression value, and any error as text for the LLM."""
    sections = []
    if output:
        sections.append(f"output:\n{_truncate(output.rstrip(), max_chars)}")
    if result is not None:
        sections.append(f"result:\n{_truncate(repr(result), max_chars)}")
    if error is not None:
        sections.append(f"error:\n{error}")
    return "\n\n".join(sections) or "The code ran successfully and produced no output."


class MontyPythonTool(Tool):
    """
    A Haystack `Tool` that lets an `Agent` run Python code in a [Monty](https://pydantic.dev/docs/monty/) sandbox.

    Monty is a minimal Python interpreter written in Rust. The tool keeps a pool of Monty worker processes and runs
    every call in a fresh interpreter, so nothing leaks between calls, users, or concurrent tool invocations. The
    tool returns what the code printed, the `repr()` of its last expression, and any error as a traceback, so the
    LLM can read the result and fix its code.

    ### Security model

    Monty is a language-level sandbox: its interpreter implements no operation that reaches the host, so code has
    no access to files, the network, environment variables, or subprocesses. It runs in worker subprocesses started
    with an empty environment, so a crash never takes down the host process. The tool mounts no directories and
    exposes no host functions to the sandbox. Execution time and heap memory are capped by `resource_limits`.

    ### Usage example

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.tools.monty import MontyPythonTool

    # Requires the OPENAI_API_KEY environment variable
    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[MontyPythonTool()])
    result = agent.run(messages=[ChatMessage.from_user("What is the sum of the first 100 prime numbers?")])
    print(result["last_message"].text)
    # >> The sum of the first 100 prime numbers is 24133.
    ```
    """

    def __init__(
        self,
        *,
        name: str = "run_python",
        description: str | None = None,
        resource_limits: ResourceLimits | None = None,
        type_check: bool = False,
        max_output_chars: int = 20_000,
    ) -> None:
        """
        Create a MontyPythonTool.

        :param name: Tool name exposed to the LLM.
        :param description: Tool description exposed to the LLM. If `None`, a description of the sandbox and the
            supported Python subset is used.
        :param resource_limits: Monty resource limits for each call, merged over the defaults of 30 seconds of
            execution time (`max_feed_duration_secs`) and 256 MiB of heap memory (`max_memory`). Set a key to `None`
            to disable that limit. See `pydantic_monty.ResourceLimits` for the available keys.
        :param type_check: If `True`, type-check the code with Monty's bundled type checker before running it, and
            return type errors to the LLM instead of executing the code.
        :param max_output_chars: Maximum number of characters kept from the printed output and from the result each
            before they are returned to the LLM.
        :raises ValueError: If `resource_limits` contains a key that Monty doesn't support, or `max_output_chars` is
            less than 1.
        """
        unknown_limits = set(resource_limits or {}) - set(ResourceLimits.__annotations__)
        if unknown_limits:
            msg = (
                f"Unknown resource limits: {sorted(unknown_limits)}. "
                f"Supported keys are: {sorted(ResourceLimits.__annotations__)}."
            )
            raise ValueError(msg)
        if max_output_chars < 1:
            msg = f"max_output_chars must be at least 1, got {max_output_chars}."
            raise ValueError(msg)

        self._resource_limits: ResourceLimits = {**_DEFAULT_RESOURCE_LIMITS}
        self._resource_limits.update(resource_limits or {})
        self._type_check = type_check
        self._max_output_chars = max_output_chars
        self._pool: Monty | None = None
        self._pool_lock = threading.Lock()

        parameters = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to run. The value of the last expression is returned.",
                }
            },
            "required": ["code"],
        }
        super().__init__(
            name=name,
            description=description or _DEFAULT_DESCRIPTION,
            parameters=parameters,
            function=self._run,
        )

    def warm_up(self) -> None:
        """Start the pool of Monty worker processes. Called by `Agent.warm_up()`; safe to call more than once."""
        self._get_pool()

    def close(self) -> None:
        """
        Shut down the pool of Monty worker processes.

        Safe to call more than once. The tool starts a new pool if it is invoked again.
        """
        with self._pool_lock:
            if self._pool is not None:
                self._pool.__exit__(None, None, None)
                self._pool = None

    def _get_pool(self) -> Monty:
        with self._pool_lock:
            if self._pool is None:
                pool = Monty()
                # Monty only exposes its pool lifecycle as a context manager. The tool keeps the pool open across
                # calls, so it enters the context here and exits it in `close()`.
                pool.__enter__()
                self._pool = pool
            return self._pool

    def _run(self, code: str) -> str:
        pool = self._get_pool()
        collector = CollectString()
        result = None
        error = None
        try:
            with pool.checkout(limits=self._resource_limits, type_check=self._type_check) as session:
                result = session.feed_run(code, print_callback=collector)
        except (MontySyntaxError, MontyRuntimeError, MontyTypingError) as e:
            error = e.display()
        except MontyCrashedError as e:
            # Code stuck inside a long-running builtin, such as a huge integer power, can't be interrupted by the
            # sandbox's own time limit; the pool then kills the worker and replaces it.
            if e.timed_out:
                error = f"TimeoutError: the code exceeded the time limit and the sandbox was stopped ({e})"
            else:
                error = f"The sandbox process crashed and was restarted ({e})"
        return _format_output(output=collector.output, result=result, error=error, max_chars=self._max_output_chars)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the tool to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "name": self.name,
                "description": self.description,
                "resource_limits": self._resource_limits,
                "type_check": self._type_check,
                "max_output_chars": self._max_output_chars,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MontyPythonTool":
        """
        Deserialize the tool from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized tool.
        """
        return cls(**data["data"])
