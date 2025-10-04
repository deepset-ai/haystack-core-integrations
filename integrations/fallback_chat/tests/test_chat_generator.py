# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from typing import Any, Optional

import pytest
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import Tool, Toolset

from haystack_integrations.components.generators.fallback_chat import FallbackChatGenerator


@component
class _DummySuccessGen:
    def __init__(self, text: str = "ok", delay: float = 0.0, streaming_callback: Optional[StreamingCallbackT] = None):
        self.text = text
        self.delay = delay
        self.streaming_callback = streaming_callback

    def to_dict(self) -> dict[str, Any]:
        # Include streaming_callback for nested serialization test
        # (serialize as dotted path if provided via serialize_callable, but here we keep None to keep tests simple)
        return default_to_dict(self, text=self.text, delay=self.delay, streaming_callback=None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_DummySuccessGen":
        return default_from_dict(cls, data)

    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: (list[Tool] | Toolset) | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        if self.delay:
            time.sleep(self.delay)
        if streaming_callback:
            streaming_callback({"dummy": True})  # type: ignore[arg-type]
        return {"replies": [ChatMessage.from_assistant(self.text)], "meta": {"dummy_meta": True}}

    async def run_async(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: (list[Tool] | Toolset) | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        if self.delay:
            await asyncio.sleep(self.delay)
        if streaming_callback:
            await asyncio.sleep(0)
            streaming_callback({"dummy": True})  # type: ignore[arg-type]
        return {"replies": [ChatMessage.from_assistant(self.text)], "meta": {"dummy_meta": True}}


@component
class _DummyFailGen:
    def __init__(self, exc: Exception | None = None, delay: float = 0.0):
        self.exc = exc or RuntimeError("boom")
        self.delay = delay

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, exc={"message": str(self.exc)}, delay=self.delay)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_DummyFailGen":
        # Recreate with a simple RuntimeError from message for serialization test
        init = data.get("init_parameters", {})
        msg = None
        if isinstance(init.get("exc"), dict):
            msg = init.get("exc", {}).get("message")
        return cls(exc=RuntimeError(msg or "boom"), delay=init.get("delay", 0.0))

    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: (list[Tool] | Toolset) | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        if self.delay:
            time.sleep(self.delay)
        raise self.exc

    async def run_async(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: (list[Tool] | Toolset) | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        if self.delay:
            await asyncio.sleep(self.delay)
        raise self.exc


def test_init_validation():
    with pytest.raises(ValueError):
        FallbackChatGenerator(generators=[], timeout=1.0)

    with pytest.raises(ValueError):
        FallbackChatGenerator(generators=[_DummySuccessGen()], timeout=0)

    # Duck typing: object without run
    class _NoRun:
        pass

    with pytest.raises(TypeError):
        FallbackChatGenerator(generators=[_NoRun()], timeout=1.0)  # type: ignore[arg-type]


def test_sequential_first_success():
    gen = FallbackChatGenerator(generators=[_DummySuccessGen(text="A")], timeout=1.0)
    res = gen.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "A"
    assert res["meta"]["successful_generator_index"] == 0
    assert res["meta"]["total_attempts"] == 1


def test_sequential_second_success_after_failure():
    gen = FallbackChatGenerator(generators=[_DummyFailGen(), _DummySuccessGen(text="B")], timeout=1.0)
    res = gen.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "B"
    assert res["meta"]["successful_generator_index"] == 1
    assert res["meta"]["failed_generators"]


def test_all_fail_raises():
    gen = FallbackChatGenerator(generators=[_DummyFailGen(), _DummyFailGen()], timeout=0.2)
    with pytest.raises(RuntimeError):
        gen.run([ChatMessage.from_user("hi")])


def test_timeout_handling_sync():
    # First generator sleeps longer than timeout, second succeeds
    slow = _DummySuccessGen(text="slow", delay=0.5)
    fast = _DummySuccessGen(text="fast", delay=0.0)
    gen = FallbackChatGenerator(generators=[slow, fast], timeout=0.1)
    res = gen.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "fast"


@pytest.mark.asyncio
async def test_timeout_handling_async():
    slow = _DummySuccessGen(text="slow", delay=0.5)
    fast = _DummySuccessGen(text="fast", delay=0.0)
    gen = FallbackChatGenerator(generators=[slow, fast], timeout=0.1)
    res = await gen.run_async([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "fast"


def test_streaming_callback_forwarding_sync():
    calls: list[Any] = []

    def cb(x: Any) -> None:
        calls.append(x)

    gen = FallbackChatGenerator(generators=[_DummySuccessGen(text="A")], timeout=1.0)
    _ = gen.run([ChatMessage.from_user("hi")], streaming_callback=cb)
    assert calls, "Expected streaming callback to be invoked"


@pytest.mark.asyncio
async def test_streaming_callback_forwarding_async():
    calls: list[Any] = []

    def cb(x: Any) -> None:
        calls.append(x)

    gen = FallbackChatGenerator(generators=[_DummySuccessGen(text="A")], timeout=1.0)
    _ = await gen.run_async([ChatMessage.from_user("hi")], streaming_callback=cb)
    assert calls, "Expected streaming callback to be invoked"


def test_serialization_roundtrip():
    original = FallbackChatGenerator(generators=[_DummySuccessGen(text="hello")], timeout=2.5)
    data = original.to_dict()
    restored = FallbackChatGenerator.from_dict(data)
    assert isinstance(restored, FallbackChatGenerator)
    assert len(restored.generators) == 1
    res = restored.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "hello"


def test_automatic_completion_mode_without_streaming():
    """Test that completion mode is used when no streaming callback is provided."""
    gen = FallbackChatGenerator(generators=[_DummySuccessGen(text="completion")], timeout=1.0)
    res = gen.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "completion"
    assert res["meta"]["successful_generator_index"] == 0


def test_automatic_ttft_mode_with_streaming():
    """Test that TTFT mode is used when streaming callback is provided."""
    calls: list[Any] = []

    def cb(x: Any) -> None:
        calls.append(x)

    gen = FallbackChatGenerator(generators=[_DummySuccessGen(text="ttft")], timeout=1.0)
    res = gen.run([ChatMessage.from_user("hi")], streaming_callback=cb)
    assert res["replies"][0].text == "ttft"
    assert calls, "Expected streaming callback to be invoked in TTFT mode"


@pytest.mark.asyncio
async def test_automatic_ttft_mode_with_streaming_async():
    """Test that TTFT mode is used when streaming callback is provided in async."""
    calls: list[Any] = []

    def cb(x: Any) -> None:
        calls.append(x)

    gen = FallbackChatGenerator(generators=[_DummySuccessGen(text="ttft_async")], timeout=1.0)
    res = await gen.run_async([ChatMessage.from_user("hi")], streaming_callback=cb)
    assert res["replies"][0].text == "ttft_async"
    assert calls, "Expected streaming callback to be invoked in TTFT mode"


@component
class _DummyTimeoutGen:
    """A dummy generator that has its own timeout setting."""

    def __init__(self, text: str = "ok", timeout: float | None = None, delay: float = 0.0):
        self.text = text
        self.timeout = timeout
        self.delay = delay

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, text=self.text, timeout=self.timeout, delay=self.delay)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_DummyTimeoutGen":
        return default_from_dict(cls, data)

    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: (list[Tool] | Toolset) | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        if self.delay:
            time.sleep(self.delay)
        return {"replies": [ChatMessage.from_assistant(self.text)], "meta": {"dummy_timeout": self.timeout}}


def test_respects_individual_generator_timeout():
    """Test that individual generator timeouts are respected."""
    # Generator with its own timeout that's shorter than FallbackChatGenerator timeout
    short_timeout_gen = _DummyTimeoutGen(text="short", timeout=0.1)
    gen = FallbackChatGenerator(generators=[short_timeout_gen], timeout=1.0)
    res = gen.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "short"
    assert res["meta"]["timeout_used"] == 0.1  # Should use the shorter generator timeout


def test_respects_fallback_timeout_when_generator_has_longer_timeout():
    """Test that FallbackChatGenerator timeout is used when it's shorter than generator timeout."""
    # Generator with its own timeout that's longer than FallbackChatGenerator timeout
    long_timeout_gen = _DummyTimeoutGen(text="long", timeout=2.0)
    gen = FallbackChatGenerator(generators=[long_timeout_gen], timeout=0.5)
    res = gen.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "long"
    assert res["meta"]["timeout_used"] == 0.5  # Should use the shorter FallbackChatGenerator timeout


def test_uses_fallback_timeout_when_generator_has_no_timeout():
    """Test that FallbackChatGenerator timeout is used when generator has no timeout."""
    # Generator without timeout
    no_timeout_gen = _DummyTimeoutGen(text="no_timeout", timeout=None)
    gen = FallbackChatGenerator(generators=[no_timeout_gen], timeout=1.5)
    res = gen.run([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "no_timeout"
    assert res["meta"]["timeout_used"] == 1.5  # Should use FallbackChatGenerator timeout


def test_timeout_precedence_with_multiple_generators():
    """Test timeout precedence with multiple generators having different timeout settings."""
    gen1 = _DummyTimeoutGen(text="gen1", timeout=0.2)  # Short timeout
    gen2 = _DummyTimeoutGen(text="gen2", timeout=2.0)  # Long timeout
    gen3 = _DummyTimeoutGen(text="gen3", timeout=None)  # No timeout

    fallback = FallbackChatGenerator(generators=[gen1, gen2, gen3], timeout=1.0)
    res = fallback.run([ChatMessage.from_user("hi")])

    # Should succeed with first generator
    assert res["replies"][0].text == "gen1"
    assert res["meta"]["timeout_used"] == 0.2  # Should use the shortest timeout


def test_timeout_precedence_with_failing_generators():
    """Test timeout precedence when some generators fail."""
    # First generator fails, second has longer timeout, third has shorter timeout
    fail_gen = _DummyFailGen()
    long_timeout_gen = _DummyTimeoutGen(text="long", timeout=2.0)
    short_timeout_gen = _DummyTimeoutGen(text="short", timeout=0.3)

    fallback = FallbackChatGenerator(generators=[fail_gen, long_timeout_gen, short_timeout_gen], timeout=1.0)
    res = fallback.run([ChatMessage.from_user("hi")])

    # Should succeed with second generator (first failed)
    assert res["replies"][0].text == "long"
    assert res["meta"]["timeout_used"] == 1.0  # Should use FallbackChatGenerator timeout (minimum of 1.0 and 2.0)
    assert res["meta"]["successful_generator_index"] == 1


@pytest.mark.asyncio
async def test_respects_individual_generator_timeout_async():
    """Test that individual generator timeouts are respected in async mode."""
    # Generator with its own timeout that's shorter than FallbackChatGenerator timeout
    short_timeout_gen = _DummyTimeoutGen(text="short_async", timeout=0.1)
    gen = FallbackChatGenerator(generators=[short_timeout_gen], timeout=1.0)
    res = await gen.run_async([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "short_async"
    assert res["meta"]["timeout_used"] == 0.1  # Should use the shorter generator timeout


@pytest.mark.asyncio
async def test_respects_fallback_timeout_when_generator_has_longer_timeout_async():
    """Test that FallbackChatGenerator timeout is used when it's shorter than generator timeout in async mode."""
    # Generator with its own timeout that's longer than FallbackChatGenerator timeout
    long_timeout_gen = _DummyTimeoutGen(text="long_async", timeout=2.0)
    gen = FallbackChatGenerator(generators=[long_timeout_gen], timeout=0.5)
    res = await gen.run_async([ChatMessage.from_user("hi")])
    assert res["replies"][0].text == "long_async"
    assert res["meta"]["timeout_used"] == 0.5  # Should use the shorter FallbackChatGenerator timeout


# HTTP Error Classes for Testing Failover Triggers
class HTTPError(Exception):
    """Base HTTP error class for testing."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class RateLimitError(HTTPError):
    """429 Rate Limit Error."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(429, message)


class AuthenticationError(HTTPError):
    """401 Authentication Error."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(401, message)


class BadRequestError(HTTPError):
    """400 Bad Request Error (e.g., context length exceeded)."""

    def __init__(self, message: str = "Bad request"):
        super().__init__(400, message)


class ServerError(HTTPError):
    """500+ Server Error."""

    def __init__(self, message: str = "Internal server error"):
        super().__init__(500, message)


@component
class _DummyHTTPErrorGen:
    """A dummy generator that raises specific HTTP errors for testing failover triggers."""

    def __init__(self, text: str = "success", error: Exception | None = None):
        self.text = text
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, text=self.text, error=str(self.error) if self.error else None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_DummyHTTPErrorGen":
        init = data.get("init_parameters", {})
        error = None
        if init.get("error"):
            error = RuntimeError(init["error"])  # Simplified for serialization
        return cls(text=init.get("text", "success"), error=error)

    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: (list[Tool] | Toolset) | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        if self.error:
            raise self.error
        return {
            "replies": [ChatMessage.from_assistant(self.text)],
            "meta": {"error_type": type(self.error).__name__ if self.error else None},
        }


def test_failover_trigger_429_rate_limit():
    """Test that 429 rate limit errors trigger failover."""
    rate_limit_gen = _DummyHTTPErrorGen(text="rate_limited", error=RateLimitError())
    success_gen = _DummySuccessGen(text="success_after_rate_limit")

    fallback = FallbackChatGenerator(generators=[rate_limit_gen, success_gen], timeout=1.0)
    result = fallback.run([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_rate_limit"
    assert result["meta"]["successful_generator_index"] == 1
    assert result["meta"]["failed_generators"] == ["_DummyHTTPErrorGen"]


def test_failover_trigger_401_authentication():
    """Test that 401 authentication errors trigger failover."""
    auth_error_gen = _DummyHTTPErrorGen(text="auth_failed", error=AuthenticationError())
    success_gen = _DummySuccessGen(text="success_after_auth")

    fallback = FallbackChatGenerator(generators=[auth_error_gen, success_gen], timeout=1.0)
    result = fallback.run([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_auth"
    assert result["meta"]["successful_generator_index"] == 1
    assert result["meta"]["failed_generators"] == ["_DummyHTTPErrorGen"]


def test_failover_trigger_400_bad_request():
    """Test that 400 bad request errors (e.g., context length exceeded) trigger failover."""
    bad_request_gen = _DummyHTTPErrorGen(text="bad_request", error=BadRequestError("Context length exceeded"))
    success_gen = _DummySuccessGen(text="success_after_bad_request")

    fallback = FallbackChatGenerator(generators=[bad_request_gen, success_gen], timeout=1.0)
    result = fallback.run([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_bad_request"
    assert result["meta"]["successful_generator_index"] == 1
    assert result["meta"]["failed_generators"] == ["_DummyHTTPErrorGen"]


def test_failover_trigger_500_server_error():
    """Test that 500+ server errors trigger failover."""
    server_error_gen = _DummyHTTPErrorGen(text="server_error", error=ServerError())
    success_gen = _DummySuccessGen(text="success_after_server_error")

    fallback = FallbackChatGenerator(generators=[server_error_gen, success_gen], timeout=1.0)
    result = fallback.run([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_server_error"
    assert result["meta"]["successful_generator_index"] == 1
    assert result["meta"]["failed_generators"] == ["_DummyHTTPErrorGen"]


def test_failover_trigger_408_timeout():
    """Test that 408 timeout errors trigger failover (already covered by asyncio.TimeoutError)."""
    # This is already tested in the existing timeout tests, but let's be explicit
    slow_gen = _DummySuccessGen(text="slow", delay=0.5)  # Will timeout
    fast_gen = _DummySuccessGen(text="fast", delay=0.0)

    fallback = FallbackChatGenerator(generators=[slow_gen, fast_gen], timeout=0.1)
    result = fallback.run([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "fast"
    assert result["meta"]["successful_generator_index"] == 1
    assert result["meta"]["failed_generators"] == ["_DummySuccessGen"]


def test_failover_trigger_multiple_errors():
    """Test that multiple different error types all trigger failover."""
    rate_limit_gen = _DummyHTTPErrorGen(text="rate_limited", error=RateLimitError())
    auth_error_gen = _DummyHTTPErrorGen(text="auth_failed", error=AuthenticationError())
    server_error_gen = _DummyHTTPErrorGen(text="server_error", error=ServerError())
    success_gen = _DummySuccessGen(text="success_after_all_errors")

    fallback = FallbackChatGenerator(
        generators=[rate_limit_gen, auth_error_gen, server_error_gen, success_gen], timeout=1.0
    )
    result = fallback.run([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_all_errors"
    assert result["meta"]["successful_generator_index"] == 3
    assert len(result["meta"]["failed_generators"]) == 3


def test_failover_trigger_all_generators_fail():
    """Test that when all generators fail with different error types, a RuntimeError is raised."""
    rate_limit_gen = _DummyHTTPErrorGen(text="rate_limited", error=RateLimitError())
    auth_error_gen = _DummyHTTPErrorGen(text="auth_failed", error=AuthenticationError())
    server_error_gen = _DummyHTTPErrorGen(text="server_error", error=ServerError())

    fallback = FallbackChatGenerator(generators=[rate_limit_gen, auth_error_gen, server_error_gen], timeout=1.0)

    with pytest.raises(RuntimeError) as exc_info:
        fallback.run([ChatMessage.from_user("test")])

    error_msg = str(exc_info.value)
    assert "All 3 generators failed" in error_msg
    assert "Failed generators: [_DummyHTTPErrorGen, _DummyHTTPErrorGen, _DummyHTTPErrorGen]" in error_msg


@pytest.mark.asyncio
async def test_failover_trigger_429_rate_limit_async():
    """Test that 429 rate limit errors trigger failover in async mode."""
    rate_limit_gen = _DummyHTTPErrorGen(text="rate_limited", error=RateLimitError())
    success_gen = _DummySuccessGen(text="success_after_rate_limit")

    fallback = FallbackChatGenerator(generators=[rate_limit_gen, success_gen], timeout=1.0)
    result = await fallback.run_async([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_rate_limit"
    assert result["meta"]["successful_generator_index"] == 1
    assert result["meta"]["failed_generators"] == ["_DummyHTTPErrorGen"]


@pytest.mark.asyncio
async def test_failover_trigger_401_authentication_async():
    """Test that 401 authentication errors trigger failover in async mode."""
    auth_error_gen = _DummyHTTPErrorGen(text="auth_failed", error=AuthenticationError())
    success_gen = _DummySuccessGen(text="success_after_auth")

    fallback = FallbackChatGenerator(generators=[auth_error_gen, success_gen], timeout=1.0)
    result = await fallback.run_async([ChatMessage.from_user("test")])

    assert result["replies"][0].text == "success_after_auth"
    assert result["meta"]["successful_generator_index"] == 1
    assert result["meta"]["failed_generators"] == ["_DummyHTTPErrorGen"]
