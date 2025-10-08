# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from threading import Event as ThreadEvent
from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import Tool, Toolset
from haystack.utils.deserialization import deserialize_component_inplace

logger = logging.getLogger(__name__)


@component
class FallbackChatGenerator:
    """
    A chat generator wrapper that tries multiple chat generators sequentially with a per-generator timeout.

    It forwards all parameters transparently to the underlying generators and returns the first successful result.
    If all generators fail or time out, it raises a RuntimeError with details.

    The timeout behavior respects individual generator timeouts:
    - If a generator has its own timeout setting, the effective timeout is the minimum of the FallbackChatGenerator
      timeout and the generator's timeout
    - If a generator has no timeout setting, the FallbackChatGenerator timeout is used
    - For non-streaming calls, the timeout is applied to the complete response
    - For streaming calls, the timeout is applied as a time-to-first-token (TTFT) deadline

    Failover is automatically triggered for these error types:
    - 429 Rate limit errors
    - 401 Authentication errors
    - 400 Context length errors
    - 408 Timeout errors
    - 500+ Server errors
    - Any other exception raised by a generator
    """

    def __init__(
        self,
        generators: list[ChatGenerator],
        timeout: float = 8.0,
    ):
        """
        :param generators: A non-empty list of chat generator components to try in order.
        :param timeout: Per-generator timeout in seconds. Must be > 0.
                       This acts as an upper bound for individual generator timeouts.
                       For non-streaming calls, this is the completion timeout.
                       For streaming calls, this is the time-to-first-token (TTFT) timeout.
                       If individual generators have their own timeout settings, the effective
                       timeout will be the minimum of this value and the generator's timeout.
        """
        if not generators:
            msg = "'generators' must be a non-empty list"
            raise ValueError(msg)
        if timeout is None or timeout <= 0:
            msg = "'timeout' must be a positive number of seconds"
            raise ValueError(msg)

        # Validation via duck-typing: require a callable 'run' method
        for gen in generators:
            if not hasattr(gen, "run") or not callable(gen.run):
                msg = "All items in 'generators' must expose a callable 'run' method (duck-typed ChatGenerator)"
                raise TypeError(msg)

        self.generators = list(generators)
        self.timeout = float(timeout)

    # ---------------------- Serialization ----------------------
    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            generators=[gen.to_dict() for gen in self.generators if hasattr(gen, "to_dict")],
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FallbackChatGenerator:
        # Reconstruct nested generators from their serialized dicts
        init_params = data.get("init_parameters", {})
        serialized = init_params.get("generators") or []
        deserialized: list[Any] = []
        for g in serialized:
            # Use the generic component deserializer available in Haystack
            holder = {"component": g}
            deserialize_component_inplace(holder, key="component")
            deserialized.append(holder["component"])  # type: ignore[assignment]
        init_params["generators"] = deserialized
        data["init_parameters"] = init_params
        return default_from_dict(cls, data)

    # ---------------------- Execution helpers ----------------------
    def _get_effective_timeout(self, gen: Any) -> float:
        """
        Get the effective timeout for a generator.
        Uses the minimum of FallbackChatGenerator timeout and the generator's own timeout.

        :param gen: The generator to get timeout for
        :return: The effective timeout in seconds
        """
        # Get the generator's timeout if it has one
        gen_timeout = getattr(gen, "timeout", None)

        # If generator has no timeout or it's None, use FallbackChatGenerator timeout
        if gen_timeout is None:
            return self.timeout

        # Use the minimum of both timeouts
        return min(self.timeout, float(gen_timeout))

    def _run_single_sync(
        self,
        gen: Any,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None,
        tools: (list[Tool] | Toolset) | None,
        streaming_callback: StreamingCallbackT | None,
    ) -> dict[str, Any]:
        # Get the effective timeout for this generator
        effective_timeout = self._get_effective_timeout(gen)

        # Completion mode: when no streaming callback is provided
        if streaming_callback is None:

            def _call_completion() -> dict[str, Any]:
                return gen.run(
                    messages=messages,
                    generation_kwargs=generation_kwargs,
                    tools=tools,
                    streaming_callback=streaming_callback,
                )

            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_call_completion)
                try:
                    return fut.result(timeout=effective_timeout)
                except FuturesTimeoutError as e:  # Map to asyncio.TimeoutError for consistency
                    raise asyncio.TimeoutError(str(e)) from e

        # TTFT mode with streaming: treat timeout as TTFT deadline
        ttft = ThreadEvent()
        active = {"value": True}

        def wrapped_cb(chunk: Any) -> None:
            if not ttft.is_set():
                ttft.set()
            if not active["value"]:
                return
            if streaming_callback:
                streaming_callback(chunk)

        def _call_ttft() -> dict[str, Any]:
            return gen.run(
                messages=messages,
                generation_kwargs=generation_kwargs,
                tools=tools,
                streaming_callback=wrapped_cb,
            )

        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(_call_ttft)

        deadline = time.monotonic() + effective_timeout
        tick = 0.05
        try:
            while True:
                if fut.done():
                    # Completed before (or irrespective of) first token
                    return fut.result()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # Wait a short tick for first token; breaks early when token arrives
                ttft.wait(timeout=min(remaining, tick))
                if ttft.is_set():
                    # Commit to this generator: now wait for completion with remaining time
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        try:
                            return fut.result(timeout=remaining)
                        except FuturesTimeoutError as e:
                            raise asyncio.TimeoutError(str(e)) from e
                    else:
                        # Already at/past deadline after TTFT
                        break
        finally:
            # If we time out on TTFT, stop forwarding tokens and detach the worker
            if not ttft.is_set():
                active["value"] = False
            ex.shutdown(wait=False)

        # TTFT exceeded and not completed: raise timeout to trigger fallback
        msg = "TTFT deadline exceeded for generator"
        raise asyncio.TimeoutError(msg)

    async def _run_single_async(
        self,
        gen: Any,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None,
        tools: (list[Tool] | Toolset) | None,
        streaming_callback: StreamingCallbackT | None,
    ) -> dict[str, Any]:
        # Get the effective timeout for this generator
        effective_timeout = self._get_effective_timeout(gen)

        # Completion mode: when no streaming callback is provided
        if streaming_callback is None:
            if hasattr(gen, "run_async") and callable(gen.run_async):
                return await asyncio.wait_for(
                    gen.run_async(
                        messages=messages,
                        generation_kwargs=generation_kwargs,
                        tools=tools,
                        streaming_callback=streaming_callback,
                    ),
                    timeout=effective_timeout,
                )
            return await asyncio.wait_for(
                asyncio.to_thread(
                    gen.run,
                    messages=messages,
                    generation_kwargs=generation_kwargs,
                    tools=tools,
                    streaming_callback=streaming_callback,
                ),
                timeout=effective_timeout,
            )

        # TTFT mode: treat timeout as shared deadline for entire operation
        loop = asyncio.get_running_loop()
        ttft = asyncio.Event()
        active = {"value": True}
        start_time = time.time()

        def wrapped_cb(chunk: Any) -> None:
            if not ttft.is_set():
                # Ensure thread-safety if invoked from a different thread
                loop.call_soon_threadsafe(ttft.set)
            if not active["value"]:
                return
            if streaming_callback:
                streaming_callback(chunk)

        if hasattr(gen, "run_async") and callable(gen.run_async):
            task = asyncio.create_task(
                gen.run_async(
                    messages=messages,
                    generation_kwargs=generation_kwargs,
                    tools=tools,
                    streaming_callback=wrapped_cb,
                )
            )
        else:
            task = asyncio.create_task(
                asyncio.to_thread(
                    gen.run,
                    messages=messages,
                    generation_kwargs=generation_kwargs,
                    tools=tools,
                    streaming_callback=wrapped_cb,
                )
            )

        ttft_task = asyncio.create_task(ttft.wait())
        done, pending = await asyncio.wait(
            {task, ttft_task},
            timeout=effective_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if task in done:
            # Completed before first token (or with no streaming)
            ttft_task.cancel()
            try:
                _ = await ttft_task
            except Exception:  # noqa: S110
                # Intentionally ignore TTFT task exceptions - we only care about the main task
                pass
            return await task

        if ttft_task in done:
            # Commit to this generator; await completion with remaining time from shared deadline
            ttft_task.cancel()
            try:
                _ = await ttft_task
            except Exception:  # noqa: S110
                # Intentionally ignore TTFT task exceptions - we only care about the main task
                pass
            # Calculate remaining time from original deadline
            elapsed = time.time() - start_time
            remaining = effective_timeout - elapsed
            if remaining > 0:
                try:
                    return await asyncio.wait_for(task, timeout=remaining)
                except asyncio.TimeoutError as e:
                    # Completion took too long after first token
                    active["value"] = False
                    raise e
            else:
                # Already at/past deadline after TTFT
                active["value"] = False
                msg = "Deadline exceeded after TTFT (async)"
                raise asyncio.TimeoutError(msg)

        # Neither completed nor produced a first token within deadline
        active["value"] = False
        for p in pending:
            p.cancel()
        try:
            await task
        except Exception:  # noqa: S110
            # Intentionally ignore task exceptions - we're timing out and will raise our own error
            pass
        msg = "TTFT deadline exceeded for generator (async)"
        raise asyncio.TimeoutError(msg)

    # ---------------------- Public API ----------------------
    @component.output_types(replies=list[ChatMessage], meta=dict[str, Any])
    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: (list[Tool] | Toolset) | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        start = time.time()
        failed: list[str] = []
        last_error: BaseException | None = None

        for idx, gen in enumerate(self.generators):
            gen_name = gen.__class__.__name__
            effective_timeout = self._get_effective_timeout(gen)
            try:
                result = self._run_single_sync(gen, messages, generation_kwargs, tools, streaming_callback)
                replies = result.get("replies", [])
                meta = dict(result.get("meta", {}))
                meta.update(
                    {
                        "successful_generator_index": idx,
                        "successful_generator_class": gen_name,
                        "total_attempts": idx + 1,
                        "failed_generators": failed,
                        "timeout_used": effective_timeout,
                        "execution_time": time.time() - start,
                    }
                )
                return {"replies": replies, "meta": meta}
            except asyncio.TimeoutError as e:
                logger.warning("Generator %s timed out after %.2fs", gen_name, effective_timeout)
                failed.append(gen_name)
                last_error = e
            except Exception as e:
                logger.warning("Generator %s failed with error: %s", gen_name, e)
                failed.append(gen_name)
                last_error = e

        elapsed = time.time() - start
        failed_names = ", ".join(failed)
        msg = (
            f"All {len(self.generators)} generators failed after {elapsed:.2f}s. "
            f"Last error: {last_error}. Failed generators: [{failed_names}]"
        )
        raise RuntimeError(msg)

    @component.output_types(replies=list[ChatMessage], meta=dict[str, Any])
    async def run_async(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        tools: (list[Tool] | Toolset) | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> dict[str, Any]:
        start = time.time()
        failed: list[str] = []
        last_error: BaseException | None = None

        for idx, gen in enumerate(self.generators):
            gen_name = gen.__class__.__name__
            effective_timeout = self._get_effective_timeout(gen)
            try:
                result = await self._run_single_async(gen, messages, generation_kwargs, tools, streaming_callback)
                replies = result.get("replies", [])
                meta = dict(result.get("meta", {}))
                meta.update(
                    {
                        "successful_generator_index": idx,
                        "successful_generator_class": gen_name,
                        "total_attempts": idx + 1,
                        "failed_generators": failed,
                        "timeout_used": effective_timeout,
                        "execution_time": time.time() - start,
                    }
                )
                return {"replies": replies, "meta": meta}
            except asyncio.TimeoutError as e:
                logger.warning("Generator %s timed out after %.2fs (async)", gen_name, effective_timeout)
                failed.append(gen_name)
                last_error = e
            except Exception as e:
                logger.warning("Generator %s failed (async) with error: %s", gen_name, e)
                failed.append(gen_name)
                last_error = e

        elapsed = time.time() - start
        failed_names = ", ".join(failed)
        msg = (
            f"All {len(self.generators)} generators failed after {elapsed:.2f}s (async). "
            f"Last error: {last_error}. Failed generators: [{failed_names}]"
        )
        raise RuntimeError(msg)
