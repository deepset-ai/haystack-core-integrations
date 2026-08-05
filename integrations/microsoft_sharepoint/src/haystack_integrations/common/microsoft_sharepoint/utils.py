# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import TypeVar

import httpx
from haystack import logging
from haystack.utils import Secret
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from .errors import SharePointConfigError, SharePointRequestError

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_GRAPH_URL = "https://graph.microsoft.com/v1.0"
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
# Fallback backoff when a throttled response carries no `Retry-After` header: 1s, 2s, 4s, ...
_EXPONENTIAL_BACKOFF = wait_exponential()


async def _gather_tasks_with_cancel(tasks: list[asyncio.Task[T]]) -> list[T]:
    """
    Wait for all tasks, cancelling and draining unfinished siblings if one fails.

    :param tasks: Tasks to wait for.
    :returns:
        The task results in input order.
    """
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def resolve_access_token(access_token: str | Secret) -> str:
    """Return the bearer token string, resolving it if a `Secret` was passed."""
    if not isinstance(access_token, Secret):
        return access_token
    resolved = access_token.resolve_value()
    if not isinstance(resolved, str):
        msg = "The access_token Secret did not resolve to a string value."
        raise SharePointConfigError(msg)
    return resolved


def _retry_response(retry_state: RetryCallState) -> httpx.Response | None:
    """Return the `httpx.Response` recorded for a retry attempt, or `None` if the attempt raised."""
    outcome = retry_state.outcome
    if outcome is None or outcome.failed:
        return None
    return outcome.result()


def _is_retryable_response(response: httpx.Response) -> bool:
    """Return whether a Microsoft Graph response should trigger a retry."""
    return response.status_code in _RETRYABLE_STATUS


def _wait_with_retry_after(retry_state: RetryCallState) -> float:
    """Wait strategy honoring a numeric `Retry-After` header, falling back to exponential backoff."""
    response = _retry_response(retry_state)
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

    return _EXPONENTIAL_BACKOFF(retry_state)


def _last_response(retry_state: RetryCallState) -> httpx.Response:
    """Return the final response once retries are exhausted, so the caller can raise a typed error."""
    response = _retry_response(retry_state)
    if response is None:
        msg = "Microsoft Graph retries were exhausted without a recorded response."
        raise SharePointRequestError(msg)
    return response


def _log_retry(retry_state: RetryCallState) -> None:
    """Log a warning before sleeping between Microsoft Graph retries."""
    response = _retry_response(retry_state)
    next_action = retry_state.next_action
    if response is None or next_action is None:
        return
    logger.warning(
        "Microsoft Graph returned status {status}; retrying in {delay}s (attempt {attempt}).",
        status=response.status_code,
        delay=next_action.sleep,
        attempt=retry_state.attempt_number,
    )


def build_retrying(max_retries: int) -> Retrying:
    """Build a tenacity `Retrying` that retries Microsoft Graph requests on throttling/transient errors."""
    return Retrying(
        stop=stop_after_attempt(max_retries + 1),
        wait=_wait_with_retry_after,
        retry=retry_if_result(_is_retryable_response),
        before_sleep=_log_retry,
        retry_error_callback=_last_response,
    )


def build_async_retrying(max_retries: int) -> AsyncRetrying:
    """Async variant of `build_retrying`."""
    return AsyncRetrying(
        stop=stop_after_attempt(max_retries + 1),
        wait=_wait_with_retry_after,
        retry=retry_if_result(_is_retryable_response),
        before_sleep=_log_retry,
        retry_error_callback=_last_response,
    )
