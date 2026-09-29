# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import time
from collections.abc import Mapping
from typing import Any, Literal

import httpx
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.memory_stores.everos.errors import EverOSMemoryStoreError
from haystack_integrations.memory_stores.everos.filters import build_search_filters

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.evermind.ai"
_DEFAULT_AGENT_ID = "haystack-agent"
_API_PREFIX = "/api/v2"
_SUPPORTED_ROLES = {"user", "assistant", "tool"}
_MAX_TOP_K = 100


class EverOSMemoryStore:
    """
    A Haystack memory store backed by the EverOS HTTP API.

    The store connects to EverOS Cloud. EverOS keeps user episodes and profiles separate from agent cases and skills,
    so searches must provide exactly one of `user_id` or `agent_id`.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        api_key: Secret = Secret.from_env_var("EVEROS_CLOUD_API_KEY"),
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the EverOS memory store.

        :param base_url: EverOS server root URL or a URL ending in `/api/v2`. Defaults to EverOS Cloud.
        :param api_key: EverOS Cloud bearer token, normally read from `EVEROS_CLOUD_API_KEY`.
        :param timeout: HTTP request timeout in seconds.
        """
        if not base_url.strip():
            msg = "base_url must not be empty."
            raise ValueError(msg)
        if timeout <= 0:
            msg = "timeout must be greater than zero."
            raise ValueError(msg)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def warm_up(self) -> None:
        """Create the HTTP client. Calling this method more than once is a no-op."""
        if self._client is not None:
            return
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "everos-haystack",
        }
        api_key = self.api_key.resolve_value()
        headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=self.timeout, headers=headers)

    @property
    def client(self) -> httpx.Client:
        """Return the initialized HTTP client, creating it on first use."""
        self.warm_up()
        if self._client is None:  # pragma: no cover - defensive guard
            msg = "EverOS HTTP client could not be initialized."
            raise EverOSMemoryStoreError(msg)
        return self._client

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the store configuration to a dictionary."""
        return default_to_dict(
            self,
            base_url=self.base_url,
            api_key=self.api_key.to_dict(),
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EverOSMemoryStore":
        """Deserialize a store configuration from a dictionary."""
        if data.get("init_parameters"):
            deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def add_memories(
        self,
        *,
        messages: list[ChatMessage],
        session_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str = "default",
        project_id: str = "default",
        defer_extraction: bool = False,
        flush: bool = False,
    ) -> dict[str, Any]:
        """
        Add Haystack chat messages to an EverOS session buffer.

        EverOS extracts durable memories at a semantic boundary. Set `flush=True` to force extraction after adding
        the messages instead of waiting for a later boundary.

        :param messages: Chat messages to add. System messages are ignored because EverOS accepts user, assistant,
            and tool roles only.
        :param session_id: Stable conversation or run identifier.
        :param user_id: Sender ID used for user messages and the owner of extracted user memory.
        :param agent_id: Sender ID used for assistant and tool messages. Defaults to `haystack-agent` when omitted.
        :param app_id: EverOS application scope.
        :param project_id: EverOS project scope.
        :param defer_extraction: Buffer messages without running boundary detection until a later flush.
        :param flush: Force extraction after the add request.
        :returns: The EverOS add response data plus `request_id` and optional `flush_status`.
        :raises ValueError: If required IDs are missing or no supported messages remain.
        :raises EverOSMemoryStoreError: If an EverOS API request fails or returns an invalid response.
        """
        if not session_id:
            msg = "session_id must not be empty."
            raise ValueError(msg)
        payload_messages = self._convert_messages(messages=messages, user_id=user_id, agent_id=agent_id)
        if not payload_messages:
            logger.warning("No supported, non-empty messages were available to send to EverOS.")
            return {
                "message_count": 0,
                "status": "skipped",
                "request_id": None,
                "flush_status": None,
            }

        response = self._post(
            "/memory/add",
            {
                "session_id": session_id,
                "app_id": app_id,
                "project_id": project_id,
                "messages": payload_messages,
                "async_mode": False,
                "defer_extraction": defer_extraction,
            },
        )
        data = self._response_data(response, operation="add memories")
        result = {
            "message_count": self._required_int(data, "message_count", operation="add memories"),
            "status": self._required_str(data, "status", operation="add memories"),
            "request_id": response.get("request_id"),
            "flush_status": None,
        }
        if flush:
            flushed = self.flush_memories(session_id=session_id, app_id=app_id, project_id=project_id)
            result["flush_status"] = flushed["status"]
        return result

    def flush_memories(
        self, *, session_id: str, app_id: str = "default", project_id: str = "default"
    ) -> dict[str, Any]:
        """
        Force extraction of an EverOS session buffer.

        :param session_id: Session buffer to flush.
        :param app_id: EverOS application scope.
        :param project_id: EverOS project scope.
        :returns: Dictionary containing `status` and `request_id`.
        :raises EverOSMemoryStoreError: If the endpoint is unavailable or the response is invalid.
        """
        response = self._post(
            "/memory/flush",
            {"session_id": session_id, "app_id": app_id, "project_id": project_id},
        )
        data = self._response_data(response, operation="flush memories")
        return {
            "status": self._required_str(data, "status", operation="flush memories"),
            "request_id": response.get("request_id"),
        }

    def search_memories(
        self,
        *,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str = "default",
        project_id: str = "default",
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        method: Literal["keyword", "vector", "hybrid", "agentic"] = "hybrid",
        top_k: int = 5,
        radius: float | None = None,
        min_score: float | None = None,
        include_profile: bool = False,
        enable_llm_rerank: bool = False,
        include_unprocessed: bool = False,
    ) -> list[ChatMessage]:
        """
        Search user or agent memory and return the results as system ChatMessages.

        :param query: Non-empty retrieval query.
        :param user_id: User-memory owner. Mutually exclusive with `agent_id`.
        :param agent_id: Agent-memory owner. Mutually exclusive with `user_id`.
        :param app_id: EverOS application scope.
        :param project_id: EverOS project scope.
        :param session_id: Optional session filter.
        :param filters: Optional Haystack metadata-filter expression. EverOS supports `session_id`, `parent_type`,
            `parent_id`, `timestamp`, and `sender_id` fields.
        :param method: EverOS retrieval method.
        :param top_k: Maximum number of results per memory kind, from 1 through 100.
        :param radius: Optional cosine-similarity threshold.
        :param min_score: Optional post-fusion relevance floor.
        :param include_profile: Include the user profile in user-memory results.
        :param enable_llm_rerank: Enable EverOS LLM reranking for compatible agent-memory searches.
        :param include_unprocessed: Include matching messages that remain in the session buffer.
        :returns: System messages containing episodes/profiles or agent cases/skills, with EverOS metadata.
        :raises ValueError: If the query or owner scope is invalid.
        :raises EverOSMemoryStoreError: If the EverOS API call fails or returns an invalid response.
        """
        if not query or not query.strip():
            msg = "EverOS search requires a non-empty query."
            raise ValueError(msg)
        if (user_id is None) == (agent_id is None):
            msg = "Exactly one of user_id or agent_id must be provided."
            raise ValueError(msg)
        if top_k < 1 or top_k > _MAX_TOP_K:
            msg = "top_k must be in 1..100."
            raise ValueError(msg)

        payload: dict[str, Any] = {
            "user_id": user_id,
            "agent_id": agent_id,
            "app_id": app_id,
            "project_id": project_id,
            "query": query,
            "method": method,
            "top_k": top_k,
            "include_profile": include_profile,
            "enable_llm_rerank": enable_llm_rerank,
        }
        converted_filters = build_search_filters(filters=filters, session_id=session_id)
        if converted_filters:
            payload["filters"] = converted_filters
        if radius is not None:
            payload["radius"] = radius
        if min_score is not None:
            payload["min_score"] = min_score

        response = self._post("/memory/search", payload)
        data = self._response_data(response, operation="search memories")
        return self._search_data_to_messages(
            data=data,
            request_id=response.get("request_id"),
            include_unprocessed=include_unprocessed,
        )

    def _convert_messages(
        self, *, messages: list[ChatMessage], user_id: str | None, agent_id: str | None
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        timestamp_base = int(time.time() * 1000)
        for index, message in enumerate(messages):
            role = message.role.value
            if role not in _SUPPORTED_ROLES:
                logger.debug("Ignoring Haystack {role} message because EverOS does not ingest that role.", role=role)
                continue
            if role == "user":
                if not user_id:
                    msg = "user_id is required when adding user messages to EverOS."
                    raise ValueError(msg)
                sender_id = user_id
            else:
                sender_id = agent_id or _DEFAULT_AGENT_ID

            content = self._message_text(message)
            tool_calls = self._tool_calls(message)
            tool_call_id = self._tool_call_id(message)
            if not content and not tool_calls:
                logger.debug("Ignoring an empty Haystack {role} message.", role=role)
                continue

            item: dict[str, Any] = {
                "sender_id": sender_id,
                "role": role,
                "timestamp": self._timestamp_ms(message, timestamp_base + index),
                "content": content,
            }
            sender_name = message.meta.get("sender_name")
            if isinstance(sender_name, str) and sender_name:
                item["sender_name"] = sender_name
            if tool_calls:
                item["tool_calls"] = tool_calls
            if tool_call_id:
                item["tool_call_id"] = tool_call_id
            converted.append(item)
        return converted

    @staticmethod
    def _message_text(message: ChatMessage) -> str:
        if message.text:
            return message.text
        results = message.tool_call_results
        if results:
            return "\n".join(str(result.result) for result in results)
        return ""

    @staticmethod
    def _tool_calls(message: ChatMessage) -> list[dict[str, Any]]:
        output = []
        for index, tool_call in enumerate(message.tool_calls):
            output.append(
                {
                    "id": tool_call.id or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": tool_call.tool_name,
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=False, separators=(",", ":")),
                    },
                }
            )
        return output

    @staticmethod
    def _tool_call_id(message: ChatMessage) -> str | None:
        results = message.tool_call_results
        if not results:
            return None
        return results[0].origin.id

    @staticmethod
    def _timestamp_ms(message: ChatMessage, fallback: int) -> int:
        value = message.meta.get("everos_timestamp_ms")
        if isinstance(value, int) and value > 0:
            return value
        return fallback

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._endpoint(path)
        response: httpx.Response | None = None
        try:
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as error:
            suffix = self._http_error_detail(error.response)
            msg = f"EverOS API request failed with HTTP {error.response.status_code}: {suffix}"
            raise EverOSMemoryStoreError(msg) from error
        except httpx.RequestError as error:
            msg = f"Could not reach EverOS at {self.base_url}: {error}"
            raise EverOSMemoryStoreError(msg) from error
        except ValueError as error:
            msg = "EverOS returned a non-JSON response."
            raise EverOSMemoryStoreError(msg) from error
        if not isinstance(body, dict):
            msg = "EverOS returned an invalid JSON response object."
            raise EverOSMemoryStoreError(msg)
        return body

    def _endpoint(self, path: str) -> str:
        if self.base_url.endswith(_API_PREFIX):
            return f"{self.base_url}{path}"
        return f"{self.base_url}{_API_PREFIX}{path}"

    @staticmethod
    def _http_error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text.strip() or response.reason_phrase
        if isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, Mapping):
                code = error.get("code")
                message = error.get("message")
                if code and message:
                    return f"{code}: {message}"
                if message:
                    return str(message)
            detail = body.get("detail")
            if detail:
                return str(detail)
        return response.reason_phrase

    @staticmethod
    def _response_data(response: dict[str, Any], *, operation: str) -> dict[str, Any]:
        data = response.get("data")
        if not isinstance(data, dict):
            msg = f"EverOS returned an invalid response while attempting to {operation}: missing object 'data'."
            raise EverOSMemoryStoreError(msg)
        return data

    @staticmethod
    def _required_str(data: dict[str, Any], key: str, *, operation: str) -> str:
        value = data.get(key)
        if not isinstance(value, str):
            msg = f"EverOS returned an invalid response while attempting to {operation}: '{key}' is not a string."
            raise EverOSMemoryStoreError(msg)
        return value

    @staticmethod
    def _required_int(data: dict[str, Any], key: str, *, operation: str) -> int:
        value = data.get(key)
        if not isinstance(value, int):
            msg = f"EverOS returned an invalid response while attempting to {operation}: '{key}' is not an integer."
            raise EverOSMemoryStoreError(msg)
        return value

    def _search_data_to_messages(
        self, *, data: dict[str, Any], request_id: Any, include_unprocessed: bool
    ) -> list[ChatMessage]:
        messages = []
        for item in self._items(data, "episodes"):
            facts = item.get("atomic_facts")
            fact_lines = []
            if isinstance(facts, list):
                fact_lines = [
                    str(fact.get("content")) for fact in facts if isinstance(fact, dict) and fact.get("content")
                ]
            content = str(item.get("episode") or item.get("summary") or "")
            if fact_lines:
                content = f"{content}\n\nRelevant facts:\n" + "\n".join(f"- {fact}" for fact in fact_lines)
            messages.append(self._memory_message(content, "episode", item, request_id))

        for item in self._items(data, "profiles"):
            profile = item.get("profile_data")
            content = (
                json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)
                if isinstance(profile, dict)
                else str(profile or "")
            )
            messages.append(self._memory_message(content, "profile", item, request_id))

        for item in self._items(data, "agent_cases"):
            parts = [f"Task: {item.get('task_intent', '')}", f"Approach: {item.get('approach', '')}"]
            if item.get("key_insight"):
                parts.append(f"Key insight: {item['key_insight']}")
            messages.append(self._memory_message("\n".join(parts), "agent_case", item, request_id))

        for item in self._items(data, "agent_skills"):
            parts = [f"Skill: {item.get('name', '')}", f"Description: {item.get('description', '')}"]
            if item.get("content"):
                parts.append(str(item["content"]))
            messages.append(self._memory_message("\n".join(parts), "agent_skill", item, request_id))

        if include_unprocessed:
            for item in self._items(data, "unprocessed_messages"):
                messages.append(
                    self._memory_message(str(item.get("content") or ""), "unprocessed_message", item, request_id)
                )
        return messages

    @staticmethod
    def _items(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
        raw = data.get(key, [])
        if not isinstance(raw, list):
            msg = f"EverOS returned an invalid search response: 'data.{key}' is not a list."
            raise EverOSMemoryStoreError(msg)
        return [item for item in raw if isinstance(item, dict)]

    @staticmethod
    def _memory_message(content: str, memory_type: str, item: dict[str, Any], request_id: Any) -> ChatMessage:
        content_fields = {"episode", "profile_data", "content", "approach"}
        metadata = {key: value for key, value in item.items() if key not in content_fields}
        metadata["memory_type"] = memory_type
        metadata["request_id"] = request_id
        return ChatMessage.from_system(content, meta={"everos": metadata})
