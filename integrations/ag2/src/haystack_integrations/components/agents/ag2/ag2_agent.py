# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, cast

from autogen import AssistantAgent, UserProxyAgent
from haystack import component, default_from_dict, default_to_dict

HUMAN_INPUT_MODES = {"ALWAYS", "NEVER", "TERMINATE"}


@component
class AG2Agent:
    """
    A Haystack component that wraps AG2 (formerly AutoGen) multi-agent conversations.

    Creates an AssistantAgent and UserProxyAgent pair to execute a conversation
    and return the final reply with full message history.

    The API key is read automatically from environment variables by AG2/litellm
    (e.g. ``OPENAI_API_KEY``).

    Usage example:

    ```python
    from haystack import Pipeline
    from haystack_integrations.components.agents.ag2 import AG2Agent

    pipeline = Pipeline()
    pipeline.add_component("agent", AG2Agent(model="gpt-4o-mini"))
    result = pipeline.run({"agent": {"query": "Explain RAG in one sentence."}})
    print(result["agent"]["reply"])
    ```
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_type: str = "openai",
        system_message: str | None = None,
        human_input_mode: str = "NEVER",
        code_execution: bool = False,
        max_consecutive_auto_reply: int = 10,
    ):
        """
        Initialize the AG2Agent component.

        :param model: The model identifier to use (e.g. ``"gpt-4o-mini"``).
        :param api_type: The API type string passed to AG2's ``LLMConfig``
            (e.g. ``"openai"``).
        :param system_message: Optional system message for the AssistantAgent.
            Defaults to AG2's built-in assistant system prompt.
        :param human_input_mode: Controls when the UserProxyAgent asks for human
            input. Must be one of ``"NEVER"``, ``"TERMINATE"``, or ``"ALWAYS"``.
            Use ``"NEVER"`` for fully automated pipeline use.
        :param code_execution: Whether to enable code execution in the
            UserProxyAgent. Defaults to ``False`` (safe for pipeline use).
        :param max_consecutive_auto_reply: Maximum number of consecutive
            auto-replies before the UserProxyAgent stops. Defaults to ``10``.
        :raises ValueError: If ``human_input_mode`` is not one of the valid modes.
        """
        if human_input_mode not in HUMAN_INPUT_MODES:
            msg = f"Invalid human_input_mode '{human_input_mode}'. Must be one of {sorted(HUMAN_INPUT_MODES)}"
            raise ValueError(msg)

        self.model = model
        self.api_type = api_type
        self.system_message = system_message
        self.human_input_mode = human_input_mode
        self.code_execution = code_execution
        self.max_consecutive_auto_reply = max_consecutive_auto_reply

    def _build_llm_config(self) -> dict[str, Any]:
        return {
            "config_list": [
                {
                    "model": self.model,
                    "api_type": self.api_type,
                }
            ]
        }

    def _build_code_execution_config(self) -> "dict[str, Any] | Literal[False]":
        if not self.code_execution:
            return False
        return {"use_docker": False}

    @component.output_types(reply=str, messages=list[dict[str, Any]])
    def run(self, query: str) -> dict[str, Any]:
        """
        Run a multi-agent conversation for the given query.

        Creates a fresh AssistantAgent and UserProxyAgent for each call,
        executes the conversation, and returns the final assistant reply
        along with the full message history.

        :param query: The user query to send to the AssistantAgent.
        :returns: A dict with:
            - ``reply`` (str): the final assistant reply text.
            - ``messages`` (list[dict]): the full conversation history.
        :raises ValueError: If no reply could be extracted from the conversation.
        """
        llm_config = self._build_llm_config()
        code_execution_config = self._build_code_execution_config()

        assistant_kwargs: dict[str, Any] = {
            "name": "assistant",
            "llm_config": llm_config,
            "human_input_mode": "NEVER",
        }
        if self.system_message is not None:
            assistant_kwargs["system_message"] = self.system_message

        assistant = AssistantAgent(**assistant_kwargs)

        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode=cast(Literal["ALWAYS", "TERMINATE", "NEVER"], self.human_input_mode),
            max_consecutive_auto_reply=self.max_consecutive_auto_reply,
            code_execution_config=code_execution_config,
            llm_config=False,
            is_termination_msg=lambda _: True,
        )

        chat_result = user_proxy.initiate_chat(
            assistant,
            message=query,
            silent=True,
        )

        messages: list[dict[str, Any]] = chat_result.chat_history or []

        reply = self._extract_last_assistant_reply(messages)

        return {"reply": reply, "messages": messages}

    @staticmethod
    def _extract_last_assistant_reply(messages: list[dict[str, Any]]) -> str:
        """
        Return the content of the last AssistantAgent message in the history.

        AG2 stores chat history from the UserProxyAgent's perspective: the proxy's
        own messages use role="assistant" and the AssistantAgent's replies use
        role="user". We therefore identify the real assistant replies by the agent
        name field ("assistant") rather than the role field.
        """
        for entry in reversed(messages):
            name = entry.get("name", "")
            content = entry.get("content", "")
            if name == "assistant" and content:
                return str(content)
        err = "No assistant reply found in conversation history."
        raise ValueError(err)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            model=self.model,
            api_type=self.api_type,
            system_message=self.system_message,
            human_input_mode=self.human_input_mode,
            code_execution=self.code_execution,
            max_consecutive_auto_reply=self.max_consecutive_auto_reply,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AG2Agent":
        """Deserialize a component from a dictionary."""
        return default_from_dict(cls, data)
