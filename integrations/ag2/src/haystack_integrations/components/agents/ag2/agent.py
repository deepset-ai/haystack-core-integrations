# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any

from autogen import AssistantAgent, LLMConfig, UserProxyAgent
from haystack import component, default_from_dict, default_to_dict

logger = logging.getLogger(__name__)


@component
class AG2Agent:
    """
    A Haystack component that wraps AG2 (formerly AutoGen) multi-agent conversations.

    AG2 is a multi-agent conversation framework with 500K+ monthly PyPI downloads,
    4,300+ GitHub stars, and 400+ contributors.

    This component enables using AG2's powerful multi-agent orchestration within
    Haystack pipelines. It creates an AssistantAgent and a UserProxyAgent, sends
    the input query to the agents, and returns the conversation result.

    ### Usage example

    ```python
    from haystack_integrations.components.agents.ag2 import AG2Agent

    agent = AG2Agent(
        model="gpt-4o-mini",
        system_message="You are a helpful research assistant.",
        api_key_env_var="OPENAI_API_KEY",
    )
    result = agent.run(query="What are the latest advances in RAG?")
    print(result["reply"])
    ```

    ### With Haystack Pipeline

    ```python
    from haystack import Pipeline
    from haystack_integrations.components.agents.ag2 import AG2Agent

    pipeline = Pipeline()
    pipeline.add_component("ag2_agent", AG2Agent(
        model="gpt-4o-mini",
        system_message="You are a helpful assistant that answers questions.",
    ))

    result = pipeline.run({"ag2_agent": {"query": "Explain RAG in simple terms."}})
    print(result["ag2_agent"]["reply"])
    ```
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_message: str = "You are a helpful AI assistant.",
        api_key_env_var: str = "OPENAI_API_KEY",
        api_type: str = "openai",
        max_consecutive_auto_reply: int = 10,
        human_input_mode: str = "NEVER",
        assistant_name: str = "assistant",
        user_proxy_name: str = "user_proxy",
        code_execution: bool = False,
    ):
        """
        Initialize the AG2Agent component.

        :param model: The LLM model name to use (e.g., "gpt-4o-mini", "gpt-4o").
        :param system_message: The system message for the AG2 AssistantAgent.
        :param api_key_env_var: Environment variable name containing the API key.
        :param api_type: The API type for AG2 LLMConfig (e.g., "openai", "bedrock").
        :param max_consecutive_auto_reply: Max consecutive auto-replies.
        :param human_input_mode: Human input mode ("NEVER", "ALWAYS", "TERMINATE").
        :param assistant_name: Name of the AG2 AssistantAgent.
        :param user_proxy_name: Name of the AG2 UserProxyAgent.
        :param code_execution: Whether to enable code execution in UserProxyAgent.
        """
        self.model = model
        self.system_message = system_message
        self.api_key_env_var = api_key_env_var
        self.api_type = api_type
        self.max_consecutive_auto_reply = max_consecutive_auto_reply
        self.human_input_mode = human_input_mode
        self.assistant_name = assistant_name
        self.user_proxy_name = user_proxy_name
        self.code_execution = code_execution

    @component.output_types(reply=str, messages=list[dict[str, Any]])
    def run(self, query: str) -> dict[str, Any]:
        """
        Run the AG2 multi-agent conversation with the given query.

        :param query: The input query to send to the AG2 agents.
        :returns: A dictionary with:
            - `reply`: The final assistant reply as a string.
            - `messages`: The full conversation history as a list of message dicts.
        """
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            msg = (
                f"Environment variable '{self.api_key_env_var}' is not set. "
                "Please set it with your API key."
            )
            raise ValueError(msg)

        # Create AG2 LLMConfig — positional argument, NOT keyword
        llm_config = LLMConfig(
            {
                "model": self.model,
                "api_key": api_key,
                "api_type": self.api_type,
            }
        )

        # Create AG2 agents — llm_config as parameter, NOT context manager
        assistant = AssistantAgent(
            name=self.assistant_name,
            system_message=self.system_message,
            llm_config=llm_config,
        )

        code_execution_config = {"use_docker": False} if self.code_execution else False

        user_proxy = UserProxyAgent(
            name=self.user_proxy_name,
            human_input_mode=self.human_input_mode,
            max_consecutive_auto_reply=self.max_consecutive_auto_reply,
            code_execution_config=code_execution_config,
            is_termination_msg=lambda x: (
                x.get("content", "") and "TERMINATE" in x.get("content", "")
            ),
        )

        # Execute conversation — run().process(), NOT initiate_chat()
        user_proxy.run(assistant, message=query).process()

        # Extract messages
        messages = assistant.chat_messages.get(user_proxy, [])

        # Get final reply
        reply = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                content = msg["content"]
                if "TERMINATE" in content:
                    content = content.replace("TERMINATE", "").strip()
                if content:
                    reply = content
                    break

        return {"reply": reply, "messages": messages}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            model=self.model,
            system_message=self.system_message,
            api_key_env_var=self.api_key_env_var,
            api_type=self.api_type,
            max_consecutive_auto_reply=self.max_consecutive_auto_reply,
            human_input_mode=self.human_input_mode,
            assistant_name=self.assistant_name,
            user_proxy_name=self.user_proxy_name,
            code_execution=self.code_execution,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AG2Agent":
        """Deserialize this component from a dictionary."""
        return default_from_dict(cls, data)
