# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import logging
from haystack.components.agents.state import State
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole

from haystack_integrations.agent_pack.advanced_rag import prompts

logger = logging.getLogger(__name__)


class BackupAnswerHook:
    """
    Produce a final answer when the agent run ends without one. Runs as an `after_run` hook.

    When the agent exhausts `max_agent_steps` mid-investigation, the run ends on a tool call or tool result instead of
    an assistant text answer (and only `after_run` hooks run in this situation). This hook detects that case and makes
    one LLM call over the conversation so far to produce a best-effort answer from the already-gathered evidence.
    """

    allowed_hook_points = ("after_run",)

    def __init__(self, generator: ChatGenerator) -> None:
        """
        Create the hook.

        :param generator: LLM that writes the backup answer from the gathered evidence.
        """
        self.generator = generator

    def warm_up(self) -> None:
        """Prepare the hook's generator for use; called from the Agent's `warm_up`."""
        if hasattr(self.generator, "warm_up"):
            self.generator.warm_up()

    def close(self) -> None:
        """Release the hook's generator resources; called from the Agent's `close`."""
        if hasattr(self.generator, "close"):
            self.generator.close()

    def to_dict(self) -> dict:
        """
        Serialize the hook to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(self, generator=self.generator)

    @classmethod
    def from_dict(cls, data: dict) -> "BackupAnswerHook":
        """
        Deserialize the hook from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized hook.
        """
        return default_from_dict(cls, data)

    @staticmethod
    def _needs_backup(messages: list[ChatMessage]) -> bool:
        """
        Whether the run ended without a final assistant text answer.

        :param messages: The run's messages.
        :returns: True when the last message is not a plain assistant text reply.
        """
        if not messages:
            return False
        last = messages[-1]
        return not (last.is_from(ChatRole.ASSISTANT) and bool(last.text) and not last.tool_calls)

    def run(self, state: State) -> None:
        """
        Append a best-effort final answer when the run ended without one (e.g. step exhaustion).

        :param state: The agent run's state.
        """
        messages = state.data.get("messages") or []
        if not self._needs_backup(messages):
            return
        logger.info("run ended without a final answer (likely max_agent_steps); writing a backup answer")
        transcript = [m for m in messages if not m.is_from(ChatRole.SYSTEM)]
        prompt = [ChatMessage.from_system(prompts.BACKUP_ANSWER_PROMPT), *transcript]
        reply = self.generator.run(messages=prompt)["replies"][0]
        state.set("messages", [reply])  # merge_lists handler: appended as the final message
