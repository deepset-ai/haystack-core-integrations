# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The SCOPE and WRITE stages, as serializable Agent hooks."""

from haystack import logging
from haystack.components.agents.state import State
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

logger = logging.getLogger(__name__)


class ScopeHook:
    """Turn the user query into a research brief, once. Runs as a `before_run` hook."""

    allowed_hook_points = ("before_run",)

    def __init__(self, generator: ChatGenerator, prompt_builder: ChatPromptBuilder) -> None:
        self.generator = generator
        self.prompt_builder = prompt_builder

    def to_dict(self) -> dict:
        """
        Serialize the hook to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(self, generator=self.generator, prompt_builder=self.prompt_builder)

    @classmethod
    def from_dict(cls, data: dict) -> "ScopeHook":
        """
        Deserialize the hook from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized hook.
        """
        return default_from_dict(cls, data)

    def run(self, state: State) -> None:
        """Rewrite the query into a research brief and seed it as the sole user message."""
        prompt = self.prompt_builder.run(query=state.data["messages"][-1].text)["prompt"]
        brief = self.generator.run(messages=prompt)["replies"][0].text
        logger.info("brief:\n{brief}", brief=brief)
        state.set("brief", brief)
        state.set("messages", [ChatMessage.from_user(brief)])


class WriteHook:
    """Write the final report from the brief and notes, once. Runs as an `after_run` hook."""

    allowed_hook_points = ("after_run",)

    def __init__(self, generator: ChatGenerator, prompt_builder: ChatPromptBuilder) -> None:
        self.generator = generator
        self.prompt_builder = prompt_builder

    def to_dict(self) -> dict:
        """
        Serialize the hook to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(self, generator=self.generator, prompt_builder=self.prompt_builder)

    @classmethod
    def from_dict(cls, data: dict) -> "WriteHook":
        """
        Deserialize the hook from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized hook.
        """
        return default_from_dict(cls, data)

    def run(self, state: State) -> None:
        """Write the final report from the brief and collected notes."""
        subtopics = [
            tc.arguments["messages"][0]["content"]
            for m in state.data.get("messages", [])
            for tc in (m.tool_calls or [])
            if tc.tool_name == "research_subtopic"
        ]
        for i, subtopic in enumerate(subtopics, 1):
            logger.info("subtopic #{index} delegated: {subtopic}", index=i, subtopic=subtopic)
        notes = [n for n in state.get("notes", []) if n]
        prompt = self.prompt_builder.run(replies=[ChatMessage.from_assistant(state.get("brief", ""))], notes=notes)[
            "prompt"
        ]
        report = self.generator.run(messages=prompt)["replies"][0].text
        state.set("report", report)
        state.set("messages", [ChatMessage.from_assistant(report)])
