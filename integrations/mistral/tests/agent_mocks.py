# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared mock classes for MistralAgent tests."""

from typing import Optional


class MockUsage:
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20, total_tokens: int = 30):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockMessage:
    def __init__(self, content: str = "", tool_calls: Optional[list] = None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockChoice:
    def __init__(self, message: MockMessage, index: int = 0, finish_reason: str = "stop"):
        self.message = message
        self.index = index
        self.finish_reason = finish_reason


class MockResponse:
    def __init__(self, choices: list, model: str = "agent-model", usage: Optional[MockUsage] = None):
        self.choices = choices
        self.model = model
        self.usage = usage or MockUsage()
