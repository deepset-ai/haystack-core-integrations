# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for MistralAgent tests."""

import pytest
from haystack.dataclasses import ChatMessage

from tests.agent_mocks import MockChoice, MockMessage, MockResponse, MockUsage


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France?"),
    ]


@pytest.fixture
def mock_sdk_response():
    return MockResponse(
        choices=[
            MockChoice(
                message=MockMessage(content="The capital of France is Paris."),
                finish_reason="stop",
                index=0,
            )
        ],
        model="mistral-agent-model",
        usage=MockUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40),
    )
