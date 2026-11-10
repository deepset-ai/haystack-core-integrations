# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

# The integration tests call the gateway started by docker-compose.yml unless OTARI_API_BASE_URL names another one
OTARI_API_BASE_URL = os.environ.get("OTARI_API_BASE_URL", "http://localhost:8000/api/v1")

requires_api_key = pytest.mark.skipif(
    not os.environ.get("OTARI_API_KEY", None),
    reason="Export an env var called OTARI_API_KEY containing an Otari API key to run this test.",
)


@pytest.fixture(autouse=True)
def allow_deserialization_of_test_modules(monkeypatch):
    """
    haystack-ai >= 3.0 refuses to deserialize classes and callables from modules outside its
    trusted-module allowlist. Tools and callbacks defined in the test modules live outside that
    allowlist, so trust them explicitly.
    """
    monkeypatch.setenv("HAYSTACK_DESERIALIZATION_ALLOWLIST", "tests,test_*")


def weather(city: str):
    """Get weather for a given city."""
    return f"The weather in {city} is sunny and 32°C"


@pytest.fixture
def tools():
    return [
        Tool(
            name="weather",
            description="useful to determine the weather of a given location",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=weather,
        )
    ]


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]
