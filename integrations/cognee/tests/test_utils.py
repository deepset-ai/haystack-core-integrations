# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

from haystack_integrations.components.connectors.cognee._utils import extract_text, run_sync


class TestExtractText:
    def test_string_input(self):
        assert extract_text("hello world") == "hello world"

    def test_object_with_content_attr(self):
        class FakeResult:
            content = "object content"

        assert extract_text(FakeResult()) == "object content"

    def test_object_with_text_attr(self):
        class FakeResult:
            text = "object text"

        assert extract_text(FakeResult()) == "object text"

    def test_object_with_description_attr(self):
        class FakeResult:
            description = "object description"

        assert extract_text(FakeResult()) == "object description"

    def test_object_with_name_attr(self):
        class FakeResult:
            name = "object name"

        assert extract_text(FakeResult()) == "object name"

    def test_object_attr_priority(self):
        class FakeResult:
            content = "first"
            text = "second"

        assert extract_text(FakeResult()) == "first"

    def test_object_skips_none_attr(self):
        class FakeResult:
            content = None
            text = "fallback text"

        assert extract_text(FakeResult()) == "fallback text"

    def test_object_skips_non_string_attr(self):
        class FakeResult:
            content = 123
            text = "string value"

        assert extract_text(FakeResult()) == "string value"

    def test_dict_with_content_key(self):
        assert extract_text({"content": "dict content"}) == "dict content"

    def test_dict_with_text_key(self):
        assert extract_text({"text": "dict text"}) == "dict text"

    def test_dict_with_description_key(self):
        assert extract_text({"description": "dict description"}) == "dict description"

    def test_dict_with_name_key(self):
        assert extract_text({"name": "dict name"}) == "dict name"

    def test_dict_key_priority(self):
        assert extract_text({"content": "first", "text": "second"}) == "first"

    def test_dict_skips_non_string_value(self):
        assert extract_text({"content": 123, "text": "fallback"}) == "fallback"

    def test_fallback_to_str(self):
        assert extract_text(42) == "42"

    def test_fallback_dict_no_known_keys(self):
        result = extract_text({"unknown": "value"})
        assert "unknown" in result


class TestRunSync:
    def test_run_simple_coroutine(self):
        async def coro():
            return 42

        result = run_sync(coro())
        assert result == 42

    def test_run_async_coroutine(self):
        async def coro():
            await asyncio.sleep(0)
            return "done"

        result = run_sync(coro())
        assert result == "done"
