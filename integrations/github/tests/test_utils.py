# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack import Document

from haystack_integrations.tools.github.utils import message_handler, serialize_handlers


class TestMessageHandler:
    @pytest.mark.parametrize(
        "documents,expected",
        [
            (
                [Document(content="error message", meta={"type": "error"})],
                "error message",
            ),
            (
                [
                    Document(content="docs", meta={"type": "dir"}),
                    Document(content="README.md", meta={"type": "file"}),
                ],
                "docs\nREADME.md",
            ),
            (
                [Document(content="print('hi')", meta={"type": "file_content", "path": "main.py"})],
                "File Content for main.py\n\nprint('hi')",
            ),
        ],
    )
    def test_message_handler_renders_documents(self, documents, expected):
        assert message_handler(documents) == expected

    def test_message_handler_truncates_to_max_length(self):
        big_content = "x" * 200
        doc = Document(content=big_content, meta={"type": "file_content", "path": "main.py"})
        result = message_handler([doc], max_length=50)
        assert result.endswith("...(large file can't be fully displayed)")
        assert len(result) == 50 + len("...(large file can't be fully displayed)")


class TestSerializeHandlers:
    @pytest.mark.parametrize(
        "outputs_to_state,outputs_to_string,error_match",
        [
            (
                {"documents": {"handler": "not_callable"}},
                None,
                "outputs_to_state\\[documents\\] is not a callable",
            ),
            (
                None,
                {"handler": "not_callable"},
                "outputs_to_string is not a callable",
            ),
        ],
    )
    def test_serialize_handlers_raises_when_handler_not_callable(
        self, outputs_to_state, outputs_to_string, error_match
    ):
        serialized: dict = {}
        with pytest.raises(ValueError, match=error_match):
            serialize_handlers(serialized, outputs_to_state, outputs_to_string)
