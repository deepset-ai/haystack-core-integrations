# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack import Document

from haystack_integrations.components.connectors.github.utils import encode_path, encode_ref, validate_repo_part
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


class TestEncodePath:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("README.md", "README.md"),
            ("src/app.py", "src/app.py"),
            ("/docs/", "docs"),
            ("docs/a b#1?.md", "docs/a%20b%231%3F.md"),
            ("%2e%2e/x", "%252e%252e/x"),
        ],
    )
    def test_encode_path(self, path, expected):
        assert encode_path(path) == expected

    @pytest.mark.parametrize("path", ["..", "../x", "a/../b", "./x", "a/.", "a//b", "", "/"])
    def test_encode_path_rejects_invalid_path(self, path):
        with pytest.raises(ValueError, match="Invalid path"):
            encode_path(path)

    @pytest.mark.parametrize("path", ["", "/"])
    def test_encode_path_allow_empty(self, path):
        assert encode_path(path, allow_empty=True) == ""


class TestEncodeRef:
    @pytest.mark.parametrize("ref,expected", [("main", "main"), ("feature/x", "feature/x"), ("fix/#1", "fix/%231")])
    def test_encode_ref(self, ref, expected):
        assert encode_ref(ref) == expected

    @pytest.mark.parametrize("ref", ["", "..", "../x", "a/../b", "./x", "/main", "main/", "a//b"])
    def test_encode_ref_rejects_invalid_ref(self, ref):
        with pytest.raises(ValueError, match="Invalid branch"):
            encode_ref(ref)


class TestValidateRepoPart:
    @pytest.mark.parametrize("name", ["owner", "my-repo", "repo.js", "under_score", "..."])
    def test_validate_repo_part(self, name):
        validate_repo_part(name)

    # "repo\n" would pass a `^...$` regex, as `$` also matches before a trailing newline
    @pytest.mark.parametrize("name", ["", ".", "..", "a/b", "repo?x", "repo#x", "re po", "repo\n"])
    def test_validate_repo_part_rejects_invalid_name(self, name):
        with pytest.raises(ValueError, match="Invalid repository owner or name"):
            validate_repo_part(name)
