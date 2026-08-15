# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import patch

import pytest
from haystack.utils import Secret
from unstructured.documents.elements import ElementMetadata, Text, Title

from haystack_integrations.components.converters.unstructured import UnstructuredFileConverter

CONVERTER_MODULE = "haystack_integrations.components.converters.unstructured.converter"

LOCAL_API_URL = "http://localhost:8000/general/v0/general"


def _element(text: str, **metadata) -> Text:
    """Build a real Unstructured element, so the fakes stay honest against the SDK."""
    return Text(text, metadata=ElementMetadata(**metadata))


@pytest.fixture
def converter() -> UnstructuredFileConverter:
    """A converter pointing at a local API, so no API key is required."""
    return UnstructuredFileConverter(api_url=LOCAL_API_URL)


class TestCreateDocuments:
    """`_create_documents` is a pure static method: elements in, Haystack Documents out."""

    def test_one_doc_per_file_joins_all_elements_with_the_separator(self):
        docs = UnstructuredFileConverter._create_documents(
            filepath=Path("a/file.pdf"),
            elements=[_element("first"), _element("second"), _element("third")],
            document_creation_mode="one-doc-per-file",
            separator="|",
            meta={"key": "value"},
        )

        assert len(docs) == 1
        assert docs[0].content == "first|second|third"
        assert docs[0].meta == {"key": "value", "file_path": "a/file.pdf"}

    def test_one_doc_per_page_groups_elements_by_page_number(self):
        docs = UnstructuredFileConverter._create_documents(
            filepath=Path("a/file.pdf"),
            elements=[
                _element("page one, first", page_number=1),
                _element("page two", page_number=2),
                _element("page one, second", page_number=1),
            ],
            document_creation_mode="one-doc-per-page",
            separator="|",
            meta={},
        )

        assert len(docs) == 2
        assert docs[0].content == "page one, first|page one, second|"
        assert docs[0].meta["page_number"] == 1
        assert docs[1].content == "page two|"
        assert docs[1].meta["page_number"] == 2

    def test_one_doc_per_element_records_the_index_and_category_of_each_element(self):
        docs = UnstructuredFileConverter._create_documents(
            filepath=Path("a/file.pdf"),
            elements=[Text("body text"), Title("A Heading")],
            document_creation_mode="one-doc-per-element",
            separator="\n\n",
            meta={"key": "value"},
        )

        assert len(docs) == 2
        assert docs[0].content == "body text"
        assert docs[0].meta["element_index"] == 0
        assert docs[0].meta["category"] == "UncategorizedText"
        assert docs[1].content == "A Heading"
        assert docs[1].meta["element_index"] == 1
        assert docs[1].meta["category"] == "Title"
        assert all(doc.meta["key"] == "value" for doc in docs)

    @pytest.mark.parametrize("document_creation_mode", ["one-doc-per-file", "one-doc-per-page", "one-doc-per-element"])
    def test_the_caller_metadata_is_never_mutated(self, document_creation_mode):
        meta = {"key": "value"}

        UnstructuredFileConverter._create_documents(
            filepath=Path("a/file.pdf"),
            elements=[_element("text", page_number=1)],
            document_creation_mode=document_creation_mode,
            separator="\n\n",
            meta=meta,
        )

        assert meta == {"key": "value"}


class TestPartitionFileIntoElements:
    def test_forwards_the_api_settings_and_the_extra_kwargs(self):
        converter = UnstructuredFileConverter(
            api_url=LOCAL_API_URL,
            api_key=Secret.from_token("secret-key"),
            unstructured_kwargs={"strategy": "hi_res"},
        )

        with patch(f"{CONVERTER_MODULE}.partition_via_api") as mock_partition:
            mock_partition.return_value = [_element("text")]
            elements = converter._partition_file_into_elements(filepath=Path("a/file.pdf"))

        assert len(elements) == 1
        mock_partition.assert_called_once_with(
            filename="a/file.pdf",
            api_url=LOCAL_API_URL,
            api_key="secret-key",
            strategy="hi_res",
        )

    def test_returns_no_elements_and_warns_when_the_api_call_fails(self, converter, caplog):
        with patch(f"{CONVERTER_MODULE}.partition_via_api") as mock_partition:
            mock_partition.side_effect = RuntimeError("API is down")
            elements = converter._partition_file_into_elements(filepath=Path("a/file.pdf"))

        assert elements == []
        assert "a/file.pdf" in caplog.text
        assert "API is down" in caplog.text


class TestRun:
    def test_converts_every_file_in_a_directory_and_ignores_subdirectories(self, converter, tmp_path):
        (tmp_path / "first.txt").write_text("first")
        (tmp_path / "second.txt").write_text("second")
        (tmp_path / "nested.dir").mkdir()

        with patch(f"{CONVERTER_MODULE}.partition_via_api") as mock_partition:
            mock_partition.return_value = [_element("text")]
            documents = converter.run(paths=[tmp_path])["documents"]

        assert len(documents) == 2
        assert mock_partition.call_count == 2

    def test_zips_a_metadata_list_with_the_given_file_paths(self, converter, tmp_path):
        first = tmp_path / "first.txt"
        first.write_text("first")
        second = tmp_path / "second.txt"
        second.write_text("second")

        with patch(f"{CONVERTER_MODULE}.partition_via_api") as mock_partition:
            mock_partition.return_value = [_element("text")]
            documents = converter.run(paths=[first, second], meta=[{"source": "one"}, {"source": "two"}])["documents"]

        assert [doc.meta["source"] for doc in documents] == ["one", "two"]

    def test_rejects_a_metadata_list_when_paths_contain_a_directory(self, converter, tmp_path):
        (tmp_path / "first.txt").write_text("first")

        with pytest.raises(ValueError, match="`meta` can only be a dictionary"):
            converter.run(paths=[tmp_path], meta=[{"source": "one"}])

    def test_rejects_a_metadata_list_whose_length_does_not_match_the_paths(self, converter, tmp_path):
        first = tmp_path / "first.txt"
        first.write_text("first")

        with pytest.raises(ValueError):
            converter.run(paths=[first], meta=[{"source": "one"}, {"source": "two"}])


class TestSerializationWithoutApiKey:
    def test_to_dict_and_from_dict_round_trip_a_none_api_key(self):
        converter = UnstructuredFileConverter(api_url=LOCAL_API_URL, api_key=None)

        converter_dict = converter.to_dict()
        assert converter_dict["init_parameters"]["api_key"] is None

        deserialized = UnstructuredFileConverter.from_dict(converter_dict)
        assert deserialized.api_key is None
        assert deserialized.api_url == LOCAL_API_URL
