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

HOSTED_API_URL = "https://api.unstructured.io/general/v0/general"


def _element(text: str, **metadata) -> Text:
    """Build a real Unstructured element, so the fakes stay honest against the SDK."""
    return Text(text, metadata=ElementMetadata(**metadata))


class TestInit:
    @pytest.mark.usefixtures("set_env_variables")
    def test_init_default(self):
        converter = UnstructuredFileConverter()
        assert converter.api_url == HOSTED_API_URL
        assert converter.api_key.resolve_value() == "test-api-key"
        assert converter.document_creation_mode == "one-doc-per-file"
        assert converter.separator == "\n\n"
        assert converter.unstructured_kwargs == {}
        assert converter.progress_bar

    def test_init_with_parameters(self):
        converter = UnstructuredFileConverter(
            api_url="http://custom-url:8000/general",
            document_creation_mode="one-doc-per-element",
            separator="|",
            unstructured_kwargs={"foo": "bar"},
            progress_bar=False,
        )
        assert converter.api_url == "http://custom-url:8000/general"
        assert converter.api_key.resolve_value() is None
        assert converter.document_creation_mode == "one-doc-per-element"
        assert converter.separator == "|"
        assert converter.unstructured_kwargs == {"foo": "bar"}
        assert not converter.progress_bar

    def test_init_hosted_without_api_key_raises_error(self):
        with pytest.raises(ValueError):
            UnstructuredFileConverter(api_url=HOSTED_API_URL)


class TestSerde:
    @pytest.mark.usefixtures("set_env_variables")
    def test_to_dict(self):
        converter = UnstructuredFileConverter()
        converter_dict = converter.to_dict()

        assert converter_dict == {
            "type": "haystack_integrations.components.converters.unstructured.converter.UnstructuredFileConverter",
            "init_parameters": {
                "api_url": HOSTED_API_URL,
                "api_key": {"env_vars": ["UNSTRUCTURED_API_KEY"], "strict": False, "type": "env_var"},
                "document_creation_mode": "one-doc-per-file",
                "separator": "\n\n",
                "unstructured_kwargs": {},
                "progress_bar": True,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("UNSTRUCTURED_API_KEY", "test-api-key")
        converter_dict = {
            "type": "haystack_integrations.components.converters.unstructured.converter.UnstructuredFileConverter",
            "init_parameters": {
                "api_url": "http://custom-url:8000/general",
                "api_key": {"env_vars": ["UNSTRUCTURED_API_KEY"], "strict": False, "type": "env_var"},
                "document_creation_mode": "one-doc-per-element",
                "separator": "|",
                "unstructured_kwargs": {"foo": "bar"},
                "progress_bar": False,
            },
        }
        converter = UnstructuredFileConverter.from_dict(converter_dict)
        assert converter.api_url == "http://custom-url:8000/general"
        assert converter.api_key.resolve_value() == "test-api-key"
        assert converter.document_creation_mode == "one-doc-per-element"
        assert converter.separator == "|"
        assert converter.unstructured_kwargs == {"foo": "bar"}
        assert not converter.progress_bar

    def test_to_dict_and_from_dict_round_trip_a_none_api_key(self, local_api_url):
        converter = UnstructuredFileConverter(api_url=local_api_url, api_key=None)

        converter_dict = converter.to_dict()
        assert converter_dict["init_parameters"]["api_key"] is None

        deserialized = UnstructuredFileConverter.from_dict(converter_dict)
        assert deserialized.api_key is None
        assert deserialized.api_url == local_api_url


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
    def test_forwards_the_api_settings_and_the_extra_kwargs(self, local_api_url):
        converter = UnstructuredFileConverter(
            api_url=local_api_url,
            api_key=Secret.from_token("secret-key"),
            unstructured_kwargs={"strategy": "hi_res"},
        )

        with patch(f"{CONVERTER_MODULE}.partition_via_api") as mock_partition:
            mock_partition.return_value = [_element("text")]
            elements = converter._partition_file_into_elements(filepath=Path("a/file.pdf"))

        assert len(elements) == 1
        mock_partition.assert_called_once_with(
            filename="a/file.pdf",
            api_url=local_api_url,
            api_key="secret-key",
            strategy="hi_res",
        )

    def test_returns_no_elements_and_warns_when_the_api_call_fails(self, local_converter, caplog):
        with patch(f"{CONVERTER_MODULE}.partition_via_api") as mock_partition:
            mock_partition.side_effect = RuntimeError("API is down")
            elements = local_converter._partition_file_into_elements(filepath=Path("a/file.pdf"))

        assert elements == []
        assert "a/file.pdf" in caplog.text
        assert "API is down" in caplog.text


class TestRun:
    def test_converts_every_file_in_a_directory_and_ignores_subdirectories(self, local_converter, tmp_path):
        (tmp_path / "first.txt").write_text("first")
        (tmp_path / "second.txt").write_text("second")
        # the directory is globbed non-recursively, so neither the subdirectory itself
        # nor the file inside it is converted
        nested = tmp_path / "nested.dir"
        nested.mkdir()
        (nested / "inner.txt").write_text("inner")

        with patch(f"{CONVERTER_MODULE}.partition_via_api") as mock_partition:
            mock_partition.return_value = [_element("text")]
            documents = local_converter.run(paths=[tmp_path])["documents"]

        assert len(documents) == 2
        assert mock_partition.call_count == 2

    def test_zips_a_metadata_list_with_the_given_file_paths(self, local_converter, tmp_path):
        first = tmp_path / "first.txt"
        first.write_text("first")
        second = tmp_path / "second.txt"
        second.write_text("second")

        with patch(f"{CONVERTER_MODULE}.partition_via_api") as mock_partition:
            mock_partition.return_value = [_element("text")]
            documents = local_converter.run(paths=[first, second], meta=[{"source": "one"}, {"source": "two"}])[
                "documents"
            ]

        assert [doc.meta["source"] for doc in documents] == ["one", "two"]

    def test_rejects_a_metadata_list_when_paths_contain_a_directory(self, local_converter, tmp_path):
        (tmp_path / "first.txt").write_text("first")

        with pytest.raises(ValueError, match="`meta` can only be a dictionary"):
            local_converter.run(paths=[tmp_path], meta=[{"source": "one"}])

    def test_rejects_a_metadata_list_whose_length_does_not_match_the_paths(self, local_converter, tmp_path):
        first = tmp_path / "first.txt"
        first.write_text("first")

        with pytest.raises(ValueError, match="length of the metadata list"):
            local_converter.run(paths=[first], meta=[{"source": "one"}, {"source": "two"}])


@pytest.mark.integration
class TestRunIntegration:
    """These tests need an Unstructured API running locally, see the README."""

    def test_run_one_doc_per_file(self, samples_path, local_api_url):
        pdf_path = samples_path / "sample_pdf.pdf"

        converter = UnstructuredFileConverter(api_url=local_api_url, document_creation_mode="one-doc-per-file")

        documents = converter.run([pdf_path])["documents"]

        assert len(documents) == 1
        assert documents[0].meta == {"file_path": str(pdf_path)}

    def test_run_one_doc_per_page(self, samples_path, local_api_url):
        pdf_path = samples_path / "sample_pdf.pdf"

        converter = UnstructuredFileConverter(api_url=local_api_url, document_creation_mode="one-doc-per-page")

        documents = converter.run([pdf_path])["documents"]

        assert len(documents) == 4
        for i, doc in enumerate(documents, start=1):
            assert doc.meta["file_path"] == str(pdf_path)
            assert doc.meta["page_number"] == i

    def test_run_one_doc_per_element(self, samples_path, local_api_url):
        pdf_path = samples_path / "sample_pdf.pdf"

        converter = UnstructuredFileConverter(api_url=local_api_url, document_creation_mode="one-doc-per-element")

        documents = converter.run([pdf_path])["documents"]

        assert len(documents) > 4
        for doc in documents:
            assert doc.meta["file_path"] == str(pdf_path)
            assert "page_number" in doc.meta

            # elements have a category attribute that is saved in the document meta
            assert "category" in doc.meta

    def test_run_one_doc_per_file_with_meta(self, samples_path, local_api_url):
        pdf_path = samples_path / "sample_pdf.pdf"
        meta = {"custom_meta": "foobar"}
        converter = UnstructuredFileConverter(api_url=local_api_url, document_creation_mode="one-doc-per-file")

        documents = converter.run(paths=[pdf_path], meta=meta)["documents"]

        assert len(documents) == 1
        assert documents[0].meta["file_path"] == str(pdf_path)
        assert "custom_meta" in documents[0].meta
        assert documents[0].meta["custom_meta"] == "foobar"
        assert documents[0].meta == {"file_path": str(pdf_path), "custom_meta": "foobar"}

    def test_run_one_doc_per_page_with_meta(self, samples_path, local_api_url):
        pdf_path = samples_path / "sample_pdf.pdf"
        meta = {"custom_meta": "foobar"}
        converter = UnstructuredFileConverter(api_url=local_api_url, document_creation_mode="one-doc-per-page")

        documents = converter.run(paths=[pdf_path], meta=meta)["documents"]
        assert len(documents) == 4
        for i, doc in enumerate(documents, start=1):
            assert doc.meta["file_path"] == str(pdf_path)
            assert doc.meta["page_number"] == i
            assert "custom_meta" in doc.meta
            assert doc.meta["custom_meta"] == "foobar"

    def test_run_one_doc_per_element_with_meta(self, samples_path, local_api_url):
        pdf_path = samples_path / "sample_pdf.pdf"
        meta = {"custom_meta": "foobar"}
        converter = UnstructuredFileConverter(api_url=local_api_url, document_creation_mode="one-doc-per-element")

        documents = converter.run(paths=[pdf_path], meta=meta)["documents"]

        assert len(documents) > 4
        first_element_index = 0
        for doc in documents:
            assert doc.meta["file_path"] == str(pdf_path)
            assert "page_number" in doc.meta

            # elements have a category attribute that is saved in the document meta
            assert "category" in doc.meta
            assert "custom_meta" in doc.meta
            assert doc.meta["custom_meta"] == "foobar"
            assert doc.meta["element_index"] == first_element_index
            first_element_index += 1

    def test_run_one_doc_per_element_with_meta_list_two_files(self, samples_path, local_api_url):
        pdf_path = [samples_path / "sample_pdf.pdf", samples_path / "sample_pdf2.pdf"]
        meta = [
            {"custom_meta": "sample_pdf.pdf", "common_meta": "common"},
            {"custom_meta": "sample_pdf2.pdf", "common_meta": "common"},
        ]
        converter = UnstructuredFileConverter(api_url=local_api_url, document_creation_mode="one-doc-per-element")

        documents = converter.run(paths=pdf_path, meta=meta)["documents"]

        assert len(documents) > 4
        for doc in documents:
            assert doc.meta["custom_meta"] == doc.meta["filename"]
            assert "file_path" in doc.meta
            assert "page_number" in doc.meta
            # elements have a category attribute that is saved in the document meta
            assert "category" in doc.meta
            assert "common_meta" in doc.meta
            assert doc.meta["common_meta"] == "common"

    def test_run_one_doc_per_element_with_meta_list_folder_fail(self, samples_path, local_api_url):
        pdf_path = [samples_path]
        meta = [{"custom_meta": "foobar", "common_meta": "common"}, {"other_meta": "barfoo", "common_meta": "common"}]
        converter = UnstructuredFileConverter(api_url=local_api_url, document_creation_mode="one-doc-per-element")
        with pytest.raises(ValueError, match="`meta` can only be a dictionary"):
            converter.run(paths=pdf_path, meta=meta)["documents"]

    def test_run_one_doc_per_element_with_meta_list_folder(self, samples_path, local_api_url):
        pdf_path = [samples_path]
        meta = {"common_meta": "common"}

        converter = UnstructuredFileConverter(api_url=local_api_url, document_creation_mode="one-doc-per-element")

        documents = converter.run(paths=pdf_path, meta=meta)["documents"]

        assert len(documents) > 4
        for doc in documents:
            assert "file_path" in doc.meta
            assert "page_number" in doc.meta
            # elements have a category attribute that is saved in the document meta
            assert "category" in doc.meta
            assert "common_meta" in doc.meta
            assert doc.meta["common_meta"] == "common"
