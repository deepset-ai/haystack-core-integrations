# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack_integrations.components.converters.unstructured import UnstructuredFileConverter


class TestUnstructuredFileConverter:
    @pytest.mark.usefixtures("set_env_variables")
    def test_init_default(self):
        converter = UnstructuredFileConverter()
        assert converter.api_url == "https://api.unstructured.io/general/v0/general"
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
            UnstructuredFileConverter(api_url="https://api.unstructured.io/general/v0/general")

    @pytest.mark.usefixtures("set_env_variables")
    def test_to_dict(self):
        converter = UnstructuredFileConverter()
        converter_dict = converter.to_dict()

        assert converter_dict == {
            "type": "haystack_integrations.components.converters.unstructured.converter.UnstructuredFileConverter",
            "init_parameters": {
                "api_url": "https://api.unstructured.io/general/v0/general",
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

    @pytest.mark.integration
    def test_run_one_doc_per_file(self, samples_path):
        pdf_path = samples_path / "sample_pdf.pdf"

        local_converter = UnstructuredFileConverter(
            api_url="http://localhost:8000/general/v0/general", document_creation_mode="one-doc-per-file"
        )

        documents = local_converter.run([pdf_path])["documents"]

        assert len(documents) == 1
        assert documents[0].meta == {"file_path": str(pdf_path)}

    @pytest.mark.integration
    def test_run_one_doc_per_page(self, samples_path):
        pdf_path = samples_path / "sample_pdf.pdf"

        local_converter = UnstructuredFileConverter(
            api_url="http://localhost:8000/general/v0/general", document_creation_mode="one-doc-per-page"
        )

        documents = local_converter.run([pdf_path])["documents"]

        assert len(documents) == 4
        for i, doc in enumerate(documents, start=1):
            assert doc.meta["file_path"] == str(pdf_path)
            assert doc.meta["page_number"] == i

    @pytest.mark.integration
    def test_run_one_doc_per_element(self, samples_path):
        pdf_path = samples_path / "sample_pdf.pdf"

        local_converter = UnstructuredFileConverter(
            api_url="http://localhost:8000/general/v0/general", document_creation_mode="one-doc-per-element"
        )

        documents = local_converter.run([pdf_path])["documents"]

        assert len(documents) > 4
        for doc in documents:
            assert doc.meta["file_path"] == str(pdf_path)
            assert "page_number" in doc.meta

            # elements have a category attribute that is saved in the document meta
            assert "category" in doc.meta

    @pytest.mark.integration
    def test_run_one_doc_per_file_with_meta(self, samples_path):
        pdf_path = samples_path / "sample_pdf.pdf"
        meta = {"custom_meta": "foobar"}
        local_converter = UnstructuredFileConverter(
            api_url="http://localhost:8000/general/v0/general", document_creation_mode="one-doc-per-file"
        )

        documents = local_converter.run(paths=[pdf_path], meta=meta)["documents"]

        assert len(documents) == 1
        assert documents[0].meta["file_path"] == str(pdf_path)
        assert "custom_meta" in documents[0].meta
        assert documents[0].meta["custom_meta"] == "foobar"
        assert documents[0].meta == {"file_path": str(pdf_path), "custom_meta": "foobar"}

    @pytest.mark.integration
    def test_run_one_doc_per_page_with_meta(self, samples_path):
        pdf_path = samples_path / "sample_pdf.pdf"
        meta = {"custom_meta": "foobar"}
        local_converter = UnstructuredFileConverter(
            api_url="http://localhost:8000/general/v0/general", document_creation_mode="one-doc-per-page"
        )

        documents = local_converter.run(paths=[pdf_path], meta=meta)["documents"]
        assert len(documents) == 4
        for i, doc in enumerate(documents, start=1):
            assert doc.meta["file_path"] == str(pdf_path)
            assert doc.meta["page_number"] == i
            assert "custom_meta" in doc.meta
            assert doc.meta["custom_meta"] == "foobar"

    @pytest.mark.integration
    def test_run_one_doc_per_element_with_meta(self, samples_path):
        pdf_path = samples_path / "sample_pdf.pdf"
        meta = {"custom_meta": "foobar"}
        local_converter = UnstructuredFileConverter(
            api_url="http://localhost:8000/general/v0/general", document_creation_mode="one-doc-per-element"
        )

        documents = local_converter.run(paths=[pdf_path], meta=meta)["documents"]

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

    @pytest.mark.integration
    def test_run_one_doc_per_element_with_meta_list_two_files(self, samples_path):
        pdf_path = [samples_path / "sample_pdf.pdf", samples_path / "sample_pdf2.pdf"]
        meta = [
            {"custom_meta": "sample_pdf.pdf", "common_meta": "common"},
            {"custom_meta": "sample_pdf2.pdf", "common_meta": "common"},
        ]
        local_converter = UnstructuredFileConverter(
            api_url="http://localhost:8000/general/v0/general", document_creation_mode="one-doc-per-element"
        )

        documents = local_converter.run(paths=pdf_path, meta=meta)["documents"]

        assert len(documents) > 4
        for doc in documents:
            assert doc.meta["custom_meta"] == doc.meta["filename"]
            assert "file_path" in doc.meta
            assert "page_number" in doc.meta
            # elements have a category attribute that is saved in the document meta
            assert "category" in doc.meta
            assert "common_meta" in doc.meta
            assert doc.meta["common_meta"] == "common"

    @pytest.mark.integration
    def test_run_one_doc_per_element_with_meta_list_folder_fail(self, samples_path):
        pdf_path = [samples_path]
        meta = [{"custom_meta": "foobar", "common_meta": "common"}, {"other_meta": "barfoo", "common_meta": "common"}]
        local_converter = UnstructuredFileConverter(
            api_url="http://localhost:8000/general/v0/general", document_creation_mode="one-doc-per-element"
        )
        with pytest.raises(ValueError):
            local_converter.run(paths=pdf_path, meta=meta)["documents"]

    @pytest.mark.integration
    def test_run_one_doc_per_element_with_meta_list_folder(self, samples_path):
        pdf_path = [samples_path]
        meta = {"common_meta": "common"}

        local_converter = UnstructuredFileConverter(
            api_url="http://localhost:8000/general/v0/general", document_creation_mode="one-doc-per-element"
        )

        documents = local_converter.run(paths=pdf_path, meta=meta)["documents"]

        assert len(documents) > 4
        for doc in documents:
            assert "file_path" in doc.meta
            assert "page_number" in doc.meta
            # elements have a category attribute that is saved in the document meta
            assert "category" in doc.meta
            assert "common_meta" in doc.meta
            assert doc.meta["common_meta"] == "common"
