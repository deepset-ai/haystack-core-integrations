from pathlib import Path

import pytest

from unstructured_fileconverter_haystack import UnstructuredFileConverter


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "samples"


class TestUnstructuredFileConverter:
    def test_init_default(self):
        converter = UnstructuredFileConverter(api_key="test-api-key")
        assert converter.api_url == "https://api.unstructured.io/general/v0/general"
        assert converter.api_key == "test-api-key"
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
        assert converter.api_key is None
        assert converter.document_creation_mode == "one-doc-per-element"
        assert converter.separator == "|"
        assert converter.unstructured_kwargs == {"foo": "bar"}
        assert not converter.progress_bar

    def test_to_dict(self):
        converter = UnstructuredFileConverter(api_key="test-api-key")
        converter_dict = converter.to_dict()

        assert converter_dict == {
            "type": "unstructured_fileconverter_haystack.fileconverter.UnstructuredFileConverter",
            "init_parameters": {
                "api_url": "https://api.unstructured.io/general/v0/general",
                "document_creation_mode": "one-doc-per-file",
                "separator": "\n\n",
                "unstructured_kwargs": {},
                "progress_bar": True,
            },
        }

    @pytest.mark.integration
    def test_run_one_doc_per_file(self, samples_path):
        pdf_path = samples_path / "sample_pdf.pdf"

        local_converter = UnstructuredFileConverter(
            api_url="http://localhost:8000/general/v0/general", document_creation_mode="one-doc-per-file"
        )

        documents = local_converter.run([pdf_path])["documents"]

        assert len(documents) == 1
        assert documents[0].meta == {"name": str(pdf_path)}

    @pytest.mark.integration
    def test_run_one_doc_per_page(self, samples_path):
        pdf_path = samples_path / "sample_pdf.pdf"

        local_converter = UnstructuredFileConverter(
            api_url="http://localhost:8000/general/v0/general", document_creation_mode="one-doc-per-page"
        )

        documents = local_converter.run([pdf_path])["documents"]

        assert len(documents) == 4
        for i, doc in enumerate(documents, start=1):
            assert doc.meta["name"] == str(pdf_path)
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
            assert doc.meta["name"] == str(pdf_path)
            assert "page_number" in doc.meta

            # elements have a category attribute that is saved in the document meta
            assert "category" in doc.meta
