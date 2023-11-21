from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from haystack.preview import Document

from unstructured_fileconverter_haystack import UnstructuredFileConverter


class TestUnstructuredFileConverter:
    @pytest.mark.unit
    def test_init_default(self):
        converter = UnstructuredFileConverter(api_key="test-api-key")
        assert converter.api_url == "https://api.unstructured.io/general/v0/general"
        assert converter.api_key == "test-api-key"
        assert converter.document_creation_mode == "one-doc-per-file"
        assert converter.separator == "\n\n"
        assert converter.unstructured_kwargs == {}

    @pytest.mark.unit
    def test_init_with_parameters(self):
        converter = UnstructuredFileConverter(api_url="http://custom-url:8000/general",
        document_creation_mode="one-doc-per-element", separator="|", unstructured_kwargs={"foo": "bar"})
        assert converter.api_url == "http://custom-url:8000/general"
        assert converter.api_key is None
        assert converter.document_creation_mode == "one-doc-per-element"
        assert converter.separator == "|"
        assert converter.unstructured_kwargs == {"foo": "bar"}

    @pytest.mark.unit
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
            },
        }