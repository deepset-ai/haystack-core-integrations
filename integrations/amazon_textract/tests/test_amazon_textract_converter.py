# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import BotoCoreError, ClientError
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.components.converters.amazon_textract import AmazonTextractConverter
from haystack_integrations.components.converters.amazon_textract.errors import (
    AmazonTextractConfigurationError,
)

TEST_FILES_DIR = Path(__file__).parent / "test_files"


def _make_textract_response(lines=None, page_count=1):
    """Helper to build a mock Textract response dict."""
    blocks = [{"BlockType": "PAGE", "Id": f"page-{i}"} for i in range(page_count)]
    for text in lines or []:
        blocks.append(
            {
                "BlockType": "LINE",
                "Id": f"line-{len(blocks)}",
                "Text": text,
                "Confidence": 99.5,
            }
        )
    return {"Blocks": blocks, "ResponseMetadata": {"HTTPStatusCode": 200}}


class TestAmazonTextractConverterInit:
    def test_init_default(self):
        converter = AmazonTextractConverter(
            aws_access_key_id=Secret.from_token("fake_id"),
            aws_secret_access_key=Secret.from_token("fake_secret"),
            aws_region_name=Secret.from_token("us-east-1"),
        )

        assert converter.feature_types is None
        assert converter.store_full_path is False
        assert converter.boto3_config is None

    def test_init_custom_params(self):
        converter = AmazonTextractConverter(
            aws_access_key_id=Secret.from_token("fake_id"),
            aws_secret_access_key=Secret.from_token("fake_secret"),
            aws_region_name=Secret.from_token("eu-west-1"),
            feature_types=["TABLES", "FORMS"],
            store_full_path=True,
            boto3_config={"connect_timeout": 10},
        )

        assert converter.feature_types == ["TABLES", "FORMS"]
        assert converter.store_full_path is True
        assert converter.boto3_config == {"connect_timeout": 10}

    def test_init_invalid_feature_types(self):
        with pytest.raises(ValueError, match="Invalid feature_types"):
            AmazonTextractConverter(
                aws_access_key_id=Secret.from_token("fake"),
                aws_secret_access_key=Secret.from_token("fake"),
                feature_types=["INVALID_TYPE"],
            )

    def test_init_all_valid_feature_types(self):
        converter = AmazonTextractConverter(
            aws_access_key_id=Secret.from_token("fake"),
            aws_secret_access_key=Secret.from_token("fake"),
            feature_types=["TABLES", "FORMS", "SIGNATURES", "LAYOUT"],
        )
        assert len(converter.feature_types) == 4


class TestAmazonTextractConverterSerialization:
    def test_to_dict(self):
        converter = AmazonTextractConverter(
            aws_access_key_id=Secret.from_env_var("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=Secret.from_env_var("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),
            aws_region_name=Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),
            aws_profile_name=Secret.from_env_var("AWS_PROFILE", strict=False),
            feature_types=["TABLES"],
            store_full_path=True,
            boto3_config={"connect_timeout": 5},
        )

        data = converter.to_dict()

        expected_type = "haystack_integrations.components.converters.amazon_textract.converter.AmazonTextractConverter"
        assert data["type"] == expected_type
        assert data["init_parameters"]["feature_types"] == ["TABLES"]
        assert data["init_parameters"]["store_full_path"] is True
        assert data["init_parameters"]["boto3_config"] == {"connect_timeout": 5}
        assert data["init_parameters"]["aws_access_key_id"] == {
            "type": "env_var",
            "env_vars": ["AWS_ACCESS_KEY_ID"],
            "strict": True,
        }

    def test_to_dict_default_params(self):
        converter = AmazonTextractConverter(
            aws_access_key_id=Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),
            aws_secret_access_key=Secret.from_env_var("AWS_SECRET_ACCESS_KEY", strict=False),
        )

        data = converter.to_dict()

        assert data["init_parameters"]["feature_types"] is None
        assert data["init_parameters"]["store_full_path"] is False
        assert data["init_parameters"]["boto3_config"] is None

    def test_from_dict(self):
        expected_type = "haystack_integrations.components.converters.amazon_textract.converter.AmazonTextractConverter"
        data = {
            "type": expected_type,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "feature_types": ["TABLES", "FORMS"],
                "store_full_path": False,
                "boto3_config": None,
            },
        }

        converter = AmazonTextractConverter.from_dict(data)

        assert converter.feature_types == ["TABLES", "FORMS"]
        assert converter.store_full_path is False
        assert converter.boto3_config is None

    def test_from_dict_roundtrip(self):
        converter = AmazonTextractConverter(
            aws_access_key_id=Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),
            aws_secret_access_key=Secret.from_env_var("AWS_SECRET_ACCESS_KEY", strict=False),
            feature_types=["FORMS"],
            store_full_path=True,
        )

        data = converter.to_dict()
        restored = AmazonTextractConverter.from_dict(data)

        assert restored.feature_types == converter.feature_types
        assert restored.store_full_path == converter.store_full_path
        assert restored.boto3_config == converter.boto3_config


class TestAmazonTextractConverterWarmUp:
    @patch("haystack_integrations.components.converters.amazon_textract.converter.boto3.Session")
    def test_warm_up_creates_client(self, mock_session_cls):
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_cls.return_value = mock_session

        converter = AmazonTextractConverter(
            aws_access_key_id=Secret.from_token("fake_id"),
            aws_secret_access_key=Secret.from_token("fake_secret"),
            aws_region_name=Secret.from_token("us-east-1"),
        )
        converter.warm_up()

        mock_session_cls.assert_called_once()
        mock_session.client.assert_called_once()
        call_args = mock_session.client.call_args
        assert call_args[0][0] == "textract"
        assert converter._client is mock_client

    @patch("haystack_integrations.components.converters.amazon_textract.converter.boto3.Session")
    def test_warm_up_idempotent(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.client.return_value = MagicMock()
        mock_session_cls.return_value = mock_session

        converter = AmazonTextractConverter(
            aws_access_key_id=Secret.from_token("fake"),
            aws_secret_access_key=Secret.from_token("fake"),
        )
        converter.warm_up()
        converter.warm_up()

        mock_session_cls.assert_called_once()

    @patch(
        "haystack_integrations.components.converters.amazon_textract.converter.boto3.Session",
        side_effect=BotoCoreError(),
    )
    def test_warm_up_configuration_error(self, _mock_session_cls):
        converter = AmazonTextractConverter(
            aws_access_key_id=Secret.from_token("fake"),
            aws_secret_access_key=Secret.from_token("fake"),
        )
        with pytest.raises(AmazonTextractConfigurationError, match="Could not connect to AWS Textract"):
            converter.warm_up()


class TestAmazonTextractConverterRun:
    def _make_converter_with_mock_client(self, feature_types=None, store_full_path=False):
        converter = AmazonTextractConverter(
            aws_access_key_id=Secret.from_token("fake"),
            aws_secret_access_key=Secret.from_token("fake"),
            aws_region_name=Secret.from_token("us-east-1"),
            feature_types=feature_types,
            store_full_path=store_full_path,
        )
        converter._client = MagicMock()
        return converter

    def test_run_detect_text(self, tmp_path):
        converter = self._make_converter_with_mock_client()
        response = _make_textract_response(lines=["Hello World", "Second line"])
        converter._client.detect_document_text.return_value = response

        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake image bytes")

        result = converter.run(sources=[test_file])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Hello World\nSecond line"
        assert result["documents"][0].meta["page_count"] == 1
        assert len(result["raw_textract_response"]) == 1
        converter._client.detect_document_text.assert_called_once()

    def test_run_analyze_document(self, tmp_path):
        converter = self._make_converter_with_mock_client(feature_types=["TABLES", "FORMS"])
        response = _make_textract_response(lines=["Name: John", "Age: 30"])
        converter._client.analyze_document.return_value = response

        test_file = tmp_path / "form.png"
        test_file.write_bytes(b"fake image bytes")

        result = converter.run(sources=[test_file])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Name: John\nAge: 30"
        converter._client.analyze_document.assert_called_once()
        call_kwargs = converter._client.analyze_document.call_args[1]
        assert call_kwargs["FeatureTypes"] == ["TABLES", "FORMS"]

    def test_run_with_metadata(self, tmp_path):
        converter = self._make_converter_with_mock_client()
        converter._client.detect_document_text.return_value = _make_textract_response(lines=["text"])

        test_file = tmp_path / "doc.png"
        test_file.write_bytes(b"bytes")

        result = converter.run(
            sources=[test_file],
            meta={"custom_key": "custom_value"},
        )

        doc = result["documents"][0]
        assert doc.meta["custom_key"] == "custom_value"
        assert doc.meta["page_count"] == 1

    def test_run_with_bytestream(self):
        converter = self._make_converter_with_mock_client()
        converter._client.detect_document_text.return_value = _make_textract_response(lines=["from bytes"])

        bs = ByteStream(data=b"fake image", meta={"file_path": "/some/path/image.png"})
        result = converter.run(sources=[bs])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "from bytes"
        assert result["documents"][0].meta["file_path"] == "image.png"

    def test_run_store_full_path(self, tmp_path):
        converter = self._make_converter_with_mock_client(store_full_path=True)
        converter._client.detect_document_text.return_value = _make_textract_response(lines=["text"])

        test_file = tmp_path / "doc.png"
        test_file.write_bytes(b"bytes")

        result = converter.run(sources=[test_file])

        doc = result["documents"][0]
        assert doc.meta["file_path"] == str(test_file)

    def test_run_store_basename_only(self, tmp_path):
        converter = self._make_converter_with_mock_client(store_full_path=False)
        converter._client.detect_document_text.return_value = _make_textract_response(lines=["text"])

        test_file = tmp_path / "doc.png"
        test_file.write_bytes(b"bytes")

        result = converter.run(sources=[test_file])

        doc = result["documents"][0]
        assert doc.meta["file_path"] == "doc.png"

    def test_run_multiple_sources(self, tmp_path):
        converter = self._make_converter_with_mock_client()
        converter._client.detect_document_text.side_effect = [
            _make_textract_response(lines=["First doc"]),
            _make_textract_response(lines=["Second doc"]),
        ]

        file1 = tmp_path / "a.png"
        file1.write_bytes(b"bytes1")
        file2 = tmp_path / "b.png"
        file2.write_bytes(b"bytes2")

        result = converter.run(sources=[file1, file2])

        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "First doc"
        assert result["documents"][1].content == "Second doc"
        assert len(result["raw_textract_response"]) == 2

    def test_run_multiple_sources_with_per_source_metadata(self, tmp_path):
        converter = self._make_converter_with_mock_client()
        converter._client.detect_document_text.side_effect = [
            _make_textract_response(lines=["A"]),
            _make_textract_response(lines=["B"]),
        ]

        file1 = tmp_path / "a.png"
        file1.write_bytes(b"bytes1")
        file2 = tmp_path / "b.png"
        file2.write_bytes(b"bytes2")

        result = converter.run(
            sources=[file1, file2],
            meta=[{"source": "first"}, {"source": "second"}],
        )

        assert result["documents"][0].meta["source"] == "first"
        assert result["documents"][1].meta["source"] == "second"

    def test_run_skips_failed_sources(self, tmp_path):
        converter = self._make_converter_with_mock_client()
        error_response = {"Error": {"Code": "InvalidParameterException", "Message": "bad image"}}
        converter._client.detect_document_text.side_effect = ClientError(error_response, "DetectDocumentText")

        test_file = tmp_path / "broken_image.png"
        test_file.write_bytes(b"bad bytes")

        result = converter.run(sources=[test_file])

        assert len(result["documents"]) == 0
        assert len(result["raw_textract_response"]) == 0

    def test_run_broken_image_mixed_with_valid(self, tmp_path):
        """A broken image among valid sources should be skipped while valid ones succeed."""
        converter = self._make_converter_with_mock_client()

        error_response = {"Error": {"Code": "UnsupportedDocumentException", "Message": "unsupported format"}}
        valid_response = _make_textract_response(lines=["Valid text"])

        converter._client.detect_document_text.side_effect = [
            ClientError(error_response, "DetectDocumentText"),
            valid_response,
        ]

        valid_file = tmp_path / "good.png"
        valid_file.write_bytes(b"fake valid image")
        broken_image = TEST_FILES_DIR / "broken_image.png"

        result = converter.run(sources=[broken_image, valid_file])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Valid text"
        assert len(result["raw_textract_response"]) == 1

    def test_run_skips_unreadable_source(self):
        converter = self._make_converter_with_mock_client()

        result = converter.run(sources=["/nonexistent/path/file.png"])

        assert len(result["documents"]) == 0
        assert len(result["raw_textract_response"]) == 0

    def test_run_empty_response(self, tmp_path):
        converter = self._make_converter_with_mock_client()
        converter._client.detect_document_text.return_value = {"Blocks": [], "ResponseMetadata": {}}

        test_file = tmp_path / "empty.png"
        test_file.write_bytes(b"bytes")

        result = converter.run(sources=[test_file])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == ""
        assert result["documents"][0].meta["page_count"] == 0

    def test_run_multi_page_response(self, tmp_path):
        converter = self._make_converter_with_mock_client()
        response = _make_textract_response(lines=["Page 1 text", "Page 2 text"], page_count=2)
        converter._client.detect_document_text.return_value = response

        test_file = tmp_path / "multipage.pdf"
        test_file.write_bytes(b"fake pdf bytes")

        result = converter.run(sources=[test_file])

        assert result["documents"][0].meta["page_count"] == 2
        assert "Page 1 text" in result["documents"][0].content
        assert "Page 2 text" in result["documents"][0].content

    def test_run_auto_warm_up(self, tmp_path):
        """Verify that run() calls warm_up() if client is not initialized."""
        converter = AmazonTextractConverter(
            aws_access_key_id=Secret.from_token("fake"),
            aws_secret_access_key=Secret.from_token("fake"),
            aws_region_name=Secret.from_token("us-east-1"),
        )

        mock_client = MagicMock()
        mock_client.detect_document_text.return_value = _make_textract_response(lines=["text"])

        with patch("haystack_integrations.components.converters.amazon_textract.converter.boto3.Session") as mock_sess:
            mock_sess.return_value.client.return_value = mock_client

            test_file = tmp_path / "test.png"
            test_file.write_bytes(b"bytes")

            result = converter.run(sources=[test_file])

            assert len(result["documents"]) == 1
            mock_sess.assert_called_once()

    def test_run_with_empty_sources(self):
        converter = self._make_converter_with_mock_client()
        result = converter.run(sources=[])

        assert result["documents"] == []
        assert result["raw_textract_response"] == []

    def test_run_with_queries_only(self, tmp_path):
        """Queries alone should trigger AnalyzeDocument with QUERIES feature type."""
        converter = self._make_converter_with_mock_client()
        response = _make_textract_response(lines=["John Doe"])
        converter._client.analyze_document.return_value = response

        test_file = tmp_path / "form.png"
        test_file.write_bytes(b"fake image bytes")

        result = converter.run(
            sources=[test_file],
            queries=["What is the patient name?"],
        )

        assert len(result["documents"]) == 1
        converter._client.analyze_document.assert_called_once()
        call_kwargs = converter._client.analyze_document.call_args[1]
        assert "QUERIES" in call_kwargs["FeatureTypes"]
        assert call_kwargs["QueriesConfig"] == {
            "Queries": [{"Text": "What is the patient name?"}],
        }
        converter._client.detect_document_text.assert_not_called()

    def test_run_with_queries_and_feature_types(self, tmp_path):
        """Queries combined with existing feature_types should merge correctly."""
        converter = self._make_converter_with_mock_client(feature_types=["TABLES"])
        response = _make_textract_response(lines=["Total: $100"])
        converter._client.analyze_document.return_value = response

        test_file = tmp_path / "invoice.png"
        test_file.write_bytes(b"fake image bytes")

        result = converter.run(
            sources=[test_file],
            queries=["What is the total?", "What is the due date?"],
        )

        assert len(result["documents"]) == 1
        call_kwargs = converter._client.analyze_document.call_args[1]
        assert "TABLES" in call_kwargs["FeatureTypes"]
        assert "QUERIES" in call_kwargs["FeatureTypes"]
        assert call_kwargs["QueriesConfig"] == {
            "Queries": [
                {"Text": "What is the total?"},
                {"Text": "What is the due date?"},
            ],
        }

    @pytest.mark.parametrize("queries", [None, []])
    def test_run_no_queries_uses_detect(self, tmp_path, queries):
        """None and empty list for queries should both use detect_document_text."""
        converter = self._make_converter_with_mock_client()
        converter._client.detect_document_text.return_value = _make_textract_response(lines=["text"])

        test_file = tmp_path / "doc.png"
        test_file.write_bytes(b"bytes")

        converter.run(sources=[test_file], queries=queries)

        converter._client.detect_document_text.assert_called_once()
        converter._client.analyze_document.assert_not_called()

    def test_run_queries_does_not_mutate_feature_types(self, tmp_path):
        """Passing queries at runtime should not modify the init feature_types."""
        converter = self._make_converter_with_mock_client(feature_types=["TABLES"])
        converter._client.analyze_document.return_value = _make_textract_response(lines=["text"])

        test_file = tmp_path / "doc.png"
        test_file.write_bytes(b"bytes")

        converter.run(sources=[test_file], queries=["What?"])

        assert converter.feature_types == ["TABLES"]
        assert "QUERIES" not in converter.feature_types
