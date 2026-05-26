# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.amazon_textract import AmazonTextractConverter

SKIP_REASON_NO_CREDENTIALS = "AWS credentials not available"
SKIP_REASON_NO_REGION = "AWS region not configured"


@pytest.mark.integration
class TestAmazonTextractConverterIntegration:
    @pytest.fixture
    def test_files_path(self):
        return Path(__file__).parent / "test_files"

    @pytest.fixture
    def converter(self):
        return AmazonTextractConverter()

    @pytest.mark.skipif(not os.environ.get("AWS_ACCESS_KEY_ID"), reason=SKIP_REASON_NO_CREDENTIALS)
    @pytest.mark.skipif(not os.environ.get("AWS_DEFAULT_REGION"), reason=SKIP_REASON_NO_REGION)
    def test_run_detect_text_from_image(self, converter, test_files_path):
        """Integration test: detect text from an image file."""
        image_path = test_files_path / "sample_text.png"
        if not image_path.exists():
            pytest.skip("Test image file not available")

        results = converter.run(sources=[image_path])

        assert "documents" in results
        assert len(results["documents"]) == 1
        assert len(results["documents"][0].content) > 0
        assert results["documents"][0].meta["page_count"] >= 1
        assert "raw_textract_response" in results
        assert len(results["raw_textract_response"]) == 1

    @pytest.mark.skipif(not os.environ.get("AWS_ACCESS_KEY_ID"), reason=SKIP_REASON_NO_CREDENTIALS)
    @pytest.mark.skipif(not os.environ.get("AWS_DEFAULT_REGION"), reason=SKIP_REASON_NO_REGION)
    def test_run_analyze_document_with_tables(self, test_files_path):
        """Integration test: analyze document with table detection."""
        image_path = test_files_path / "sample_text.png"
        if not image_path.exists():
            pytest.skip("Test image file not available")

        converter = AmazonTextractConverter(feature_types=["TABLES"])
        results = converter.run(sources=[image_path])

        assert "documents" in results
        assert len(results["documents"]) == 1
        assert len(results["documents"][0].content) > 0

    @pytest.mark.skipif(not os.environ.get("AWS_ACCESS_KEY_ID"), reason=SKIP_REASON_NO_CREDENTIALS)
    @pytest.mark.skipif(not os.environ.get("AWS_DEFAULT_REGION"), reason=SKIP_REASON_NO_REGION)
    def test_run_with_metadata(self, converter, test_files_path):
        """Integration test: verify metadata handling."""
        image_path = test_files_path / "sample_text.png"
        if not image_path.exists():
            pytest.skip("Test image file not available")

        results = converter.run(
            sources=[image_path],
            meta={"custom_key": "custom_value"},
        )

        doc = results["documents"][0]
        assert doc.meta["custom_key"] == "custom_value"
        assert doc.meta["file_path"] == "sample_text.png"

    @pytest.mark.skipif(not os.environ.get("AWS_ACCESS_KEY_ID"), reason=SKIP_REASON_NO_CREDENTIALS)
    @pytest.mark.skipif(not os.environ.get("AWS_DEFAULT_REGION"), reason=SKIP_REASON_NO_REGION)
    def test_run_with_bytestream(self, converter, test_files_path):
        """Integration test: convert from ByteStream."""
        image_path = test_files_path / "sample_text.png"
        if not image_path.exists():
            pytest.skip("Test image file not available")

        data = image_path.read_bytes()
        bs = ByteStream(data=data, meta={"file_path": str(image_path)})
        results = converter.run(sources=[bs])

        assert len(results["documents"]) == 1
        assert len(results["documents"][0].content) > 0
