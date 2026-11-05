# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from haystack_integrations.components.converters.minicpm_ocr import MiniCPMOCRDocumentConverter

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.getenv("MINICPM_OCR_SERVER_URL"),
    reason="MINICPM_OCR_SERVER_URL is not set; skipping the live MiniCPM-OCR test",
)
def test_converts_image_with_live_server(tmp_path: Path) -> None:
    """Run the converter against a real MiniCPM-OCR server and check the output shape."""
    image = Image.new("RGB", (800, 300), "white")
    ImageDraw.Draw(image).text((40, 120), "Hello Haystack 12345", fill="black")
    source = tmp_path / "sample.png"
    image.save(source)

    converter = MiniCPMOCRDocumentConverter(
        server_url=os.environ["MINICPM_OCR_SERVER_URL"],
        model=os.getenv("MINICPM_OCR_MODEL", "minicpm-ocr"),
        layout_device=os.getenv("MINICPM_OCR_LAYOUT_DEVICE", "cpu"),
    )

    result = converter.run(sources=[source])

    assert len(result["documents"]) == 1
    document = result["documents"][0]
    assert document.meta["page_count"] == 1
    assert document.meta["ocr_engine"] == "minicpm-ocr"
    assert len(document.content.strip()) > 0
