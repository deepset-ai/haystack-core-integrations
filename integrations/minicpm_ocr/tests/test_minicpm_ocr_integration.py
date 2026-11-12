# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from haystack_integrations.components.converters.minicpm_ocr import MiniCPMOCRDocumentConverter

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("MINICPM_OCR_SERVER_URL"),
        reason="MINICPM_OCR_SERVER_URL is not set; skipping the live MiniCPM-OCR test",
    ),
]


def _page(title: str, body: str) -> Image.Image:
    # Large fonts on an A4-sized page: the layout model ignores very small text, which leaves nothing to recognize
    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    draw.text((120, 150), title, fill="black", font=ImageFont.load_default(size=64))
    draw.text((120, 320), body, fill="black", font=ImageFont.load_default(size=32))
    return image


@pytest.fixture
def converter() -> MiniCPMOCRDocumentConverter:
    converter = MiniCPMOCRDocumentConverter(
        server_url=os.environ["MINICPM_OCR_SERVER_URL"],
        model=os.getenv("MINICPM_OCR_MODEL", "minicpm-ocr"),
        backend=os.getenv("MINICPM_OCR_BACKEND", "vllm"),
        layout_model_dir=os.getenv("MINICPM_OCR_LAYOUT_MODEL_DIR"),
        layout_device=os.getenv("MINICPM_OCR_LAYOUT_DEVICE", "cpu"),
    )
    converter.warm_up()
    return converter


def test_converts_image(converter: MiniCPMOCRDocumentConverter, tmp_path: Path) -> None:
    source = tmp_path / "sample.png"
    _page("Haystack Report", "Invoice number 12345").save(source)

    documents = converter.run(sources=[source])["documents"]

    assert len(documents) == 1
    assert documents[0].meta == {"file_path": "sample.png", "page_count": 1, "ocr_engine": "minicpm-ocr"}
    assert "Haystack" in documents[0].content
    assert "12345" in documents[0].content


def test_converts_multipage_pdf(converter: MiniCPMOCRDocumentConverter, tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    first, second = _page("First Page", "Invoice number 12345"), _page("Second Page", "Total amount 42 EUR")
    first.save(source, save_all=True, append_images=[second], resolution=150)

    documents = converter.run(sources=[source])["documents"]

    assert len(documents) == 1
    assert documents[0].meta["page_count"] == 2
    pages = documents[0].content.split("\f")
    assert len(pages) == 2
    assert "12345" in pages[0]
    assert "42" in pages[1]
