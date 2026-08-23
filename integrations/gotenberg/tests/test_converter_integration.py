# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.gotenberg import GotenbergFileConverter


def assert_pdf(result: dict) -> None:
    assert len(result["output"]) == 1
    output = result["output"][0]
    assert isinstance(output, ByteStream)
    assert output.mime_type == "application/pdf"
    assert output.data.startswith(b"%PDF-")


@pytest.mark.integration
def test_libreoffice_converts_mime_typed_bytestream_without_file_path() -> None:
    source = ByteStream(
        data=Path("tests/test_files/docx/sample_docx.docx").read_bytes(),
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    result = GotenbergFileConverter().run([source], conversion_type="libreoffice")

    assert_pdf(result)


@pytest.mark.integration
def test_html_converts_document() -> None:
    result = GotenbergFileConverter().run(
        ["<!doctype html><html><body><h1>Haystack HTML integration test</h1></body></html>"],
        conversion_type="html",
    )

    assert_pdf(result)


@pytest.mark.integration
def test_markdown_converts_document() -> None:
    result = GotenbergFileConverter().run(
        ["# Haystack Markdown integration test\n\nConverted by Gotenberg."], conversion_type="markdown"
    )

    assert_pdf(result)


@pytest.mark.integration
def test_url_converts_page_without_external_network() -> None:
    # The Gotenberg container's own health endpoint makes this deterministic and external-network independent.
    result = GotenbergFileConverter().run(["http://localhost:3000/health"], conversion_type="url")

    assert_pdf(result)
