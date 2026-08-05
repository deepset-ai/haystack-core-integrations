from pathlib import Path

import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.opendataloader_pdf import (
    OpenDataLoaderConverter,
)


@pytest.fixture
def _mock_opendataloader(monkeypatch):
    monkeypatch.setattr(
        "shutil.which",
        lambda _command: "java",
    )

    def fake_convert(
        input_path,
        output_dir,
        **kwargs,
    ):
        output_dir = Path(output_dir)
        extension = {
            "markdown": "md",
            "text": "txt",
            "html": "html",
            "json": "json",
        }[kwargs["format"]]

        for pdf in input_path:
            pdf_path = Path(pdf)

            output_file = output_dir / f"{pdf_path.stem}.{extension}"

            output_file.write_text(
                "This is extracted PDF content",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        "opendataloader_pdf.convert",
        fake_convert,
    )


def test_converter_with_pdf_path(
    tmp_path,
    _mock_opendataloader,
):

    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF fake pdf")

    converter = OpenDataLoaderConverter()
    result = converter.run(sources=[pdf_file])

    assert len(result["documents"]) == 1
    document = result["documents"][0]
    assert document.content == "This is extracted PDF content"
    assert document.meta["output_format"] == "markdown"


def test_converter_with_bytestream(
    _mock_opendataloader,
):
    stream = ByteStream(
        data=b"%PDF fake pdf",
        mime_type="application/pdf",
    )

    converter = OpenDataLoaderConverter()
    result = converter.run(sources=[stream])
    assert len(result["documents"]) == 1
    document = result["documents"][0]
    assert document.content == "This is extracted PDF content"


def test_converter_raises_when_java_is_unavailable(
    tmp_path,
    monkeypatch,
):
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF fake pdf")
    monkeypatch.setattr(
        "shutil.which",
        lambda _command: None,
    )
    converter = OpenDataLoaderConverter()
    with pytest.raises(
        RuntimeError,
        match=r"Java.*required",
    ):
        converter.run(sources=[pdf_file])


def test_converter_passes_options(
    tmp_path,
    _mock_opendataloader,
    monkeypatch,
):

    calls = {}

    def fake_convert(
        output_dir,
        **kwargs,
    ):
        calls.update(kwargs)
        output_file = Path(output_dir) / "document.md"
        output_file.write_text(
            "content",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "opendataloader_pdf.convert",
        fake_convert,
    )

    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF fake pdf")
    converter = OpenDataLoaderConverter(
        convert_kwargs={
            "hybrid": True,
            "table_method": "cluster",
        }
    )
    converter.run(sources=[pdf_file])
    assert calls["hybrid"] is True
    assert calls["table_method"] == "cluster"


def test_converter_rejects_non_pdf(
    tmp_path,
):
    text_file = tmp_path / "document.txt"
    text_file.write_text("hello")
    converter = OpenDataLoaderConverter()

    with pytest.raises(
        ValueError,
        match="only supports PDFs",
    ):
        converter.run(sources=[text_file])


def test_converter_serialization():
    converter = OpenDataLoaderConverter(
        output_format="text",
        convert_kwargs={"hybrid": True},
    )

    data = converter.to_dict()
    restored = OpenDataLoaderConverter.from_dict(data)
    assert restored.output_format == "text"
    assert restored.convert_kwargs["hybrid"] is True


@pytest.mark.integration
def test_real_pdf_conversion():
    pdf_file = Path(__file__).parent / "test_files" / "hello_world.pdf"
    converter = OpenDataLoaderConverter(output_format="markdown")
    result = converter.run(sources=[pdf_file])
    assert len(result["documents"]) == 1

    document = result["documents"][0]
    assert document.content
    assert "output_format" in document.meta
    assert document.meta["output_format"] == "markdown"
