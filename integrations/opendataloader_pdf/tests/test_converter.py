import subprocess
from pathlib import Path

import pytest
from haystack.dataclasses import ByteStream

import haystack_integrations.components.converters.opendataloader_pdf.converter as converter_module
from haystack_integrations.components.converters.opendataloader_pdf import (
    OpenDataLoaderConverter,
)


@pytest.fixture
def _mock_opendataloader(monkeypatch):
    calls = []
    monkeypatch.setattr(
        converter_module.shutil,
        "which",
        lambda _command: "/usr/bin/java",
    )
    monkeypatch.setattr(
        converter_module.subprocess,
        "run",
        lambda *_args, **_kwargs: None,
    )

    def fake_convert(
        input_path,
        output_dir,
        **kwargs,
    ):
        calls.append(
            {
                "input_path": input_path,
                "output_dir": output_dir,
                **kwargs,
            }
        )
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
        converter_module.opendataloader_pdf,
        "convert",
        fake_convert,
    )
    return calls


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
    assert document.meta["file_path"] == "document.pdf"
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
    assert document.meta["file_path"] == "document_0.pdf"


def test_converter_raises_when_java_is_unavailable(
    tmp_path,
    monkeypatch,
):
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF fake pdf")

    monkeypatch.setattr(
        converter_module.shutil,
        "which",
        lambda _command: None,
    )
    converter = OpenDataLoaderConverter()
    with pytest.raises(
        RuntimeError,
        match="Java 11 or newer is required",
    ):
        converter.run(sources=[pdf_file])


def test_converter_raises_when_java_cannot_execute(
    tmp_path,
    monkeypatch,
):
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF fake pdf")

    monkeypatch.setattr(
        converter_module.shutil,
        "which",
        lambda _command: "/usr/bin/java",
    )

    def fail_java(
        *_args,
        **_kwargs,
    ):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["java", "-version"],
        )

    monkeypatch.setattr(
        converter_module.subprocess,
        "run",
        fail_java,
    )
    converter = OpenDataLoaderConverter()
    with pytest.raises(
        RuntimeError,
        match="Java 11 or newer is required",
    ):
        converter.run(sources=[pdf_file])


def test_converter_passes_options(
    tmp_path,
    _mock_opendataloader,
):
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF fake pdf")
    converter = OpenDataLoaderConverter(
        convert_kwargs={
            "hybrid": "docling-fast",
            "table_method": "cluster",
        }
    )

    converter.run(sources=[pdf_file])
    call = _mock_opendataloader[0]

    assert call["hybrid"] == "docling-fast"
    assert call["table_method"] == "cluster"
    assert call["image_output"] == "off"


def test_converter_rejects_non_pdf(
    tmp_path,
    _mock_opendataloader,
):
    text_file = tmp_path / "document.txt"
    text_file.write_text(
        "hello",
        encoding="utf-8",
    )
    converter = OpenDataLoaderConverter()
    with pytest.raises(
        ValueError,
        match="only supports PDFs",
    ):
        converter.run(sources=[text_file])


def test_converter_handles_duplicate_file_names(
    tmp_path,
    _mock_opendataloader,
    monkeypatch,
):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_pdf = first_dir / "report.pdf"
    second_pdf = second_dir / "report.pdf"
    first_pdf.write_bytes(b"%PDF first")
    second_pdf.write_bytes(b"%PDF second")

    def fake_convert(
        input_path,
        output_dir,
        **kwargs,
    ):
        extension = {
            "markdown": "md",
            "text": "txt",
            "html": "html",
            "json": "json",
        }[kwargs["format"]]
        for pdf in input_path:
            pdf_path = Path(pdf)
            output_file = Path(output_dir) / f"{pdf_path.stem}.{extension}"

            output_file.write_text(
                pdf_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    monkeypatch.setattr(
        converter_module.opendataloader_pdf,
        "convert",
        fake_convert,
    )
    converter = OpenDataLoaderConverter()
    result = converter.run(
        sources=[
            first_pdf,
            second_pdf,
        ]
    )

    assert [document.content for document in result["documents"]] == [
        "%PDF first",
        "%PDF second",
    ]


def test_converter_preserves_bytestream_and_per_source_metadata(
    _mock_opendataloader,
):
    first_stream = ByteStream(
        data=b"%PDF first",
        mime_type="application/pdf",
        meta={
            "file_path": "uploads/first.pdf",
            "source_id": "first-id",
        },
    )
    second_stream = ByteStream(
        data=b"%PDF second",
        mime_type="application/pdf",
        meta={
            "file_path": "uploads/second.pdf",
            "source_id": "second-id",
        },
    )

    converter = OpenDataLoaderConverter()
    result = converter.run(
        sources=[
            first_stream,
            second_stream,
        ],
        meta=[
            {"category": "first"},
            {"category": "second"},
        ],
    )
    first_document, second_document = result["documents"]

    assert first_document.meta["file_path"] == "uploads/first.pdf"
    assert first_document.meta["source_id"] == "first-id"
    assert first_document.meta["category"] == "first"
    assert second_document.meta["file_path"] == "uploads/second.pdf"
    assert second_document.meta["source_id"] == "second-id"
    assert second_document.meta["category"] == "second"


def test_converter_serialization():
    converter = OpenDataLoaderConverter(
        output_format="text",
        convert_kwargs={"hybrid": "docling-fast"},
    )
    data = converter.to_dict()
    restored = OpenDataLoaderConverter.from_dict(data)

    assert restored.output_format == "text"
    assert restored.convert_kwargs["hybrid"] == "docling-fast"


@pytest.mark.integration
def test_real_pdf_conversion():
    pdf_file = Path(__file__).parent / "test_files" / "hello_world.pdf"
    converter = OpenDataLoaderConverter(
        output_format="markdown",
    )
    result = converter.run(
        sources=[pdf_file],
    )
    assert len(result["documents"]) == 1
    document = result["documents"][0]
    assert document.content
    assert document.meta["file_path"] == "hello_world.pdf"
    assert document.meta["output_format"] == "markdown"
