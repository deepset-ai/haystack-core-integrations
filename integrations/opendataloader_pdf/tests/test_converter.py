import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

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

            if kwargs["image_output"] == "external":
                image_dir = Path(kwargs["image_dir"])
                image_dir.mkdir(parents=True, exist_ok=True)
                (image_dir / f"{pdf_path.stem}_image_1.png").write_bytes(b"fake image")

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
    assert result["image_documents"] == []


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
    assert "image_dir" not in call


def test_converter_extracts_images_to_persistent_directory(tmp_path, _mock_opendataloader):
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF fake pdf")
    image_output_dir = tmp_path / "nested" / "images"
    converter = OpenDataLoaderConverter(extract_images=True, image_output_dir=image_output_dir)

    result = converter.run(sources=[pdf_file])

    call = _mock_opendataloader[0]
    staged_pdf_stem = Path(call["input_path"][0]).stem
    call_image_dir = Path(call["image_dir"])
    expected_image_path = call_image_dir / f"{staged_pdf_stem}_image_1.png"
    assert call["image_output"] == "external"
    assert call_image_dir.parent == image_output_dir
    assert expected_image_path.read_bytes() == b"fake image"
    assert len(result["documents"]) == 1
    assert len(result["image_documents"]) == 1
    image_document = result["image_documents"][0]
    assert image_document.content is None
    assert image_document.meta == {"file_path": str(expected_image_path)}


def test_converter_uses_unique_image_paths_across_runs(tmp_path, _mock_opendataloader):
    first_pdf = tmp_path / "report_a.pdf"
    second_pdf = tmp_path / "report_b.pdf"
    first_pdf.write_bytes(b"%PDF first")
    second_pdf.write_bytes(b"%PDF second")
    image_output_dir = tmp_path / "images"
    converter = OpenDataLoaderConverter(extract_images=True, image_output_dir=image_output_dir)

    first_result = converter.run(sources=[first_pdf])
    first_image_path = Path(first_result["image_documents"][0].meta["file_path"])
    second_result = converter.run(sources=[second_pdf])
    second_image_path = Path(second_result["image_documents"][0].meta["file_path"])

    assert first_image_path != second_image_path
    assert first_image_path.parent != second_image_path.parent
    assert first_image_path.exists()
    assert second_image_path.exists()
    assert set(image_output_dir.rglob("*.png")) == {first_image_path, second_image_path}


def test_converter_isolates_images_between_concurrent_runs(tmp_path, _mock_opendataloader, monkeypatch):
    first_pdf = tmp_path / "report_a.pdf"
    second_pdf = tmp_path / "report_b.pdf"
    first_pdf.write_bytes(b"%PDF first")
    second_pdf.write_bytes(b"%PDF second")
    image_output_dir = tmp_path / "images"
    converter = OpenDataLoaderConverter(extract_images=True, image_output_dir=image_output_dir)
    convert = converter_module.opendataloader_pdf.convert
    conversions_finished = Barrier(2)

    def synchronized_convert(*args, **kwargs):
        convert(*args, **kwargs)
        conversions_finished.wait(timeout=5)

    monkeypatch.setattr(converter_module.opendataloader_pdf, "convert", synchronized_convert)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(converter.run, sources=[first_pdf])
        second_future = executor.submit(converter.run, sources=[second_pdf])
        first_result = first_future.result(timeout=5)
        second_result = second_future.result(timeout=5)

    first_images = first_result["image_documents"]
    second_images = second_result["image_documents"]
    assert len(first_images) == 1
    assert len(second_images) == 1
    first_image_path = Path(first_images[0].meta["file_path"])
    second_image_path = Path(second_images[0].meta["file_path"])
    assert first_image_path.parent != second_image_path.parent
    assert first_image_path.parent.parent == image_output_dir
    assert second_image_path.parent.parent == image_output_dir


def test_converter_returns_empty_image_output_when_pdf_has_no_images(tmp_path, _mock_opendataloader, monkeypatch):
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF fake pdf")
    image_output_dir = tmp_path / "images"

    def fake_convert_without_images(input_path, output_dir, **_kwargs):
        for pdf in input_path:
            output_file = Path(output_dir) / f"{Path(pdf).stem}.md"
            output_file.write_text("This is extracted PDF content", encoding="utf-8")

    monkeypatch.setattr(converter_module.opendataloader_pdf, "convert", fake_convert_without_images)
    converter = OpenDataLoaderConverter(extract_images=True, image_output_dir=image_output_dir)

    result = converter.run(sources=[pdf_file])

    assert len(result["documents"]) == 1
    assert result["image_documents"] == []
    assert image_output_dir.is_dir()


def test_converter_only_returns_images_created_by_current_run(tmp_path, _mock_opendataloader):
    pdf_file = tmp_path / "document.pdf"
    pdf_file.write_bytes(b"%PDF fake pdf")
    image_output_dir = tmp_path / "images"
    image_output_dir.mkdir()
    existing_image = image_output_dir / "existing.png"
    existing_image.write_bytes(b"existing image")
    converter = OpenDataLoaderConverter(extract_images=True, image_output_dir=str(image_output_dir))

    result = converter.run(sources=[pdf_file])

    call = _mock_opendataloader[0]
    expected_image_path = Path(call["image_dir"]) / f"{Path(call['input_path'][0]).stem}_image_1.png"
    assert existing_image.exists()
    assert [document.meta["file_path"] for document in result["image_documents"]] == [str(expected_image_path)]


def test_converter_requires_output_directory_when_extracting_images():
    with pytest.raises(ValueError, match="image_output_dir is required"):
        OpenDataLoaderConverter(extract_images=True)


def test_converter_ignores_managed_image_convert_kwargs(caplog):
    convert_kwargs = {"image_output": "external", "image_dir": "images", "language": "en"}

    converter = OpenDataLoaderConverter(convert_kwargs=convert_kwargs)

    assert "Ignoring component-managed image options" in caplog.text
    assert converter.convert_kwargs == {"language": "en"}
    assert convert_kwargs == {"image_output": "external", "image_dir": "images", "language": "en"}


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


def test_converter_empty_sources_returns_both_outputs_without_conversion(tmp_path, monkeypatch):
    converter = OpenDataLoaderConverter(extract_images=True, image_output_dir=tmp_path / "images")

    def fail_java_check():
        pytest.fail("Java should not be checked for empty sources")

    monkeypatch.setattr(converter, "_check_java_available", fail_java_check)

    assert converter.run(sources=[]) == {"documents": [], "image_documents": []}
    assert not (tmp_path / "images").exists()


def test_converter_serialization(tmp_path):
    image_output_dir = tmp_path / "images"
    converter = OpenDataLoaderConverter(
        output_format="text",
        convert_kwargs={"hybrid": "docling-fast"},
        extract_images=True,
        image_output_dir=image_output_dir,
    )
    data = converter.to_dict()
    restored = OpenDataLoaderConverter.from_dict(data)

    assert data["init_parameters"]["image_output_dir"] == str(image_output_dir)
    assert restored.output_format == "text"
    assert restored.convert_kwargs["hybrid"] == "docling-fast"
    assert restored.extract_images is True
    assert restored.image_output_dir == image_output_dir


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


@pytest.mark.integration
def test_real_pdf_conversion_extracts_images(tmp_path):
    pdf_file = Path(__file__).parent / "test_files" / "pdf_with_image.pdf"
    image_output_dir = tmp_path / "images"
    converter = OpenDataLoaderConverter(extract_images=True, image_output_dir=image_output_dir)

    result = converter.run(sources=[pdf_file])

    assert result["image_documents"]
    assert Path(result["image_documents"][0].meta["file_path"]).exists()
