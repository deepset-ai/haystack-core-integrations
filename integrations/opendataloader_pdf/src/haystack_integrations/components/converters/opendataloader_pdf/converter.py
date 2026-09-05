# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import normalize_metadata
from haystack.dataclasses import ByteStream

import opendataloader_pdf  # type: ignore[import-untyped]

OutputFormat = Literal["markdown", "text", "json", "html"]

logger = logging.getLogger(__name__)


@component
class OpenDataLoaderConverter:
    """
    OpenDataLoader PDF converter component.

    The component accepts PDF file paths and Haystack ByteStream objects, runs OpenDataLoader PDF extraction, and
    returns Haystack Document objects. It can also extract images to a persistent directory and return one image
    Document per extracted file.

    Java 11 or newer must be installed and available on PATH.

    ### Usage example
    ```python
    from haystack_integrations.components.converters.opendataloader_pdf import OpenDataLoaderConverter

    converter = OpenDataLoaderConverter(
        output_format="markdown", extract_images=True, image_output_dir="extracted_images"
    )
    result = converter.run(sources=["report.pdf"], meta={"source": "annual-report"})

    documents = result["documents"]
    image_documents = result["image_documents"]
    print(documents[0].content)
    print(documents[0].meta["file_path"])
    ```
    """

    def __init__(
        self,
        *,
        output_format: OutputFormat = "markdown",
        convert_kwargs: dict[str, Any] | None = None,
        extract_images: bool = False,
        image_output_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the OpenDataLoader converter.

        :param output_format: Format OpenDataLoader should produce.
        :param convert_kwargs: Additional arguments passed to `opendataloader_pdf.convert`. See the
            [OpenDataLoader PDF Python options](https://opendataloader.org/docs/quick-start-python#convert-options).
            The `image_output` and `image_dir` arguments are managed by this component; supplied values are ignored.
        :param extract_images: Whether to extract images and return them through the `image_documents` output.
        :param image_output_dir: Persistent directory for extracted image files. Required when `extract_images` is
            `True`.
        :raises ValueError: If image extraction is enabled without an output directory.
        """
        conversion_options = convert_kwargs.copy() if convert_kwargs else {}
        if managed_options := {"image_dir", "image_output"}.intersection(conversion_options):
            message = (
                f"Ignoring component-managed image options in convert_kwargs: {', '.join(sorted(managed_options))}"
            )
            logger.warning(message)
            for option in managed_options:
                conversion_options.pop(option)
        if extract_images and image_output_dir is None:
            message = "image_output_dir is required when extract_images is enabled"
            raise ValueError(message)

        self.output_format = output_format
        self.convert_kwargs = conversion_options
        self.extract_images = extract_images
        self.image_output_dir = Path(image_output_dir) if image_output_dir is not None else None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component.

        :returns:
            Dictionary representation of the converter.
        """
        return default_to_dict(
            self,
            output_format=self.output_format,
            convert_kwargs=self.convert_kwargs,
            extract_images=self.extract_images,
            image_output_dir=str(self.image_output_dir) if self.image_output_dir is not None else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenDataLoaderConverter":
        """
        Deserialize the component.

        :param data: Serialized component dictionary.
        :returns:
            Reconstructed OpenDataLoaderConverter.
        """
        return default_from_dict(cls=cls, data=data)

    def _prepare_sources(
        self, sources: list[str | Path | ByteStream], metadata: list[dict[str, Any]], input_dir: Path
    ) -> list[tuple[Path, dict[str, Any]]]:
        """
        Stage PDF sources under unique names and preserve their metadata.

        :param sources: PDF file paths or Haystack ByteStream objects.
        :param metadata: User-provided metadata aligned with `sources`.
        :param input_dir: Temporary directory where sources are staged.
        :raises ValueError: If a ByteStream has a non-PDF MIME type, or a file path does not have a `.pdf` extension.
        :returns:
            Staged PDF paths paired with their merged metadata.
        """
        prepared_sources: list[tuple[Path, dict[str, Any]]] = []
        for index, (source, user_meta) in enumerate(zip(sources, metadata, strict=True)):
            staged_path = input_dir / f"document_{index}.pdf"
            if isinstance(source, ByteStream):
                if source.mime_type:
                    mime_type = source.mime_type.split(sep=";", maxsplit=1)[0].strip().lower()
                    if mime_type != "application/pdf":
                        message = (
                            "OpenDataLoaderConverter only supports PDF ByteStreams. "
                            f"Received MIME type: {source.mime_type}"
                        )
                        raise ValueError(message)
                staged_path.write_bytes(source.data)
                source_meta = dict(source.meta)
                if not source_meta.get("file_path"):
                    source_meta["file_path"] = f"document_{index}.pdf"
            else:
                source_path = Path(source)
                if source_path.suffix.lower() != ".pdf":
                    message = f"OpenDataLoaderConverter only supports PDFs: {source_path}"
                    raise ValueError(message)
                shutil.copyfile(src=source_path, dst=staged_path)
                source_meta = {"file_path": source_path.name}
            prepared_sources.append((staged_path, {**source_meta, **user_meta}))
        return prepared_sources

    @staticmethod
    def _check_java_available() -> None:
        """
        Check whether a usable Java runtime is available.

        :raises RuntimeError: If `java` is not available on PATH or cannot be executed.
        """
        message = (
            "Java 11 or newer is required to use OpenDataLoaderConverter. "
            "Install Java and ensure `java` is available on PATH."
        )
        java = shutil.which("java")
        if java is None:
            raise RuntimeError(message)

        try:
            subprocess.run(  # noqa: S603
                args=[java, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(message) from exc

    def _read_output(self, output_dir: Path, pdf_path: Path) -> str:
        """
        Read the OpenDataLoader output file.

        :param output_dir: Directory containing converted files.
        :param pdf_path: Staged PDF path used to determine the output filename.
        :raises RuntimeError: If OpenDataLoader did not create the expected output file.
        :returns:
            Extracted document content.
        """
        extension_map = {"markdown": "md", "text": "txt", "html": "html", "json": "json"}
        output_file = output_dir / f"{pdf_path.stem}.{extension_map[self.output_format]}"
        if not output_file.exists():
            message = f"OpenDataLoader did not create expected file: {output_file}"
            raise RuntimeError(message)
        return output_file.read_text(encoding="utf-8")

    @staticmethod
    def _file_state(directory: Path) -> dict[Path, tuple[int, int, int]]:
        """
        Capture the state of files in an image output directory.

        The state is used to distinguish files extracted by the current conversion from unrelated files already in
        the user-provided directory.

        :param directory: Directory whose files should be captured recursively.
        :returns:
            Mapping of file paths to their size, modification time, and change time.
        """
        state: dict[Path, tuple[int, int, int]] = {}
        for path in directory.rglob("*"):
            if path.is_file():
                stat = path.stat()
                state[path] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        return state

    @component.output_types(documents=list[Document], image_documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Convert PDF sources into Haystack Documents.

        :param sources: PDF file paths or Haystack ByteStream objects.
        :param meta: Optional metadata attached to the generated Documents. A single dictionary is applied to every
            source. A list must contain one dictionary per source. ByteStream metadata is also preserved.
        :returns:
            Dictionary containing the converted text Documents and image Documents. Each image Document has the
            persistent extracted image path in its `file_path` metadata field.
        """
        if not sources:
            return {"documents": [], "image_documents": []}

        self._check_java_available()
        metadata = normalize_metadata(meta=meta, sources_count=len(sources))

        documents: list[Document] = []
        image_documents: list[Document] = []
        if self.image_output_dir is not None:
            self.image_output_dir.mkdir(parents=True, exist_ok=True)
            image_files_before = self._file_state(self.image_output_dir)

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            prepared_sources = self._prepare_sources(sources=sources, metadata=metadata, input_dir=input_dir)

            conversion_kwargs = {"image_output": "off", **self.convert_kwargs}
            if self.image_output_dir is not None:
                conversion_kwargs.update(image_output="external", image_dir=str(self.image_output_dir))

            opendataloader_pdf.convert(
                input_path=[str(pdf_path) for pdf_path, _ in prepared_sources],
                output_dir=str(output_dir),
                format=self.output_format,
                **conversion_kwargs,
            )
            for pdf_path, document_meta in prepared_sources:
                content = self._read_output(output_dir=output_dir, pdf_path=pdf_path)
                documents.append(Document(content=content, meta={**document_meta, "output_format": self.output_format}))

        if self.image_output_dir is not None:
            image_files_after = self._file_state(self.image_output_dir)
            image_documents = [
                Document(meta={"file_path": str(path)})
                for path, state in image_files_after.items()
                if image_files_before.get(path) != state
            ]

        return {"documents": documents, "image_documents": image_documents}
