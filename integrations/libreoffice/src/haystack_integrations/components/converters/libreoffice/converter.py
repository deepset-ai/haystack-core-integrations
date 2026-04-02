# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
import os
import shutil
import subprocess
from asyncio import create_subprocess_exec
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, Literal, TypedDict, get_args

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream
from typing_extensions import Self

OUTPUT_FILE_TYPE = Literal[
    "pdf",
    "doc",
    "docx",
    "odt",
    "rtf",
    "txt",
    "html",
    "xlsx",
    "xls",
    "ods",
    "csv",
    "pptx",
    "ppt",
    "odp",
    "epub",
    "png",
    "jpg",
]


class LibreOfficeFileConverterOutput(TypedDict):
    output: list[ByteStream]


@component
class LibreOfficeFileConverter:
    """
    Component that uses libreoffice's command line utility (soffice) to convert files into various formats.

    ### Usage examples

    **Simple conversion:**
    ```python
    from pathlib import Path

    from haystack_integrations.components.converters.libreoffice import LibreOfficeFileConverter

    # Convert documents
    converter = LibreOfficeFileConverter()
    results = converter.run(sources=[Path("sample.doc")], output_file_type="docx")
    print(results["output"])  # [ByteStream(data=b'...', meta={}, mime_type=None)]
    ```

    **Conversion pipeline:**
    ```python
    from pathlib import Path

    from haystack import Pipeline
    from haystack.components.converters import DOCXToDocument

    from haystack_integrations.components.converters.libreoffice import LibreOfficeFileConverter

    # Create pipeline with components
    pipeline = Pipeline()
    pipeline.add_component("libreoffice_converter", LibreOfficeFileConverter())
    pipeline.add_component("docx_converter", DOCXToDocument())

    pipeline.connect("libreoffice_converter.output", "docx_converter.sources")

    # Run pipeline and convert legacy documents into Haystack documents
    results = pipeline.run(
        {
            "libreoffice_converter": {
                "sources": [Path("sample_doc.doc")],
                "output_file_type": "docx",
            }
        }
    )
    print(results["docx_converter"]["documents"])
    ```
    """

    SUPPORTED_TYPES: ClassVar[dict[str, frozenset[str]]] = {
        # Documents
        "doc": frozenset(["pdf", "docx", "odt", "rtf", "txt", "html", "epub"]),
        "docx": frozenset(["pdf", "doc", "odt", "rtf", "txt", "html", "epub"]),
        "odt": frozenset(["pdf", "docx", "doc", "rtf", "txt", "html", "epub"]),
        "rtf": frozenset(["pdf", "docx", "doc", "odt", "txt", "html"]),
        "txt": frozenset(["pdf", "docx", "doc", "odt", "rtf", "html"]),
        "html": frozenset(["pdf", "docx", "doc", "odt", "rtf", "txt"]),
        # Spreadsheets
        "xlsx": frozenset(["pdf", "xls", "ods", "csv", "html"]),
        "xls": frozenset(["pdf", "xlsx", "ods", "csv", "html"]),
        "ods": frozenset(["pdf", "xlsx", "xls", "csv", "html"]),
        "csv": frozenset(["pdf", "xlsx", "xls", "ods"]),
        # Presentations
        "pptx": frozenset(["pdf", "ppt", "odp", "html", "png", "jpg"]),
        "ppt": frozenset(["pdf", "pptx", "odp", "html", "png", "jpg"]),
        "odp": frozenset(["pdf", "pptx", "ppt", "html", "png", "jpg"]),
    }
    """A non-exhaustive mapping of supported conversion types by this component.
    See https://help.libreoffice.org/latest/en-GB/text/shared/guide/convertfilters.html for more information."""

    MIME_TYPE_FALLBACKS: ClassVar[dict[str, str]] = {
        "pdf": "application/pdf",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "odt": "application/vnd.oasis.opendocument.text",
        "rtf": "application/rtf",
        "txt": "text/plain",
        "html": "text/html",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
        "ods": "application/vnd.oasis.opendocument.spreadsheet",
        "csv": "text/csv",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "ppt": "application/vnd.ms-powerpoint",
        "odp": "application/vnd.oasis.opendocument.presentation",
        "epub": "application/epub+zip",
        "png": "image/png",
        "jpg": "image/jpeg",
    }

    def __init__(
        self,
        output_file_type: OUTPUT_FILE_TYPE | None = None,
    ) -> None:
        """
        Check whether soffice is installed.

        :param output_file_type:
            Target file format to convert to. Must be a valid conversion target for
            each source's input type — see :attr:`SUPPORTED_TYPES` for the full mapping.
        """
        soffice_path = shutil.which("soffice")
        if soffice_path is None:
            msg = """LibreOffice (soffice) is required but not installed or not in PATH.

- Install instructions: https://www.libreoffice.org/get-help/install-howto/"""
            raise FileNotFoundError(msg)

        self.soffice_path = soffice_path
        self.output_file_type = output_file_type

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    def _get_conversion_args(
        self, source: str | Path, output_directory: str | Path, output_file_type: str
    ) -> tuple[Path, list[str]]:
        """
        Validate source file and return the soffice arguments for conversion.

        :param source: Source file path.
        :param output_directory: Output directory to save converted files to.
        :param output_file_type: Target file format extension (e.g. `"pdf"`).
        :returns: Tuple of `(output_path, soffice_args)` where `output_path` is the
            expected path of the converted file and `soffice_args` is the list of
            arguments to pass to `soffice`.
        :raises FileNotFoundError: If `source` does not exist.
        :raises OSError: If `output_directory` does not exist or is not writable.
        """
        source_path = Path(source)
        output_path = Path(output_directory)

        # Source file must exist
        if not source_path.is_file():
            msg = f"{source=} does not exist"
            raise FileNotFoundError(msg)

        # Output directory must exist and be writable
        if not output_path.is_dir() or not os.access(output_path, os.W_OK):
            msg = f"{output_directory=} must exist and be writable"
            raise OSError(msg)

        args = [
            self.soffice_path,
            "--headless",
            "--convert-to",
            output_file_type,
            "--outdir",
            str(output_directory),
            str(source),
        ]
        return (output_path / source_path.name).with_suffix(f".{output_file_type}"), args

    def _validate_args(self, output_file_type: str, input_file_type: str | None = None) -> None:
        """
        Validate that the input and output file types are supported.

        :param output_file_type: Target file format extension to convert to.
        :param input_file_type: Source file format extension. If provided, validates that
            it is a supported input type and that `output_file_type` is a valid conversion
            target for it.
        :raises ValueError: If `input_file_type` is not in :attr:`SUPPORTED_TYPES`, or if
            `output_file_type` is not a valid conversion target for the given `input_file_type`.
        """
        # Validate specified output type is one of allow output file types
        supported_output_types = get_args(OUTPUT_FILE_TYPE)
        if output_file_type not in supported_output_types:
            supported_types = ", ".join(supported_output_types)
            msg = f"{output_file_type=} is not supported and must be one of type {supported_types}"
            raise ValueError(msg)

        # Cannot further validate conversion types if input conversions is not known - i.e., source is `ByteStream`
        if input_file_type is None:
            return

        if input_file_type not in self.SUPPORTED_TYPES:
            supported_types = ", ".join(self.SUPPORTED_TYPES)
            msg = f"{input_file_type=} is not supported and must be one of type {supported_types}"
            raise ValueError(msg)

        if output_file_type not in (output_types := self.SUPPORTED_TYPES[input_file_type]):
            supported_types = ", ".join(output_types)
            msg = (
                f"{output_file_type=} is not supported for {input_file_type=} and must be one of type {supported_types}"
            )
            raise ValueError(msg)

    def _resolve_mime_type(self, output_path: Path, output_file_type: str) -> str | None:
        mime_type, _ = mimetypes.guess_type(str(output_path))
        if mime_type is None:
            return self.MIME_TYPE_FALLBACKS.get(output_file_type)
        return mime_type

    @component.output_types(output=list[ByteStream])
    def run(
        self,
        sources: Iterable[str | Path | ByteStream],
        output_file_type: OUTPUT_FILE_TYPE | None = None,
    ) -> LibreOfficeFileConverterOutput:
        """
        Convert office files to the specified output format using LibreOffice.

        :param sources:
            List of sources to convert. Each source can be a file path (`str` or
            `Path`) or a `ByteStream`. For `ByteStream` sources, the input file
            type cannot be inferred from the filename, so only `output_file_type` is
            validated (not the source type).
        :param output_file_type:
            Target file format to convert to. Must be a valid conversion target for
            each source's input type — see :attr:`SUPPORTED_TYPES` for the full mapping.
            If set, it will override the `output_file_type` parameter provided during initialization.
        :returns:
            A dictionary with the following key:
            - `output`: List of `ByteStream` objects containing the converted file
              data, in the same order as `sources`.
        :raises FileNotFoundError: If a source file path does not exist.
        :raises OSError: If the internal temporary output directory is not writable.
        :raises ValueError: If a source's file type is not in :attr:`SUPPORTED_TYPES`,
            or if `output_file_type` is not a valid conversion target for it,
            or if `output_file_type` has not been provided anywhere.
        """
        resolved_output_file_type = output_file_type or self.output_file_type
        if resolved_output_file_type is None:
            msg = "output_file_type must be provided either during initialization or for this method"
            raise ValueError(msg)

        outputs: list[ByteStream] = []
        with TemporaryDirectory() as tmpdir:
            for source in sources:
                # Handle case where source is a `ByteStream` using tempfile
                if isinstance(source, ByteStream):
                    tmp_path = Path(tmpdir) / "input"
                    tmp_path.write_bytes(source.data)

                    self._validate_args(resolved_output_file_type)
                    output_path, args = self._get_conversion_args(tmp_path, tmpdir, resolved_output_file_type)

                    subprocess.run(args, check=True)  # noqa: S603 - ruff doesn't know the arguments have been validated
                    outputs.append(
                        ByteStream(
                            data=output_path.read_bytes(),
                            mime_type=self._resolve_mime_type(output_path, resolved_output_file_type),
                        )
                    )
                    continue

                self._validate_args(resolved_output_file_type, str(source).split(".")[-1])
                output_path, args = self._get_conversion_args(source, tmpdir, resolved_output_file_type)

                subprocess.run(args, check=True)  # noqa: S603
                outputs.append(
                    ByteStream(
                        data=output_path.read_bytes(),
                        mime_type=self._resolve_mime_type(output_path, resolved_output_file_type),
                    )
                )

        return {"output": outputs}

    @component.output_types(output=list[ByteStream])
    async def run_async(
        self,
        sources: Iterable[str | Path | ByteStream],
        output_file_type: OUTPUT_FILE_TYPE | None = None,
    ) -> LibreOfficeFileConverterOutput:
        """
        Asynchronously convert office files to the specified output format using LibreOffice.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param sources:
            List of sources to convert. Each source can be a file path (`str` or
            `Path`) or a `ByteStream`. For `ByteStream` sources, the input file
            type cannot be inferred from the filename, so only `output_file_type` is
            validated (not the source type).
        :param output_file_type:
            Target file format to convert to. Must be a valid conversion target for
            each source's input type — see :attr:`SUPPORTED_TYPES` for the full mapping.
            If set, it will override the `output_file_type` parameter provided during initialization.
        :returns:
            A dictionary with the following key:
            - `output`: List of `ByteStream` objects containing the converted file
              data, in the same order as `sources`.
        :raises FileNotFoundError: If a source file path does not exist.
        :raises OSError: If the internal temporary output directory is not writable.
        :raises ValueError: If a source's file type is not in :attr:`SUPPORTED_TYPES`,
            or if `output_file_type` is not a valid conversion target for it,
            or if `output_file_type` has not been provided anywhere.
        """
        resolved_output_file_type = output_file_type or self.output_file_type
        if resolved_output_file_type is None:
            msg = "output_file_type must be provided either during initialization or for this method"
            raise ValueError(msg)

        outputs: list[ByteStream] = []
        with TemporaryDirectory() as tmpdir:
            for source in sources:
                # Handle case where source is a `ByteStream`
                if isinstance(source, ByteStream):
                    tmp_path = Path(tmpdir) / "input"
                    tmp_path.write_bytes(source.data)

                    self._validate_args(resolved_output_file_type)
                    output_path, args = self._get_conversion_args(tmp_path, tmpdir, resolved_output_file_type)

                    process = await create_subprocess_exec(*args)
                    # Wait for process to complete as only one instance of soffice can occur at once
                    await process.wait()
                    outputs.append(
                        ByteStream(
                            data=output_path.read_bytes(),
                            mime_type=self._resolve_mime_type(output_path, resolved_output_file_type),
                        )
                    )
                    continue

                self._validate_args(resolved_output_file_type, str(source).split(".")[-1])
                output_path, args = self._get_conversion_args(source, tmpdir, resolved_output_file_type)

                process = await create_subprocess_exec(*args)
                # Wait for process to complete as only one instance of soffice can occur at once
                await process.wait()

                outputs.append(
                    ByteStream(
                        data=output_path.read_bytes(),
                        mime_type=self._resolve_mime_type(output_path, resolved_output_file_type),
                    )
                )

        return {"output": outputs}
