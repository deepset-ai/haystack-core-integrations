# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal, get_args

import minicpmvocr
from haystack import Document, component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.utils import Secret
from minicpmvocr import assemble_document_markdown, infer_run, load_config, load_prompt_map, prepare_run
from minicpmvocr.layouts.download import ensure_layout_model

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
# Backend names accepted by the MiniCPM-OCR SDK. `openai` also covers SGLang
# and LMDeploy, which expose the same chat-completions API.
Backend = Literal["vllm", "llamacpp", "ollama", "openai"]
_SUFFIX_BY_MIME = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}
# Same lookup the SDK's own CLI uses when no config file is given.
_SDK_CONFIG_PATH = Path(minicpmvocr.__file__).parent / "configs" / "minicpmvocr.yaml"
_DEFAULT_LAYOUT_MODEL_DIR = Path.home() / ".cache" / "minicpm-ocr" / "layout"
# Form feed separates pages, so `DocumentSplitter(split_by="page")` can split the output.
_PAGE_SEPARATOR = "\f"


def _resolve_suffix(source: str | Path | ByteStream, mime_type: str | None) -> str | None:
    """
    Infer a file suffix from the source path or its MIME type.

    :param source:
        Original source object.
    :param mime_type:
        MIME type reported by the ByteStream, if any.
    :returns:
        A supported suffix, or None when the file type cannot be determined.
    """
    path: str | None = None
    if isinstance(source, (str, Path)):
        path = str(source)
    elif isinstance(source, ByteStream) and source.meta.get("file_path"):
        path = str(source.meta["file_path"])

    if path:
        suffix = Path(path).suffix.lower()
        if suffix in _SUPPORTED_SUFFIXES:
            return suffix

    if mime_type:
        return _SUFFIX_BY_MIME.get(mime_type.lower())
    return None


@component
class MiniCPMOCRDocumentConverter:
    """
    Converts PDF and image files to Documents using self-hosted MiniCPM-OCR.

    The component runs the official `minicpm-ocr` SDK locally: it detects the
    layout of every page with PP-DocLayout, crops each block, sends it to a
    MiniCPM-OCR inference server, and assembles the recognized blocks into
    Markdown. The inference server can run on vLLM, SGLang, llama.cpp or
    Ollama; only the layout model runs in this process.

    Returns one Document per source, with pages separated by form feed characters (`\\f`),
    so `DocumentSplitter` can split the content by page.

    ### Usage example

    ```python
    from haystack_integrations.components.converters.minicpm_ocr import MiniCPMOCRDocumentConverter

    converter = MiniCPMOCRDocumentConverter(server_url="http://localhost:8000", model="minicpm-ocr")
    converter.warm_up()
    result = converter.run(sources=["sample.pdf"])
    documents = result["documents"]
    ```
    """

    def __init__(
        self,
        *,
        server_url: str,
        model: str = "minicpm-ocr",
        backend: Backend = "vllm",
        api_key: Secret = Secret.from_env_var("MINICPM_OCR_API_KEY", strict=False),
        layout_model_dir: str | None = None,
        layout_device: str = "cpu",
        request_timeout: float = 300.0,
        page_workers: int = 1,
        max_workers: int = 1,
        store_full_path: bool = False,
    ) -> None:
        """
        Create a MiniCPM-OCR document converter.

        :param server_url:
            Base URL of the MiniCPM-OCR inference server, for example
            `http://localhost:8000` for vLLM or `http://localhost:11434`
            for Ollama. A trailing `/v1` is accepted and removed, because the
            SDK appends `/v1/chat/completions` itself.
        :param model:
            Model name served by the inference server.
        :param backend:
            Server dialect: `vllm`, `llamacpp`, `ollama` or `openai`
            (SGLang and LMDeploy use `openai`).
        :param api_key:
            Optional API key for the inference server.
        :param layout_model_dir:
            Directory holding the PP-DocLayout model. When unset, the SDK uses
            `~/.cache/minicpm-ocr/layout`. The model is downloaded there in
            `warm_up()` if it is missing.
        :param layout_device:
            Device for layout detection, for example `cpu` or `cuda:0`.
        :param request_timeout:
            Timeout in seconds for one recognition request.
        :param page_workers:
            Pages rendered and detected concurrently.
        :param max_workers:
            Recognition requests in flight concurrently.
        :param store_full_path:
            If True, the full path of the source file is stored in the Document metadata.
            If False, only the file name is stored.
        :raises ValueError:
            If `server_url` is empty, `backend` is unknown, or a worker
            count is not positive.
        """
        normalized_url = server_url.strip().rstrip("/") if server_url else ""
        if normalized_url.endswith("/v1"):
            normalized_url = normalized_url[: -len("/v1")]
        if not normalized_url:
            msg = "server_url must be a non-empty base URL, for example http://localhost:8000"
            raise ValueError(msg)
        # Checked at runtime too: pipelines loaded from YAML bypass the type hint
        if backend not in get_args(Backend):
            msg = f"Unsupported backend {backend!r}. Choose one of: {', '.join(get_args(Backend))}."
            raise ValueError(msg)
        if page_workers < 1 or max_workers < 1:
            msg = "page_workers and max_workers must be at least 1"
            raise ValueError(msg)

        self.server_url = normalized_url
        self.model = model
        self.backend = backend
        self.api_key = api_key
        self.layout_model_dir = layout_model_dir
        self.layout_device = layout_device
        self.request_timeout = request_timeout
        self.page_workers = page_workers
        self.max_workers = max_workers
        self.store_full_path = store_full_path

        self._layout_model_path: Path | None = None

    def warm_up(self) -> None:
        """
        Make sure the PP-DocLayout layout model is available locally, downloading it on first use.
        """
        if self._layout_model_path is None:
            layout_model_path = (
                Path(self.layout_model_dir).expanduser() if self.layout_model_dir else _DEFAULT_LAYOUT_MODEL_DIR
            )
            ensure_layout_model(layout_model_path)
            self._layout_model_path = layout_model_path

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Convert PDF or image files to Documents.

        :param sources:
            List of file paths or ByteStream objects. Supported formats are PDF,
            PNG, JPEG, BMP, TIFF and WebP.
        :param meta:
            Optional metadata to attach to the Documents. A single dict is
            applied to all documents; a list must match the number of sources.
        :returns:
            A dictionary with the following key:
            - `documents`: One Document per converted source. The content is Markdown with pages
              separated by `\\f`, and the metadata includes `page_count` and `ocr_engine`.
        """
        if self._layout_model_path is None:
            self.warm_up()

        documents: list[Document] = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))
        api_key = self.api_key.resolve_value() if self.api_key else None

        with tempfile.TemporaryDirectory(prefix="minicpm-ocr-haystack-") as temp_dir:
            for source, metadata in zip(sources, meta_list, strict=True):
                try:
                    bytestream = get_bytestream_from_source(source)
                except Exception as error:
                    logger.warning(
                        "Could not read {source}. Skipping it. Error: {error}",
                        source=source,
                        error=error,
                    )
                    continue

                mime_type = bytestream.mime_type if bytestream.mime_type else None
                suffix = _resolve_suffix(source, mime_type)
                if suffix is None:
                    logger.warning(
                        "Could not determine a supported file type for {source}. Skipping it.",
                        source=source,
                    )
                    continue

                file_path = Path(temp_dir) / f"input-{uuid.uuid4().hex}{suffix}"
                file_path.write_bytes(bytestream.data)

                try:
                    markdown, page_count = self._parse(file_path, Path(temp_dir), api_key)
                except Exception as error:
                    logger.warning(
                        "Could not convert {source} to a Document. Skipping it. Error: {error}",
                        source=source,
                        error=error,
                    )
                    continue

                if not markdown.strip():
                    logger.warning(
                        "MiniCPM-OCR returned no text for {source}. Returning an empty document.",
                        source=source,
                    )

                merged_metadata = {**bytestream.meta, **metadata}
                if not self.store_full_path and (source_file_path := bytestream.meta.get("file_path")):
                    merged_metadata["file_path"] = os.path.basename(source_file_path)

                documents.append(
                    Document(
                        content=markdown,
                        meta={
                            **merged_metadata,
                            "page_count": page_count,
                            "ocr_engine": "minicpm-ocr",
                        },
                    )
                )

        return {"documents": documents}

    def _parse(self, file_path: Path, temp_dir: Path, api_key: str | None) -> tuple[str, int]:
        """
        Run the MiniCPM-OCR pipeline for one file.

        :param file_path:
            Temporary file holding the source bytes.
        :param temp_dir:
            Directory used as the SDK run root.
        :param api_key:
            Resolved API key, or None.
        :returns:
            The page Markdown joined with `\\f` and the page count.
        :raises RuntimeError:
            If the SDK produced no pages for the file.
        """
        overrides = {
            "run": {
                "run_tag": f"haystack_{uuid.uuid4().hex}",
                "run_root": str(temp_dir),
                "mode": "oneshot_file",
            },
            "input": {
                "file_path": str(file_path),
                "manifest": "",
                "image_dir": "",
                "files_dir": "",
                "limit": 0,
            },
            "layout": {
                "enabled": True,
                "model_dir": str(self._layout_model_path),
                "device": self.layout_device,
            },
            "engine": {
                "type": self.backend,
                "model_name": self.model,
                "server_url": self.server_url,
                "api_key": api_key or "",
                "request_timeout_sec": self.request_timeout,
            },
            "upright": {"enabled": False},
            "pipeline": {
                "overlap_layout": False,
                "page_workers": self.page_workers,
                "max_workers": self.max_workers,
            },
            "logging": {"progress_log_every": 10**9},
        }
        config = load_config(str(_SDK_CONFIG_PATH), overrides=overrides)

        prepare_run(config)
        infer_run(config)

        markdown, page_count, _ = assemble_document_markdown(
            config["_meta"]["run_dir"],
            load_prompt_map(config),
            merge_tables=False,
            delimiter=_PAGE_SEPARATOR,
            cfg=config,
        )
        if page_count == 0:
            msg = f"MiniCPM-OCR did not produce any page for {file_path}"
            raise RuntimeError(msg)

        return markdown, page_count
