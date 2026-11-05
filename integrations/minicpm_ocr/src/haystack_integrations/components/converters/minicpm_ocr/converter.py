# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

import minicpmvocr
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace
from minicpmvocr import load_config
from minicpmvocr.layouts.download import ensure_layout_model
from minicpmvocr.pipeline import infer_run, prepare_run

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
# Backend names accepted by the MiniCPM-OCR SDK. ``openai`` also covers SGLang
# and LMDeploy, which expose the same chat-completions API.
_SUPPORTED_BACKENDS = {"vllm", "llamacpp", "ollama", "openai"}
_SUFFIX_BY_MIME = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}


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

    **Usage Example:**

    ```python
    from haystack_integrations.components.converters.minicpm_ocr import (
        MiniCPMOCRDocumentConverter,
    )

    converter = MiniCPMOCRDocumentConverter(
        server_url="http://localhost:8000",
        model="minicpm-ocr",
    )
    result = converter.run(sources=["sample.pdf"])
    documents = result["documents"]
    ```
    """

    def __init__(
        self,
        *,
        server_url: str,
        model: str = "minicpm-ocr",
        backend: str = "vllm",
        api_key: Secret = Secret.from_env_var("MINICPM_OCR_API_KEY", strict=False),
        layout_model_dir: str | None = None,
        layout_device: str = "cpu",
        request_timeout: float = 300.0,
        page_workers: int = 1,
        max_workers: int = 1,
    ) -> None:
        """
        Create a MiniCPM-OCR document converter.

        :param server_url:
            Base URL of the MiniCPM-OCR inference server, for example
            ``http://localhost:8000`` for vLLM or ``http://localhost:11434``
            for Ollama. A trailing ``/v1`` is accepted and removed, because the
            SDK appends ``/v1/chat/completions`` itself.
        :param model:
            Model name served by the inference server.
        :param backend:
            Server dialect: ``vllm``, ``llamacpp``, ``ollama`` or ``openai``
            (SGLang and LMDeploy use ``openai``).
        :param api_key:
            Optional API key for the inference server.
        :param layout_model_dir:
            Directory holding the PP-DocLayout model. When unset, the SDK uses
            ``~/.cache/minicpm-ocr/layout`` and downloads the model there on
            first use.
        :param layout_device:
            Device for layout detection, for example ``cpu`` or ``cuda:0``.
        :param request_timeout:
            Timeout in seconds for one recognition request.
        :param page_workers:
            Pages rendered and detected concurrently.
        :param max_workers:
            Recognition requests in flight concurrently.
        :raises ValueError:
            If ``server_url`` is empty, ``backend`` is unknown, or a worker
            count is not positive.
        """
        normalized_url = server_url.strip().rstrip("/") if server_url else ""
        if normalized_url.endswith("/v1"):
            normalized_url = normalized_url[: -len("/v1")]
        if not normalized_url:
            msg = "server_url must be a non-empty base URL, for example http://localhost:8000"
            raise ValueError(msg)
        normalized_backend = backend.strip().lower()
        if normalized_backend not in _SUPPORTED_BACKENDS:
            msg = f"Unsupported backend {backend!r}. Choose one of: {', '.join(sorted(_SUPPORTED_BACKENDS))}."
            raise ValueError(msg)
        if page_workers < 1 or max_workers < 1:
            msg = "page_workers and max_workers must be at least 1"
            raise ValueError(msg)

        self.server_url = normalized_url
        self.model = model
        self.backend = normalized_backend
        self.api_key = api_key
        self.layout_model_dir = layout_model_dir
        self.layout_device = layout_device
        self.request_timeout = request_timeout
        self.page_workers = page_workers
        self.max_workers = max_workers

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            server_url=self.server_url,
            model=self.model,
            backend=self.backend,
            api_key=self.api_key.to_dict(),
            layout_model_dir=self.layout_model_dir,
            layout_device=self.layout_device,
            request_timeout=self.request_timeout,
            page_workers=self.page_workers,
            max_workers=self.max_workers,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MiniCPMOCRDocumentConverter":
        """
        Deserialize the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Convert PDF or image files to Documents.

        :param sources:
            List of file paths or ByteStream objects. Supported formats are PDF,
            PNG, JPEG, BMP, TIFF and WebP.
        :param meta:
            Optional metadata to attach to the Documents. A single dict is
            applied to all documents; a list must match the number of sources.
        :returns:
            A dictionary with the ``documents`` key. Each document contains the
            page Markdown and the metadata ``page_count`` and ``ocr_engine``.
        """
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

                if not markdown:
                    logger.warning(
                        "MiniCPM-OCR returned no text for {source}. Returning an empty document.",
                        source=source,
                    )

                merged_metadata = {**bytestream.meta, **metadata}
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
            The page Markdown joined with blank lines and the page count.
        """
        layout_model_dir = (
            Path(self.layout_model_dir).expanduser()
            if self.layout_model_dir
            else Path.home() / ".cache" / "minicpm-ocr" / "layout"
        )
        ensure_layout_model(layout_model_dir)

        config_path = Path(minicpmvocr.__file__).parent / "configs" / "minicpmvocr.yaml"
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
                "model_dir": str(layout_model_dir),
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
        config = load_config(str(config_path), overrides=overrides)

        prepare_run(config)
        infer_run(config)

        run_dir = Path(config["_meta"]["run_dir"])
        tasks_path = run_dir / "runtime" / "page_tasks.jsonl"
        tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not tasks:
            msg = f"MiniCPM-OCR did not produce page tasks for {file_path}"
            raise RuntimeError(msg)

        pages = []
        for task in tasks:
            page_path = run_dir / "output" / f"{task['image_id']}.md"
            pages.append(page_path.read_text(encoding="utf-8"))

        return "\n\n".join(pages).strip(), len(tasks)
