# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path
from typing import Any, Literal

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace

from paddleocr import Model, PaddleOCRClient, PaddleOCRVLOptions  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

FileTypeInput = Literal["pdf", "image"] | None

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_PDF_EXTENSIONS = {".pdf"}
_EXTENSION_FOR_FILE_TYPE = {0: ".pdf", 1: ".jpg"}


def _infer_file_type_from_source(
    source: str | Path | ByteStream,
    mime_type: str | None = None,
) -> int | None:
    """
    Infer file type from file extension or MIME type.

    :param source:
        Original source (file path, Path object, or ByteStream).
    :param mime_type:
        MIME type of the source.
    :returns:
        Inferred file type: 0 for PDF, 1 for image, or None if cannot be determined.
    """
    file_path: str | None = None

    if isinstance(source, (str, Path)):
        file_path = str(source)
    elif isinstance(source, ByteStream) and source.meta:
        file_path = source.meta.get("file_path")

    if file_path:
        extension = Path(file_path).suffix.lower()
        if extension in _PDF_EXTENSIONS:
            return 0
        if extension in _IMAGE_EXTENSIONS:
            return 1

    if mime_type:
        mime_lower = mime_type.lower()
        if mime_lower == "application/pdf":
            return 0
        if mime_lower.startswith("image/"):
            return 1

    return None


def _normalize_file_type(file_type: FileTypeInput) -> int | None:
    """
    Normalize file type input to the numeric format used internally.

    :param file_type:
        "pdf", "image", or None for auto-detection.
        Integers 0 and 1 are also accepted for deserialization round-trips.
    :returns:
        0 for PDF, 1 for image, or None for auto-detection.
    """
    if file_type is None:
        return None
    if file_type in ("pdf", 0):
        return 0
    if file_type in ("image", 1):
        return 1
    if isinstance(file_type, str):
        msg = f"Invalid `file_type` string: {file_type}. Must be 'pdf' or 'image'."
        raise ValueError(msg)
    msg = f"Invalid `file_type` value: {file_type}. Must be 'pdf', 'image', or `None`."
    raise ValueError(msg)


@component
class PaddleOCRVLDocumentConverter:
    """
    Extracts text from documents using PaddleOCR's official document parsing API.

    Uses `PaddleOCRClient` to parse documents via the PaddleOCR serving API.
    For more information, please refer to:
    https://www.paddleocr.ai/latest/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html

    **Usage Example:**

    ```python
    from haystack_integrations.components.converters.paddleocr import PaddleOCRVLDocumentConverter

    converter = PaddleOCRVLDocumentConverter(
        base_url="http://xxxxx.aistudio-app.com",
    )
    result = converter.run(sources=["sample.pdf"])
    documents = result["documents"]
    raw_responses = result["raw_paddleocr_responses"]
    ```
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        access_token: Secret = Secret.from_env_var("PADDLEOCR_ACCESS_TOKEN"),
        model: Model | str = Model.PADDLE_OCR_VL_16,
        file_type: FileTypeInput = None,
        use_doc_orientation_classify: bool | None = None,
        use_doc_unwarping: bool | None = None,
        use_layout_detection: bool | None = None,
        use_chart_recognition: bool | None = None,
        use_seal_recognition: bool | None = None,
        use_ocr_for_image_block: bool | None = None,
        layout_threshold: float | dict | None = None,
        layout_nms: bool | None = None,
        layout_unclip_ratio: float | list | dict | None = None,
        layout_merge_bboxes_mode: str | dict | None = None,
        layout_shape_mode: str | None = None,
        prompt_label: str | None = None,
        format_block_content: bool | None = None,
        repetition_penalty: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens: int | None = None,
        merge_layout_blocks: bool | None = None,
        markdown_ignore_labels: list[str] | None = None,
        vlm_extra_args: dict | None = None,
        prettify_markdown: bool | None = None,
        show_formula_number: bool | None = None,
        restructure_pages: bool | None = None,
        merge_tables: bool | None = None,
        relevel_titles: bool | None = None,
        visualize: bool | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a `PaddleOCRVLDocumentConverter` component.

        :param base_url:
            Base URL for the PaddleOCR API. Falls back to `PADDLEOCR_BASE_URL`
            env var, then the SDK default.
        :param access_token:
            PaddleOCR access token. Falls back to `PADDLEOCR_ACCESS_TOKEN` env var.
        :param model:
            Document parsing model. Defaults to `Model.PADDLE_OCR_VL_16`.
        :param file_type:
            "pdf", "image", or None for auto-detection.
        :param use_doc_orientation_classify:
            Enable document orientation classification.
        :param use_doc_unwarping:
            Enable text image unwarping.
        :param use_layout_detection:
            Enable layout detection.
        :param use_chart_recognition:
            Enable chart recognition.
        :param use_seal_recognition:
            Enable seal recognition.
        :param use_ocr_for_image_block:
            Recognize text in image blocks.
        :param layout_threshold:
            Layout detection threshold.
        :param layout_nms:
            Perform NMS on layout detection results.
        :param layout_unclip_ratio:
            Layout unclip ratio.
        :param layout_merge_bboxes_mode:
            Layout merge bounding boxes mode.
        :param layout_shape_mode:
            Layout shape mode.
        :param prompt_label:
            Prompt type for the VLM ("ocr", "formula", "table", "chart", "seal", "spotting").
        :param format_block_content:
            Format block content.
        :param repetition_penalty:
            Repetition penalty for VLM sampling.
        :param temperature:
            Temperature for VLM sampling.
        :param top_p:
            Top-p for VLM sampling.
        :param min_pixels:
            Minimum pixels for VLM preprocessing.
        :param max_pixels:
            Maximum pixels for VLM preprocessing.
        :param max_new_tokens:
            Maximum tokens generated by the VLM.
        :param merge_layout_blocks:
            Merge layout detection boxes for cross-column content.
        :param markdown_ignore_labels:
            Layout labels to ignore in Markdown output.
        :param vlm_extra_args:
            Extra configuration for the VLM.
        :param prettify_markdown:
            Prettify output Markdown.
        :param show_formula_number:
            Include formula numbers in Markdown output.
        :param restructure_pages:
            Restructure results across multiple pages.
        :param merge_tables:
            Merge tables across pages.
        :param relevel_titles:
            Relevel titles.
        :param visualize:
            Return visualization results.
        :param additional_params:
            Extra options passed to `PaddleOCRVLOptions.extra_options`.
        """
        self.base_url = base_url
        self.access_token = access_token
        self.model = model
        self.file_type = _normalize_file_type(file_type)
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_layout_detection = use_layout_detection
        self.use_chart_recognition = use_chart_recognition
        self.use_seal_recognition = use_seal_recognition
        self.use_ocr_for_image_block = use_ocr_for_image_block
        self.layout_threshold = layout_threshold
        self.layout_nms = layout_nms
        self.layout_unclip_ratio = layout_unclip_ratio
        self.layout_merge_bboxes_mode = layout_merge_bboxes_mode
        self.layout_shape_mode = layout_shape_mode
        self.prompt_label = prompt_label
        self.format_block_content = format_block_content
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.merge_layout_blocks = merge_layout_blocks
        self.markdown_ignore_labels = markdown_ignore_labels
        self.vlm_extra_args = vlm_extra_args
        self.prettify_markdown = prettify_markdown
        self.show_formula_number = show_formula_number
        self.restructure_pages = restructure_pages
        self.merge_tables = merge_tables
        self.relevel_titles = relevel_titles
        self.visualize = visualize
        self.additional_params = additional_params

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            base_url=self.base_url,
            access_token=self.access_token.to_dict(),
            model=self.model if isinstance(self.model, str) else self.model.value,
            file_type=self.file_type,
            use_doc_orientation_classify=self.use_doc_orientation_classify,
            use_doc_unwarping=self.use_doc_unwarping,
            use_layout_detection=self.use_layout_detection,
            use_chart_recognition=self.use_chart_recognition,
            use_seal_recognition=self.use_seal_recognition,
            use_ocr_for_image_block=self.use_ocr_for_image_block,
            layout_threshold=self.layout_threshold,
            layout_nms=self.layout_nms,
            layout_unclip_ratio=self.layout_unclip_ratio,
            layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
            layout_shape_mode=self.layout_shape_mode,
            prompt_label=self.prompt_label,
            format_block_content=self.format_block_content,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            max_new_tokens=self.max_new_tokens,
            merge_layout_blocks=self.merge_layout_blocks,
            markdown_ignore_labels=self.markdown_ignore_labels,
            vlm_extra_args=self.vlm_extra_args,
            prettify_markdown=self.prettify_markdown,
            show_formula_number=self.show_formula_number,
            restructure_pages=self.restructure_pages,
            merge_tables=self.merge_tables,
            relevel_titles=self.relevel_titles,
            visualize=self.visualize,
            additional_params=self.additional_params,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaddleOCRVLDocumentConverter":
        """
        Deserialize the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["access_token"])
        return default_from_dict(cls, data)

    def _build_options(self) -> PaddleOCRVLOptions:
        return PaddleOCRVLOptions(
            use_doc_orientation_classify=self.use_doc_orientation_classify,
            use_doc_unwarping=self.use_doc_unwarping,
            use_layout_detection=self.use_layout_detection,
            use_chart_recognition=self.use_chart_recognition,
            use_seal_recognition=self.use_seal_recognition,
            use_ocr_for_image_block=self.use_ocr_for_image_block,
            layout_threshold=self.layout_threshold,
            layout_nms=self.layout_nms,
            layout_unclip_ratio=self.layout_unclip_ratio,
            layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
            layout_shape_mode=self.layout_shape_mode,
            prompt_label=self.prompt_label,
            format_block_content=self.format_block_content,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            max_new_tokens=self.max_new_tokens,
            merge_layout_blocks=self.merge_layout_blocks,
            markdown_ignore_labels=self.markdown_ignore_labels,
            vlm_extra_args=self.vlm_extra_args,
            prettify_markdown=self.prettify_markdown,
            show_formula_number=self.show_formula_number,
            restructure_pages=self.restructure_pages,
            merge_tables=self.merge_tables,
            relevel_titles=self.relevel_titles,
            visualize=self.visualize,
            extra_options=self.additional_params,
        )

    def _parse(self, data: bytes, file_type: int, client: PaddleOCRClient) -> tuple[str, dict[str, Any]]:
        extension = _EXTENSION_FOR_FILE_TYPE[file_type]
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(data)
        try:
            result = client.parse_document(
                model=self.model,
                file_path=tmp_path,
                options=self._build_options(),
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        text = "\f".join(page.markdown_text for page in result.pages)
        raw: dict[str, Any] = {
            "job_id": result.job_id,
            "pages": [page.raw for page in result.pages],
            "data_info": result.data_info,
        }
        return text, raw

    @component.output_types(documents=list[Document], raw_paddleocr_responses=list[dict[str, Any]])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Convert image or PDF files to Documents.

        :param sources:
            List of image or PDF file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents. A single dict is applied
            to all documents; a list must match the number of sources.
        :returns:
            A dictionary with:
            - `documents`: List of created Documents.
            - `raw_paddleocr_responses`: List of raw PaddleOCR API responses.
        """
        documents: list[Document] = []
        raw_responses: list[dict[str, Any]] = []

        meta_list = normalize_metadata(meta, sources_count=len(sources))
        token = self.access_token.resolve_value() if self.access_token else None

        kwargs: dict[str, Any] = {"client_platform": "haystack", "token": token}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url

        with PaddleOCRClient(**kwargs) as client:
            for source, metadata in zip(sources, meta_list, strict=True):
                try:
                    bytestream = get_bytestream_from_source(source)
                except Exception as e:
                    logger.warning(f"Could not read {source}. Skipping it. Error: {e}")
                    continue

                if self.file_type is not None:
                    file_type: int | None = self.file_type
                else:
                    mime_type = bytestream.mime_type if bytestream.mime_type else None
                    file_type = _infer_file_type_from_source(source, mime_type)

                if file_type is None:
                    logger.warning(f"Could not determine file type for {source}. Skipping it.")
                    continue

                try:
                    text, raw_resp = self._parse(bytestream.data, file_type, client)
                except Exception as e:
                    logger.warning(f"Could not convert {source} to Document, skipping. Error: {e}")
                    continue

                if not text:
                    logger.warning(
                        f"{self.__class__.__name__} could not extract text from {source}. Returning an empty document."
                    )

                merged_metadata = {**bytestream.meta, **metadata}
                documents.append(Document(content=text, meta=merged_metadata))
                raw_responses.append(raw_resp)

        return {"documents": documents, "raw_paddleocr_responses": raw_responses}
