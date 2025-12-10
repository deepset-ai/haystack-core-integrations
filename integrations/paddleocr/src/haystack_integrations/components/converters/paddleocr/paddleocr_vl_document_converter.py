# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import base64
from pathlib import Path
from typing import Any, Literal, Optional, Union

import requests
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace
from paddlex.inference.serving.schemas.paddleocr_vl import (  # type: ignore[import-untyped]
    InferRequest as PaddleOCRVLInferRequest,
)
from paddlex.inference.serving.schemas.paddleocr_vl import (  # type: ignore[import-untyped]
    InferResult as PaddleOCRVLInferResult,
)
from paddlex.inference.serving.schemas.shared.ocr import FileType  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


FileTypeInput = Union[Literal["pdf", "image"], None]

# Supported image file extensions
_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}
# Supported PDF file extensions
_PDF_EXTENSIONS = {".pdf"}


def _infer_file_type_from_source(
    source: Union[str, Path, ByteStream],
    mime_type: Optional[str] = None,
) -> Optional[FileType]:
    """
    Infer file type from file extension or MIME type.

    :param source:
        Original source (file path, Path object, or ByteStream).
    :param mime_type:
        MIME type of the source.
    :returns:
        Inferred file type: 0 for PDF, 1 for image, or None if cannot be
        determined.
    """
    # Try to get extension from file path
    file_path: Optional[str] = None

    # Check if source is a file path
    if isinstance(source, (str, Path)):
        file_path = str(source)
    # Check if source is `ByteStream` and has `file_path` in metadata
    elif isinstance(source, ByteStream) and source.meta:
        file_path = source.meta.get("file_path")

    # Try to infer from file extension
    if file_path:
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()

        if extension in _PDF_EXTENSIONS:
            return 0
        if extension in _IMAGE_EXTENSIONS:
            return 1

    # Try to infer from MIME type if available
    if mime_type:
        mime_type_lower = mime_type.lower()
        if mime_type_lower == "application/pdf":
            return 0
        if mime_type_lower.startswith("image/"):
            return 1

    return None


def _normalize_file_type(file_type: Optional[FileTypeInput]) -> Optional[FileType]:
    """
    Normalize file type input to the numeric format expected by the API.

    :param file_type:
        File type input. Can be "pdf" for PDF, "image" for image,
        or `None` for auto-detection.
    :returns:
        Normalized file type: 0 for PDF, 1 for image, or `None` for
        auto-detection.
    """
    if file_type is None:
        return None
    if isinstance(file_type, str):
        if file_type.lower() == "pdf":
            return 0
        if file_type.lower() == "image":
            return 1
        msg = f"Invalid `file_type` string: {file_type}. Must be 'pdf' or 'image'."
        raise ValueError(msg)
    msg = f"Invalid `file_type` value: {file_type}. Must be 'pdf', 'image', or `None`."
    raise ValueError(msg)


@component
class PaddleOCRVLDocumentConverter:
    """
    This component extracts text from documents using PaddleOCR's large model
    document parsing API.

    PaddleOCR-VL is used behind the scenes. For more information, please
    refer to:
    https://www.paddleocr.ai/latest/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html

    **Usage Example:**

    ```python
    from haystack.utils import Secret
    from haystack_integrations.components.converters.paddleocr import (
        PaddleOCRVLDocumentConverter,
    )

    converter = PaddleOCRVLDocumentConverter(
        api_url="http://xxxxx.aistudio-app.com/layout-parsing",
        access_token=Secret.from_env_var("AISTUDIO_ACCESS_TOKEN"),
    )

    result = converter.run(sources=["sample.pdf"])

    documents = result["documents"]
    raw_responses = result["raw_paddleocr_responses"]
    ```
    """

    def __init__(
        self,
        *,
        api_url: str,
        access_token: Secret = Secret.from_env_var("AISTUDIO_ACCESS_TOKEN"),
        file_type: Optional[FileTypeInput] = None,
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_layout_detection: Optional[bool] = None,
        use_chart_recognition: Optional[bool] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, dict]] = None,
        prompt_label: Optional[str] = None,
        format_block_content: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        prettify_markdown: Optional[bool] = None,
        show_formula_number: Optional[bool] = None,
        visualize: Optional[bool] = None,
        additional_params: Optional[dict[str, Any]] = None,
    ):
        """
        Create a `PaddleOCRVLDocumentConverter` component.

        :param api_url:
            API URL. To obtain the API URL, visit the [PaddleOCR official
            website](https://aistudio.baidu.com/paddleocr/task), click the
            **API** button in the upper-left corner, choose the example code
            for **Large Model document parsing(PaddleOCR-VL)**, and copy the
            `API_URL`.
        :param access_token:
            AI Studio access token. You can obtain it from [this
            page](https://aistudio.baidu.com/account/accessToken).
        :param file_type:
            File type. Can be "pdf" for PDF files, "image" for
            image files, or `None` for auto-detection. If not specified, the
            file type will be inferred from the file extension.
        :param use_doc_orientation_classify:
            Whether to enable the document orientation classification
            function. Enabling this feature allows the input image to be
            automatically rotated to the correct orientation.
        :param use_doc_unwarping:
            Whether to enable the text image unwarping function. Enabling
            this feature allows automatic correction of distorted text images.
        :param use_layout_detection:
            Whether to enable the layout detection function.
        :param use_chart_recognition:
            Whether to enable the chart recognition function.
        :param layout_threshold:
            Layout detection threshold. Can be a float or a dict with
            page-specific thresholds.
        :param layout_nms:
            Whether to perform NMS (Non-Maximum Suppression) on layout
            detection results.
        :param layout_unclip_ratio:
            Layout unclip ratio. Can be a float, a tuple of (min, max), or a
            dict with page-specific values.
        :param layout_merge_bboxes_mode:
            Layout merge bounding boxes mode. Can be a string or a dict.
        :param prompt_label:
            Prompt type for the VLM. Possible values are "ocr", "formula",
            "table", and "chart".
        :param format_block_content:
            Whether to format block content.
        :param repetition_penalty:
            Repetition penalty parameter used in VLM sampling.
        :param temperature:
            Temperature parameter used in VLM sampling.
        :param top_p:
            Top-p parameter used in VLM sampling.
        :param min_pixels:
            Minimum number of pixels allowed during VLM preprocessing.
        :param max_pixels:
            Maximum number of pixels allowed during VLM preprocessing.
        :param prettify_markdown:
            Whether to prettify the output Markdown text.
        :param show_formula_number:
            Whether to include formula numbers in the output markdown text.
        :param visualize:
            Whether to return visualization results.
        :param additional_params:
            Additional parameters for calling the PaddleOCR API.
        """
        self.api_url = api_url
        self.access_token = access_token
        self.file_type = _normalize_file_type(file_type)
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_layout_detection = use_layout_detection
        self.use_chart_recognition = use_chart_recognition
        self.layout_threshold = layout_threshold
        self.layout_nms = layout_nms
        self.layout_unclip_ratio = layout_unclip_ratio
        self.layout_merge_bboxes_mode = layout_merge_bboxes_mode
        self.prompt_label = prompt_label
        self.format_block_content = format_block_content
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.prettify_markdown = prettify_markdown
        self.show_formula_number = show_formula_number
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
            api_url=self.api_url,
            access_token=self.access_token.to_dict(),
            file_type=self.file_type,
            use_doc_orientation_classify=self.use_doc_orientation_classify,
            use_doc_unwarping=self.use_doc_unwarping,
            use_layout_detection=self.use_layout_detection,
            use_chart_recognition=self.use_chart_recognition,
            layout_threshold=self.layout_threshold,
            layout_nms=self.layout_nms,
            layout_unclip_ratio=self.layout_unclip_ratio,
            layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
            prompt_label=self.prompt_label,
            format_block_content=self.format_block_content,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            prettify_markdown=self.prettify_markdown,
            show_formula_number=self.show_formula_number,
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

    def _parse(self, data: bytes, file_type: FileType) -> tuple[str, dict[str, Any]]:
        """
        Parse document using PaddleOCR API.

        :param data:
            Raw file data as bytes.
        :param file_type:
            File type (0 for PDF, 1 for image).
        :returns:
            A tuple containing the extracted text (separated by form feed
            characters for multiple pages) and the raw response dictionary.
        :raises requests.RequestException:
            If the API request fails.
        :raises ValueError:
            If the API response is invalid or missing required fields.
        """
        # Encode file data to base64
        encoded_data = base64.b64encode(data).decode("ascii")

        # Build request payload
        request_data = {
            "file": encoded_data,
            "fileType": file_type,
        }

        # Add optional parameters if they are set
        if self.use_doc_orientation_classify is not None:
            request_data["useDocOrientationClassify"] = self.use_doc_orientation_classify
        if self.use_doc_unwarping is not None:
            request_data["useDocUnwarping"] = self.use_doc_unwarping
        if self.use_layout_detection is not None:
            request_data["useLayoutDetection"] = self.use_layout_detection
        if self.use_chart_recognition is not None:
            request_data["useChartRecognition"] = self.use_chart_recognition
        if self.layout_threshold is not None:
            request_data["layoutThreshold"] = self.layout_threshold
        if self.layout_nms is not None:
            request_data["layoutNms"] = self.layout_nms
        if self.layout_unclip_ratio is not None:
            request_data["layoutUnclipRatio"] = self.layout_unclip_ratio
        if self.layout_merge_bboxes_mode is not None:
            request_data["layoutMergeBboxesMode"] = self.layout_merge_bboxes_mode
        if self.prompt_label is not None:
            request_data["promptLabel"] = self.prompt_label
        if self.format_block_content is not None:
            request_data["formatBlockContent"] = self.format_block_content
        if self.repetition_penalty is not None:
            request_data["repetitionPenalty"] = self.repetition_penalty
        if self.temperature is not None:
            request_data["temperature"] = self.temperature
        if self.top_p is not None:
            request_data["topP"] = self.top_p
        if self.min_pixels is not None:
            request_data["minPixels"] = self.min_pixels
        if self.max_pixels is not None:
            request_data["maxPixels"] = self.max_pixels
        if self.prettify_markdown is not None:
            request_data["prettifyMarkdown"] = self.prettify_markdown
        if self.show_formula_number is not None:
            request_data["showFormulaNumber"] = self.show_formula_number
        if self.visualize is not None:
            request_data["visualize"] = self.visualize
        if self.additional_params is not None:
            request_data.update(self.additional_params)

        # Validate input parameters
        try:
            request = PaddleOCRVLInferRequest(**request_data)
            request_dict = request.model_dump(exclude_none=True)
        except Exception as e:
            msg = f"Invalid request parameters: {e}"
            raise ValueError(msg) from e

        # Prepare headers with authentication
        access_token_value = self.access_token.resolve_value() if self.access_token else None
        headers = {"Content-Type": "application/json"}
        if access_token_value:
            headers["Authorization"] = f"token {access_token_value}"

        # Make API request
        try:
            response = requests.post(
                self.api_url,
                json=request_dict,
                headers=headers,
                timeout=300,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to call PaddleOCR API: {e}")
            raise

        # Parse and validate response
        try:
            response_data = response.json()
        except ValueError as e:
            msg = f"Invalid JSON response from API: {e}"
            raise ValueError(msg) from e

        if "result" not in response_data:
            msg = "Response missing 'result' field"
            raise ValueError(msg)

        try:
            result = PaddleOCRVLInferResult(**response_data["result"])
        except Exception as e:
            msg = f"Invalid response format: {e}"
            raise ValueError(msg) from e

        # Extract text from markdown in layout parsing results
        # Pages are separated by form feed character (\f) for compatibility
        # with Haystack's `DocumentSplitter`
        text_parts = []
        for layout_result in result.layoutParsingResults:
            if layout_result.markdown and layout_result.markdown.text:
                text_parts.append(layout_result.markdown.text)

        text = "\f".join(text_parts) if text_parts else ""

        return text, response_data

    @component.output_types(documents=list[Document], raw_paddleocr_responses=list[dict[str, Any]])
    def run(
        self,
        sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
    ) -> dict[str, Any]:
        """
        Convert image or PDF files to Documents.

        :param sources:
            List of image or PDF file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single
            dictionary. If it's a single dictionary, its content is added to
            the metadata of all produced Documents. If it's a list, the length
            of the list must match the number of sources, because the two
            lists will be zipped. If `sources` contains ByteStream objects,
            their `meta` will be added to the output Documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of created Documents.
            - `raw_paddleocr_responses`: A list of raw PaddleOCR API responses.
        """
        documents = []
        raw_responses = []

        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning(
                    f"Could not read {source}. Skipping it. Error: {e}",
                )
                continue

            # Determine file type (either from config or inferred from extension)
            if self.file_type is not None:
                file_type = self.file_type
            else:
                mime_type = bytestream.mime_type if hasattr(bytestream, "mime_type") and bytestream.mime_type else None
                file_type = _infer_file_type_from_source(source, mime_type)
            if file_type is None:
                logger.warning(
                    f"Could not determine file type for {source}. Skipping it.",
                )
                continue

            try:
                text, raw_resp = self._parse(bytestream.data, file_type)
            except Exception as e:
                logger.warning(
                    f"Could not read {source} and convert it to Document, skipping. Error: {e}",
                )
                continue

            if not text:
                msg = (
                    f"{self.__class__.__name__} could not extract text"
                    " from the file {source}. Returning an empty document."
                )
                logger.warning(msg)

            merged_metadata = {**bytestream.meta, **metadata}

            document = Document(content=text, meta=merged_metadata)
            documents.append(document)

            raw_responses.append(raw_resp)

        return {"documents": documents, "raw_paddleocr_responses": raw_responses}
