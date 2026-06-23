# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile as _tmpmod
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.dataclasses import ByteStream
from haystack.utils import Secret
from paddleocr import Model, PaddleOCRVLOptions  # type: ignore[import-untyped]
from PIL import Image

from haystack_integrations.components.converters.paddleocr import (
    PaddleOCRVLDocumentConverter,
)
from haystack_integrations.components.converters.paddleocr.paddleocr_vl_document_converter import (
    _infer_file_type_from_source,
)


def create_empty_image(tmp_path: Path, filename: str = "test.png") -> Path:
    img = Image.new("RGB", (800, 600), color="white")
    img.save(tmp_path / filename)
    return tmp_path / filename


def make_parse_result(pages_text: list[str]) -> MagicMock:
    """Build a mock DocParsingResult with given per-page markdown texts."""
    pages = []
    for text in pages_text:
        page = MagicMock()
        page.markdown_text = text
        page.raw = {"markdown": {"text": text}}
        pages.append(page)
    result = MagicMock()
    result.job_id = "job-123"
    result.pages = pages
    result.data_info = {}
    return result


CLASS_TYPE = (
    "haystack_integrations.components.converters.paddleocr.paddleocr_vl_document_converter.PaddleOCRVLDocumentConverter"
)

_PATCH_CLIENT = "haystack_integrations.components.converters.paddleocr.paddleocr_vl_document_converter.PaddleOCRClient"


@pytest.fixture
def mock_client_ctx():
    """Patch PaddleOCRClient so parse_document can be controlled per-test."""
    client_instance = MagicMock()
    client_instance.__enter__ = MagicMock(return_value=client_instance)
    client_instance.__exit__ = MagicMock(return_value=False)
    client_instance.parse_document = MagicMock(return_value=make_parse_result(["# Sample Document\n\nThis is page 1."]))
    with patch(_PATCH_CLIENT, return_value=client_instance):
        yield client_instance


class TestInit:
    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "test-token")
        converter = PaddleOCRVLDocumentConverter()

        assert converter.base_url is None
        assert converter.model == Model.PADDLE_OCR_VL_16
        assert converter.file_type is None
        assert converter.use_doc_orientation_classify is False
        assert converter.use_doc_unwarping is False
        assert converter.use_layout_detection is None
        assert converter.layout_threshold is None
        assert converter.layout_nms is None
        assert converter.layout_unclip_ratio is None
        assert converter.layout_merge_bboxes_mode is None
        assert converter.prompt_label is None
        assert converter.format_block_content is None
        assert converter.repetition_penalty is None
        assert converter.temperature is None
        assert converter.top_p is None
        assert converter.min_pixels is None
        assert converter.max_pixels is None
        assert converter.prettify_markdown is None
        assert converter.show_formula_number is None
        assert converter.visualize is None
        assert converter.additional_params is None

    def test_custom_model_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "test-token")
        converter = PaddleOCRVLDocumentConverter(model="PaddleOCR-VL-1.5")
        assert converter.model == Model.PADDLE_OCR_VL_15

    def test_invalid_file_type_string(self) -> None:
        with pytest.raises(ValueError, match="Invalid `file_type` string"):
            PaddleOCRVLDocumentConverter(
                access_token=Secret.from_token("t"),
                file_type="invalid",  # type: ignore[arg-type]
            )

    def test_invalid_file_type_value(self) -> None:
        with pytest.raises(ValueError, match="Invalid `file_type` value"):
            PaddleOCRVLDocumentConverter(
                access_token=Secret.from_token("t"),
                file_type=123,  # type: ignore[arg-type]
            )

    def test_all_optional_params(self) -> None:
        converter = PaddleOCRVLDocumentConverter(
            base_url="http://custom.example.com",
            access_token=Secret.from_token("t"),
            file_type="pdf",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_layout_detection=True,
            use_chart_recognition=True,
            use_seal_recognition=True,
            use_ocr_for_image_block=True,
            layout_threshold=0.5,
            layout_nms=True,
            layout_unclip_ratio=1.5,
            layout_merge_bboxes_mode="merge",
            layout_shape_mode="auto",
            prompt_label="ocr",
            format_block_content=True,
            repetition_penalty=1.1,
            temperature=0.7,
            top_p=0.9,
            min_pixels=100,
            max_pixels=1000,
            max_new_tokens=500,
            merge_layout_blocks=True,
            markdown_ignore_labels=["footer"],
            vlm_extra_args={"k": "v"},
            prettify_markdown=True,
            show_formula_number=True,
            restructure_pages=True,
            merge_tables=True,
            relevel_titles=True,
            visualize=True,
            additional_params={"logId": "x"},
        )
        assert converter.base_url == "http://custom.example.com"
        assert converter.file_type == 0
        assert converter.use_doc_orientation_classify is True
        assert converter.additional_params == {"logId": "x"}


class TestSerialisation:
    def test_to_dict_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "test-token")
        converter = PaddleOCRVLDocumentConverter()
        d = converter.to_dict()

        assert d["type"] == CLASS_TYPE
        p = d["init_parameters"]
        assert p["base_url"] is None
        assert p["model"] == Model.PADDLE_OCR_VL_16.value
        assert p["file_type"] is None
        assert p["access_token"] == {
            "type": "env_var",
            "env_vars": ["PADDLEOCR_ACCESS_TOKEN", "AISTUDIO_ACCESS_TOKEN"],
            "strict": True,
        }

    def test_to_dict_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "tok")
        converter = PaddleOCRVLDocumentConverter(
            base_url="http://base.example.com",
            model=Model.PADDLE_OCR_VL_15,
            file_type="image",
            temperature=0.5,
        )
        p = converter.to_dict()["init_parameters"]
        assert p["base_url"] == "http://base.example.com"
        assert p["model"] == "PaddleOCR-VL-1.5"
        assert p["file_type"] == 1
        assert p["temperature"] == 0.5

    def test_from_dict_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "test-token")
        converter = PaddleOCRVLDocumentConverter(
            base_url="http://base.example.com",
            model=Model.PADDLE_OCR_VL_16,
            temperature=0.3,
        )
        restored = PaddleOCRVLDocumentConverter.from_dict(converter.to_dict())

        assert restored.base_url == "http://base.example.com"
        assert restored.model == Model.PADDLE_OCR_VL_16
        assert restored.temperature == 0.3

    def test_from_dict_without_model_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "test-token")
        data = {
            "type": CLASS_TYPE,
            "init_parameters": {
                "access_token": {"type": "env_var", "env_vars": ["PADDLEOCR_ACCESS_TOKEN"], "strict": True},
            },
        }
        converter = PaddleOCRVLDocumentConverter.from_dict(data)
        assert converter.model == Model.PADDLE_OCR_VL_16

    def test_from_dict_with_unknown_model_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "test-token")
        data = {
            "type": CLASS_TYPE,
            "init_parameters": {
                "access_token": {"type": "env_var", "env_vars": ["PADDLEOCR_ACCESS_TOKEN"], "strict": True},
                "model": "SomeCustomModel",
                "base_url": None,
                "file_type": None,
            },
        }
        converter = PaddleOCRVLDocumentConverter.from_dict(data)
        assert converter.model == "SomeCustomModel"

    def test_from_dict_with_model_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "test-token")
        data = {
            "type": CLASS_TYPE,
            "init_parameters": {
                "access_token": {
                    "type": "env_var",
                    "env_vars": ["PADDLEOCR_ACCESS_TOKEN"],
                    "strict": True,
                },
                "model": "PaddleOCR-VL-1.5",
                "base_url": None,
                "file_type": None,
            },
        }
        converter = PaddleOCRVLDocumentConverter.from_dict(data)
        assert converter.model == Model.PADDLE_OCR_VL_15


class TestRun:
    @pytest.mark.parametrize("source_type", ["file_path_str", "path_object", "bytestream"])
    def test_single_source(
        self,
        mock_client_ctx: MagicMock,
        tmp_path: Path,
        source_type: str,
    ) -> None:
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
        test_file = create_empty_image(tmp_path, "test.png")

        if source_type == "file_path_str":
            source = str(test_file)
        elif source_type == "path_object":
            source = test_file
        else:
            source = ByteStream(data=test_file.read_bytes(), meta={"file_path": str(test_file)})

        result = converter.run(sources=[source])

        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == "# Sample Document\n\nThis is page 1."
        assert len(result["raw_paddleocr_responses"]) == 1
        mock_client_ctx.parse_document.assert_called_once()

    def test_multiple_sources(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
        f1 = create_empty_image(tmp_path, "a.png")
        f2 = create_empty_image(tmp_path, "b.png")
        f3 = create_empty_image(tmp_path, "c.png")

        result = converter.run(sources=[str(f1), str(f2), str(f3)])

        assert len(result["documents"]) == 3
        assert mock_client_ctx.parse_document.call_count == 3

    def test_multi_page_pdf(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:
        mock_client_ctx.parse_document.return_value = make_parse_result(["# Page 1", "# Page 2"])
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 test")

        result = converter.run(sources=[str(pdf_file)])

        assert result["documents"][0].content == "# Page 1\f# Page 2"

    def test_partial_failure_skips_errored_source(self, tmp_path: Path) -> None:
        f1 = create_empty_image(tmp_path, "ok.png")
        f2 = create_empty_image(tmp_path, "fail.png")
        f3 = create_empty_image(tmp_path, "ok2.png")

        ok_result = make_parse_result(["# OK"])
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.parse_document = MagicMock(side_effect=[ok_result, Exception("API error"), ok_result])

        with patch(_PATCH_CLIENT, return_value=client):
            converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
            result = converter.run(sources=[str(f1), str(f2), str(f3)])

        assert len(result["documents"]) == 2
        assert len(result["raw_paddleocr_responses"]) == 2

    def test_unknown_extension_skipped(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:
        unknown = tmp_path / "file.xyz"
        unknown.write_bytes(b"data")
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))

        result = converter.run(sources=[str(unknown)])

        assert result == {"documents": [], "raw_paddleocr_responses": []}
        mock_client_ctx.parse_document.assert_not_called()

    def test_unreadable_source_skipped(self, mock_client_ctx: MagicMock) -> None:
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
        result = converter.run(sources=["/nonexistent/file.png"])
        assert result["documents"] == []
        mock_client_ctx.parse_document.assert_not_called()

    def test_empty_text_produces_empty_document(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:
        mock_client_ctx.parse_document.return_value = make_parse_result([""])
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
        f = create_empty_image(tmp_path, "empty.png")

        result = converter.run(sources=[str(f)])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == ""

    def test_model_forwarded_to_parse_document(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:
        converter = PaddleOCRVLDocumentConverter(
            access_token=Secret.from_token("tok"),
            model=Model.PADDLE_OCR_VL_15,
        )
        f = create_empty_image(tmp_path, "test.png")
        converter.run(sources=[str(f)])

        assert mock_client_ctx.parse_document.call_args.kwargs["model"] == Model.PADDLE_OCR_VL_15

    def test_options_forwarded(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:
        converter = PaddleOCRVLDocumentConverter(
            access_token=Secret.from_token("tok"),
            temperature=0.7,
            use_doc_orientation_classify=True,
            use_doc_unwarping=None,
            layout_nms=True,
            prettify_markdown=True,
        )
        f = create_empty_image(tmp_path, "test.png")
        converter.run(sources=[str(f)])

        options: PaddleOCRVLOptions = mock_client_ctx.parse_document.call_args.kwargs["options"]
        assert options.temperature == 0.7
        assert options.use_doc_orientation_classify is True
        assert options.use_doc_unwarping is None
        assert options.layout_nms is True
        assert options.prettify_markdown is True

    def test_meta_single_dict_applied_to_all(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:  # noqa: ARG002
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
        f1 = create_empty_image(tmp_path, "a.png")
        f2 = create_empty_image(tmp_path, "b.png")

        result = converter.run(sources=[str(f1), str(f2)], meta={"dept": "eng"})

        for doc in result["documents"]:
            assert doc.meta["dept"] == "eng"

    def test_meta_list_applied_per_source(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:  # noqa: ARG002
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
        f1 = create_empty_image(tmp_path, "a.png")
        f2 = create_empty_image(tmp_path, "b.png")

        result = converter.run(
            sources=[str(f1), str(f2)],
            meta=[{"author": "Alice"}, {"author": "Bob"}],
        )

        assert result["documents"][0].meta["author"] == "Alice"
        assert result["documents"][1].meta["author"] == "Bob"

    def test_file_type_pdf_tmp_suffix(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        converter.run(sources=[str(pdf)])

        assert mock_client_ctx.parse_document.call_args.kwargs["file_path"].endswith(".pdf")

    def test_file_type_image_tmp_suffix(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
        f = create_empty_image(tmp_path, "test.png")

        converter.run(sources=[str(f)])

        assert mock_client_ctx.parse_document.call_args.kwargs["file_path"].endswith(".jpg")

    def test_file_type_manual_override(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"), file_type="pdf")
        f = create_empty_image(tmp_path, "test.png")
        converter.run(sources=[str(f)])

        assert mock_client_ctx.parse_document.call_args.kwargs["file_path"].endswith(".pdf")

    def test_bytestream_with_file_path_meta(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:  # noqa: ARG002
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
        f = create_empty_image(tmp_path, "img.jpg")
        bs = ByteStream(data=f.read_bytes(), meta={"file_path": str(f)})

        assert len(converter.run(sources=[bs])["documents"]) == 1

    def test_bytestream_without_file_path_uses_mime_type(
        self,
        mock_client_ctx: MagicMock,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
        f = create_empty_image(tmp_path, "img.png")
        bs = ByteStream(data=f.read_bytes(), mime_type="image/png")

        assert len(converter.run(sources=[bs])["documents"]) == 1

    def test_raw_response_structure(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:  # noqa: ARG002
        converter = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok"))
        f = create_empty_image(tmp_path, "test.png")

        raw = converter.run(sources=[str(f)])["raw_paddleocr_responses"][0]

        assert raw["job_id"] == "job-123"
        assert "pages" in raw
        assert "data_info" in raw

    def test_base_url_forwarded_to_client(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.parse_document = MagicMock(return_value=make_parse_result(["text"]))

        with patch(_PATCH_CLIENT, return_value=client) as mock_cls:
            PaddleOCRVLDocumentConverter(
                access_token=Secret.from_token("tok"),
                base_url="http://custom.example.com",
            ).run(sources=[str(create_empty_image(tmp_path, "test.png"))])

        assert mock_cls.call_args.kwargs["base_url"] == "http://custom.example.com"

    def test_token_passed_to_client(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.parse_document = MagicMock(return_value=make_parse_result(["text"]))

        with patch(_PATCH_CLIENT, return_value=client) as mock_cls:
            PaddleOCRVLDocumentConverter(
                access_token=Secret.from_token("my-secret-token"),
            ).run(sources=[str(create_empty_image(tmp_path, "test.png"))])

        assert mock_cls.call_args.kwargs["token"] == "my-secret-token"

    def test_empty_sources_list(self) -> None:
        result = PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok")).run(sources=[])
        assert result == {"documents": [], "raw_paddleocr_responses": []}

    def test_no_base_url_omitted_from_client_kwargs(self, tmp_path: Path) -> None:
        f = create_empty_image(tmp_path, "test.png")
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.parse_document = MagicMock(return_value=make_parse_result(["text"]))

        with patch(_PATCH_CLIENT, return_value=client) as mock_cls:
            PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok")).run(sources=[str(f)])

        assert "base_url" not in mock_cls.call_args.kwargs

    def test_additional_params_as_extra_options(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:
        converter = PaddleOCRVLDocumentConverter(
            access_token=Secret.from_token("tok"),
            additional_params={"logId": "custom"},
        )
        converter.run(sources=[str(create_empty_image(tmp_path, "test.png"))])

        options: PaddleOCRVLOptions = mock_client_ctx.parse_document.call_args.kwargs["options"]
        assert options.extra_options == {"logId": "custom"}

    def test_tmp_file_deleted_after_parse(self, mock_client_ctx: MagicMock, tmp_path: Path) -> None:  # noqa: ARG002
        created_paths: list[str] = []
        original_ntf = _tmpmod.NamedTemporaryFile

        def tracking_ntf(**kwargs: object) -> object:
            f = original_ntf(**kwargs)
            created_paths.append(str(f.name))
            return f

        with patch("tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
            PaddleOCRVLDocumentConverter(access_token=Secret.from_token("tok")).run(
                sources=[str(create_empty_image(tmp_path, "test.png"))]
            )

        for p in created_paths:
            assert not Path(p).exists(), f"Temp file {p} was not deleted"


class TestInferFileType:
    @pytest.mark.parametrize(
        "extension,expected",
        [
            (".pdf", 0),
            (".PDF", 0),
            (".jpg", 1),
            (".jpeg", 1),
            (".png", 1),
            (".bmp", 1),
            (".tiff", 1),
            (".tif", 1),
            (".webp", 1),
            (".unknown", None),
            ("", None),
        ],
    )
    def test_from_file_path(self, tmp_path: Path, extension: str, expected: "int | None") -> None:
        assert _infer_file_type_from_source(tmp_path / f"file{extension}") == expected

    @pytest.mark.parametrize(
        "mime_type,expected",
        [
            ("application/pdf", 0),
            ("APPLICATION/PDF", 0),
            ("image/png", 1),
            ("image/jpeg", 1),
            ("image/webp", 1),
            ("text/plain", None),
            (None, None),
        ],
    )
    def test_from_mime_type(self, mime_type: "str | None", expected: "int | None") -> None:
        assert _infer_file_type_from_source(ByteStream(data=b"x"), mime_type) == expected

    def test_extension_takes_priority_over_mime(self, tmp_path: Path) -> None:
        assert _infer_file_type_from_source(tmp_path / "doc.pdf", "image/png") == 0

    def test_mime_fallback_for_unknown_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "file.unknown"
        assert _infer_file_type_from_source(f, "application/pdf") == 0
        assert _infer_file_type_from_source(f, "image/jpeg") == 1

    def test_bytestream_with_file_path_meta(self, tmp_path: Path) -> None:
        bs = ByteStream(data=b"x", meta={"file_path": str(tmp_path / "doc.pdf")})
        assert _infer_file_type_from_source(bs) == 0

    def test_bytestream_without_meta_uses_mime(self) -> None:
        bs = ByteStream(data=b"x")
        assert _infer_file_type_from_source(bs, "image/png") == 1
        assert _infer_file_type_from_source(bs, None) is None


@pytest.mark.skipif(
    not os.environ.get("PADDLEOCR_VL_BASE_URL") or not os.environ.get("PADDLEOCR_ACCESS_TOKEN"),
    reason="Export PADDLEOCR_VL_BASE_URL and PADDLEOCR_ACCESS_TOKEN to run integration tests.",
)
@pytest.mark.integration
class TestIntegration:
    @pytest.fixture
    def test_files_path(self) -> Path:
        return Path(__file__).parent / "test_files"

    @pytest.mark.parametrize(
        "source_files,expected_docs",
        [
            (["sample_pdf.pdf"], 1),
            (["sample_img.jpg"], 1),
            (["sample_pdf.pdf", "sample_img.jpg"], 2),
        ],
        ids=["pdf_only", "image_only", "mixed"],
    )
    def test_run_with_files(
        self,
        test_files_path: Path,
        source_files: list[str],
        expected_docs: int,
    ) -> None:
        converter = PaddleOCRVLDocumentConverter()
        result = converter.run(sources=[test_files_path / f for f in source_files])

        assert len(result["documents"]) == expected_docs
        assert all(isinstance(d, Document) for d in result["documents"])
        assert all(len(d.content) > 0 for d in result["documents"])
        assert len(result["raw_paddleocr_responses"]) == expected_docs
