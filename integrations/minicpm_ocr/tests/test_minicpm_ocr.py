# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.components.converters.minicpm_ocr import MiniCPMOCRDocumentConverter
from haystack_integrations.components.converters.minicpm_ocr import converter as converter_module


@pytest.fixture(autouse=True)
def mock_ensure_layout_model(monkeypatch):
    # run() warms up lazily; never download the layout model in unit tests
    mock = MagicMock()
    monkeypatch.setattr(converter_module, "ensure_layout_model", mock)
    return mock


def make_converter(**kwargs) -> MiniCPMOCRDocumentConverter:
    params = {"server_url": "http://localhost:8000/v1", "model": "minicpm-ocr"}
    params.update(kwargs)
    return MiniCPMOCRDocumentConverter(**params)


class TestInit:
    def test_rejects_empty_server_url(self):
        with pytest.raises(ValueError, match="server_url"):
            MiniCPMOCRDocumentConverter(server_url="  ")

    @pytest.mark.parametrize("backend", ["tgi", "OpenAI", " vllm"])
    def test_rejects_unknown_backend(self, backend):
        with pytest.raises(ValueError, match="Unsupported backend"):
            make_converter(backend=backend)

    @pytest.mark.parametrize("workers", [{"page_workers": 0}, {"max_workers": 0}])
    def test_rejects_non_positive_workers(self, workers):
        with pytest.raises(ValueError, match="at least 1"):
            make_converter(**workers)

    @pytest.mark.parametrize("url", ["http://localhost:8000/v1/", "http://localhost:8000/"])
    def test_normalizes_server_url(self, url):
        assert make_converter(server_url=url).server_url == "http://localhost:8000"

    def test_does_not_load_layout_model(self, mock_ensure_layout_model):
        make_converter()
        mock_ensure_layout_model.assert_not_called()


class TestWarmUp:
    def test_uses_default_layout_dir(self, mock_ensure_layout_model):
        converter = make_converter()
        converter.warm_up()

        mock_ensure_layout_model.assert_called_once_with(Path.home() / ".cache" / "minicpm-ocr" / "layout")

    def test_uses_custom_layout_dir(self, mock_ensure_layout_model):
        converter = make_converter(layout_model_dir="~/models/layout")
        converter.warm_up()

        mock_ensure_layout_model.assert_called_once_with(Path("~/models/layout").expanduser())

    def test_is_idempotent(self, mock_ensure_layout_model):
        converter = make_converter()
        converter.warm_up()
        converter.warm_up()

        mock_ensure_layout_model.assert_called_once()

    def test_run_warms_up_once(self, monkeypatch, mock_ensure_layout_model):
        converter = make_converter()
        monkeypatch.setattr(converter, "_parse", lambda *_: ("text", 1))
        source = ByteStream(data=b"%PDF", mime_type="application/pdf")

        converter.run(sources=[source])
        converter.run(sources=[source])

        mock_ensure_layout_model.assert_called_once()


class TestSerialization:
    def test_to_dict(self):
        converter = make_converter(
            layout_model_dir="/models/layout",
            layout_device="cuda:0",
            request_timeout=42.0,
            page_workers=2,
            max_workers=3,
            store_full_path=True,
        )
        data = component_to_dict(converter, "converter")

        assert data == {
            "type": "haystack_integrations.components.converters.minicpm_ocr.converter.MiniCPMOCRDocumentConverter",
            "init_parameters": {
                "server_url": "http://localhost:8000",
                "model": "minicpm-ocr",
                "backend": "vllm",
                "api_key": {"type": "env_var", "env_vars": ["MINICPM_OCR_API_KEY"], "strict": False},
                "layout_model_dir": "/models/layout",
                "layout_device": "cuda:0",
                "request_timeout": 42.0,
                "page_workers": 2,
                "max_workers": 3,
                "store_full_path": True,
            },
        }

    def test_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("OTHER_MINICPM_KEY", "secret")
        converter = make_converter(
            backend="ollama",
            api_key=Secret.from_env_var("OTHER_MINICPM_KEY"),
            request_timeout=7.5,
            store_full_path=True,
        )
        restored = component_from_dict(
            MiniCPMOCRDocumentConverter, component_to_dict(converter, "converter"), name="converter"
        )

        assert restored.server_url == converter.server_url
        assert restored.model == converter.model
        assert restored.backend == "ollama"
        assert restored.request_timeout == 7.5
        assert restored.store_full_path is True
        assert restored.api_key.resolve_value() == "secret"


class TestRun:
    def test_converts_pdf_bytestream(self, monkeypatch):
        converter = make_converter()
        captured = {}

        def fake_parse(file_path: Path, _temp_dir: Path, api_key: str | None):
            captured["suffix"] = file_path.suffix
            captured["api_key"] = api_key
            return "# Invoice\fTotal: 42", 2

        monkeypatch.setattr(converter, "_parse", fake_parse)
        source = ByteStream(data=b"%PDF-1.4 fake", mime_type="application/pdf", meta={"file_path": "invoice.pdf"})

        result = converter.run(sources=[source], meta={"company": "ACME"})

        assert captured == {"suffix": ".pdf", "api_key": None}
        documents = result["documents"]
        assert len(documents) == 1
        assert documents[0].content == "# Invoice\fTotal: 42"
        assert documents[0].meta == {
            "file_path": "invoice.pdf",
            "company": "ACME",
            "page_count": 2,
            "ocr_engine": "minicpm-ocr",
        }

    def test_uses_suffix_from_path(self, monkeypatch, tmp_path):
        converter = make_converter()
        image = tmp_path / "scan.png"
        image.write_bytes(b"png")
        captured = {}

        def fake_parse(file_path: Path, _temp_dir: Path, _api_key: str | None):
            captured["suffix"] = file_path.suffix
            return "text", 1

        monkeypatch.setattr(converter, "_parse", fake_parse)
        converter.run(sources=[image])

        assert captured["suffix"] == ".png"

    @pytest.mark.parametrize("store_full_path", [True, False])
    def test_store_full_path(self, monkeypatch, tmp_path, store_full_path):
        converter = make_converter(store_full_path=store_full_path)
        monkeypatch.setattr(converter, "_parse", lambda *_: ("text", 1))
        image = tmp_path / "scan.png"
        image.write_bytes(b"png")

        result = converter.run(sources=[image])

        expected = str(image) if store_full_path else "scan.png"
        assert result["documents"][0].meta["file_path"] == expected

    def test_resolves_api_key(self, monkeypatch):
        converter = make_converter(api_key=Secret.from_token("secret"))
        captured = {}

        def fake_parse(_file_path: Path, _temp_dir: Path, api_key: str | None):
            captured["api_key"] = api_key
            return "text", 1

        monkeypatch.setattr(converter, "_parse", fake_parse)
        converter.run(sources=[ByteStream(data=b"%PDF", mime_type="application/pdf")])

        assert captured["api_key"] == "secret"

    def test_skips_unknown_file_type(self):
        converter = make_converter()
        source = ByteStream(data=b"doc", mime_type="application/msword")

        result = converter.run(sources=[source])

        assert result["documents"] == []

    def test_skips_source_that_fails_to_parse(self, monkeypatch):
        converter = make_converter()

        def fake_parse(_file_path: Path, _temp_dir: Path, _api_key: str | None):
            msg = "server unreachable"
            raise RuntimeError(msg)

        monkeypatch.setattr(converter, "_parse", fake_parse)
        source = ByteStream(data=b"%PDF", mime_type="application/pdf")

        result = converter.run(sources=[source])

        assert result["documents"] == []

    def test_returns_empty_document_when_model_returns_no_text(self, monkeypatch):
        converter = make_converter()
        monkeypatch.setattr(converter, "_parse", lambda *_: ("\f", 2))

        result = converter.run(sources=[ByteStream(data=b"%PDF", mime_type="application/pdf")])

        assert len(result["documents"]) == 1
        # page separators are kept so the page count stays consistent with the content
        assert result["documents"][0].content == "\f"

    def test_applies_per_source_metadata(self, monkeypatch):
        converter = make_converter()
        monkeypatch.setattr(converter, "_parse", lambda *_: ("text", 1))
        sources = [
            ByteStream(data=b"%PDF", mime_type="application/pdf"),
            ByteStream(data=b"\x89PNG", mime_type="image/png"),
        ]

        result = converter.run(sources=sources, meta=[{"page": 1}, {"page": 2}])

        assert [doc.meta["page"] for doc in result["documents"]] == [1, 2]


class TestParse:
    @pytest.fixture
    def sdk(self, monkeypatch):
        mocks = {
            "load_config": MagicMock(return_value={"_meta": {"run_dir": "run_dir"}}),
            "prepare_run": MagicMock(),
            "infer_run": MagicMock(),
            "load_prompt_map": MagicMock(return_value={"prompts": {}}),
            "assemble_document_markdown": MagicMock(return_value=("page 1\fpage 2", 2, 0)),
        }
        for name, mock in mocks.items():
            monkeypatch.setattr(converter_module, name, mock)
        return mocks

    def test_parse_uses_sdk_and_form_feed(self, sdk, tmp_path):
        converter = make_converter(backend="ollama", api_key=Secret.from_token("secret"), layout_device="cuda:0")
        converter.warm_up()

        markdown, page_count = converter._parse(tmp_path / "input.pdf", tmp_path, "secret")

        assert (markdown, page_count) == ("page 1\fpage 2", 2)
        overrides = sdk["load_config"].call_args.kwargs["overrides"]
        assert overrides["input"]["file_path"] == str(tmp_path / "input.pdf")
        assert overrides["run"]["run_root"] == str(tmp_path)
        assert overrides["layout"]["model_dir"] == str(Path.home() / ".cache" / "minicpm-ocr" / "layout")
        assert overrides["layout"]["device"] == "cuda:0"
        assert overrides["engine"] == {
            "type": "ollama",
            "model_name": "minicpm-ocr",
            "server_url": "http://localhost:8000",
            "api_key": "secret",
            "request_timeout_sec": 300.0,
        }
        config = sdk["load_config"].return_value
        sdk["prepare_run"].assert_called_once_with(config)
        sdk["infer_run"].assert_called_once_with(config)
        sdk["assemble_document_markdown"].assert_called_once_with(
            "run_dir", {"prompts": {}}, merge_tables=False, delimiter="\f", cfg=config
        )

    def test_parse_raises_when_no_pages(self, sdk, tmp_path):
        sdk["assemble_document_markdown"].return_value = ("", 0, 0)
        converter = make_converter()
        converter.warm_up()

        with pytest.raises(RuntimeError, match="did not produce any page"):
            converter._parse(tmp_path / "input.pdf", tmp_path, None)
