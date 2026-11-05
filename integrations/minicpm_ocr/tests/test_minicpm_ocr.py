# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.components.converters.minicpm_ocr import MiniCPMOCRDocumentConverter


def make_converter(**kwargs) -> MiniCPMOCRDocumentConverter:
    params = {"server_url": "http://localhost:8000/v1", "model": "minicpm-ocr"}
    params.update(kwargs)
    return MiniCPMOCRDocumentConverter(**params)


class TestInit:
    def test_rejects_empty_server_url(self):
        with pytest.raises(ValueError, match="server_url"):
            MiniCPMOCRDocumentConverter(server_url="  ")

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="Unsupported backend"):
            make_converter(backend="tgi")

    def test_rejects_non_positive_workers(self):
        with pytest.raises(ValueError, match="at least 1"):
            make_converter(page_workers=0)
        with pytest.raises(ValueError, match="at least 1"):
            make_converter(max_workers=0)

    def test_normalizes_backend(self):
        assert make_converter(backend=" Openai ").backend == "openai"

    def test_normalizes_server_url(self):
        assert make_converter(server_url="http://localhost:8000/v1/").server_url == "http://localhost:8000"
        assert make_converter(server_url="http://localhost:8000/").server_url == "http://localhost:8000"


class TestSerialization:
    def test_to_dict(self):
        converter = make_converter(
            layout_model_dir="/models/layout",
            layout_device="cuda:0",
            request_timeout=42.0,
            page_workers=2,
            max_workers=3,
        )
        data = converter.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.converters.minicpm_ocr.converter.MiniCPMOCRDocumentConverter"
        )
        params = data["init_parameters"]
        assert params["server_url"] == "http://localhost:8000"
        assert params["model"] == "minicpm-ocr"
        assert params["backend"] == "vllm"
        assert params["api_key"] == {"type": "env_var", "env_vars": ["MINICPM_OCR_API_KEY"], "strict": False}
        assert params["layout_model_dir"] == "/models/layout"
        assert params["layout_device"] == "cuda:0"
        assert params["request_timeout"] == 42.0
        assert params["page_workers"] == 2
        assert params["max_workers"] == 3

    def test_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("OTHER_MINICPM_KEY", "secret")
        converter = make_converter(api_key=Secret.from_env_var("OTHER_MINICPM_KEY"), request_timeout=7.5)
        restored = MiniCPMOCRDocumentConverter.from_dict(converter.to_dict())

        assert restored.server_url == converter.server_url
        assert restored.model == converter.model
        assert restored.backend == converter.backend
        assert restored.request_timeout == 7.5
        assert restored.api_key.resolve_value() == "secret"


class TestRun:
    def test_converts_pdf_bytestream(self, monkeypatch):
        converter = make_converter()
        captured = {}

        def fake_parse(file_path: Path, _temp_dir: Path, api_key: str | None):
            captured["suffix"] = file_path.suffix
            captured["api_key"] = api_key
            return "# Invoice\n\nTotal: 42", 3

        monkeypatch.setattr(converter, "_parse", fake_parse)
        source = ByteStream(data=b"%PDF-1.4 fake", mime_type="application/pdf", meta={"file_path": "invoice.pdf"})

        result = converter.run(sources=[source], meta={"company": "ACME"})

        assert captured == {"suffix": ".pdf", "api_key": None}
        documents = result["documents"]
        assert len(documents) == 1
        assert documents[0].content == "# Invoice\n\nTotal: 42"
        assert documents[0].meta["page_count"] == 3
        assert documents[0].meta["ocr_engine"] == "minicpm-ocr"
        assert documents[0].meta["company"] == "ACME"
        assert documents[0].meta["file_path"] == "invoice.pdf"

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
        monkeypatch.setattr(converter, "_parse", lambda *_: ("", 1))

        result = converter.run(sources=[ByteStream(data=b"%PDF", mime_type="application/pdf")])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == ""

    def test_applies_per_source_metadata(self, monkeypatch):
        converter = make_converter()
        monkeypatch.setattr(converter, "_parse", lambda *_: ("text", 1))
        sources = [
            ByteStream(data=b"%PDF", mime_type="application/pdf"),
            ByteStream(data=b"\x89PNG", mime_type="image/png"),
        ]

        result = converter.run(sources=sources, meta=[{"page": 1}, {"page": 2}])

        assert [doc.meta["page"] for doc in result["documents"]] == [1, 2]
