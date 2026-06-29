# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import MagicMock, patch

import pytest
from haystack.utils import Secret

from haystack_integrations.components.converters.twelvelabs import TwelveLabsVideoConverter

_MODULE = "haystack_integrations.components.converters.twelvelabs.video_converter"
ANALYZE = f"{_MODULE}.TwelveLabsVideoConverter._analyze_source"
CLIENT = f"{_MODULE}.TwelveLabs"
_SAMPLE_VIDEO_URL = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"


def test_to_dict_and_from_dict(monkeypatch):
    monkeypatch.setenv("TWELVELABS_API_KEY", "tlk_env")
    data = TwelveLabsVideoConverter(model="pegasus1.5").to_dict()
    assert data["type"].endswith("TwelveLabsVideoConverter")
    assert data["init_parameters"]["model"] == "pegasus1.5"
    restored = TwelveLabsVideoConverter.from_dict(data)
    assert restored.model == "pegasus1.5"
    assert restored.prompt


def test_run_builds_documents():
    converter = TwelveLabsVideoConverter(api_key=Secret.from_token("tlk_test"))
    with patch(ANALYZE, return_value=("Analysis text at [0:05].", "analysis_1", "asset_1")):
        result = converter.run(
            sources=["https://example.com/clip.mp4"],
            meta={"campaign": "demo"},
        )
    docs = result["documents"]
    assert len(docs) == 1
    assert docs[0].content == "Analysis text at [0:05]."
    assert docs[0].meta["asset_id"] == "asset_1"
    assert docs[0].meta["analysis_id"] == "analysis_1"
    assert docs[0].meta["provider"] == "twelvelabs"
    assert docs[0].meta["campaign"] == "demo"


def test_run_skips_failing_sources():
    converter = TwelveLabsVideoConverter(api_key=Secret.from_token("tlk_test"))

    def fake(source, *_args):
        if "bad" in source:
            error_msg = "boom"
            raise RuntimeError(error_msg)
        return ("ok", "t", "a")

    with patch(ANALYZE, side_effect=fake):
        result = converter.run(sources=["https://x/bad.mp4", "https://x/good.mp4"])
    # The failing source is skipped, the good one is kept.
    assert len(result["documents"]) == 1
    assert result["documents"][0].content == "ok"


def test_analyze_source_with_url_passes_video_context_url():
    converter = TwelveLabsVideoConverter(api_key=Secret.from_token("tlk_test"))
    with patch(CLIENT) as mock_client_cls:
        client = mock_client_cls.return_value
        client.analyze.return_value = MagicMock(data="  A description.  ", id="analysis_42")
        text, analysis_id, asset_id = converter._analyze_source("https://example.com/clip.mp4", "tlk_test")
    assert text == "A description."
    assert analysis_id == "analysis_42"
    assert asset_id is None
    # URLs are passed straight through without uploading an asset.
    client.assets.create.assert_not_called()
    assert client.analyze.call_args.kwargs["video"].url == "https://example.com/clip.mp4"


def test_analyze_source_uploads_local_file_then_analyzes(tmp_path):
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"fake video bytes")
    converter = TwelveLabsVideoConverter(api_key=Secret.from_token("tlk_test"))
    with patch(CLIENT) as mock_client_cls:
        client = mock_client_cls.return_value
        client.assets.create.return_value = MagicMock(id="asset_99")
        client.assets.retrieve.return_value = MagicMock(status="ready")
        client.analyze.return_value = MagicMock(data="Local analysis.", id="analysis_7")
        text, analysis_id, asset_id = converter._analyze_source(str(video_file), "tlk_test")
    assert text == "Local analysis."
    assert analysis_id == "analysis_7"
    assert asset_id == "asset_99"
    assert client.analyze.call_args.kwargs["video"].asset_id == "asset_99"


@pytest.mark.parametrize("data", ["", "   ", None])
def test_analyze_source_raises_when_pegasus_returns_no_text(data):
    converter = TwelveLabsVideoConverter(api_key=Secret.from_token("tlk_test"))
    with patch(CLIENT) as mock_client_cls:
        client = mock_client_cls.return_value
        client.analyze.return_value = MagicMock(data=data, id="analysis_1")
        with pytest.raises(RuntimeError, match="produced no text"):
            converter._analyze_source("https://example.com/clip.mp4", "tlk_test")


def test_upload_asset_rejects_missing_and_oversized_files(tmp_path, monkeypatch):
    converter = TwelveLabsVideoConverter(api_key=Secret.from_token("tlk_test"))
    client = MagicMock()

    with pytest.raises(FileNotFoundError, match="not found"):
        converter._upload_asset(client, str(tmp_path / "does_not_exist.mp4"))

    oversized = tmp_path / "huge.mp4"
    oversized.write_bytes(b"x")
    monkeypatch.setattr(f"{_MODULE}._MAX_DIRECT_UPLOAD_BYTES", 0)
    with pytest.raises(ValueError, match="200 MB"):
        converter._upload_asset(client, str(oversized))
    client.assets.create.assert_not_called()


@pytest.mark.skipif(not os.environ.get("TWELVELABS_API_KEY"), reason="TWELVELABS_API_KEY env var not set")
@pytest.mark.integration
def test_run_integration():
    converter = TwelveLabsVideoConverter()
    result = converter.run(sources=[_SAMPLE_VIDEO_URL])
    docs = result["documents"]
    assert len(docs) == 1
    assert isinstance(docs[0].content, str)
    assert len(docs[0].content) > 0
    assert docs[0].meta["provider"] == "twelvelabs"
    assert docs[0].meta["model"] == "pegasus1.5"
