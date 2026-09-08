# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.embedders.twelvelabs import TwelveLabsDocumentMultimodalEmbedder

PATCH_TARGET = "haystack_integrations.components.embedders.twelvelabs.document_multimodal_embedder.embed_media"
ASYNC_PATCH_TARGET = (
    "haystack_integrations.components.embedders.twelvelabs.document_multimodal_embedder.embed_media_async"
)


def test_init_defaults(monkeypatch):
    monkeypatch.setenv("TWELVELABS_API_KEY", "tlk_env")
    embedder = TwelveLabsDocumentMultimodalEmbedder()
    assert embedder.model == "marengo3.0"
    assert embedder.file_path_meta_field == "file_path"
    assert embedder.root_path == ""
    assert embedder.modality_meta_field == "modality"
    assert embedder.batch_size == 32
    assert embedder.progress_bar is True


def test_to_dict_and_from_dict(monkeypatch):
    monkeypatch.setenv("TWELVELABS_API_KEY", "tlk_env")
    data = TwelveLabsDocumentMultimodalEmbedder(root_path="/data", file_path_meta_field="path", batch_size=8).to_dict()
    assert data["type"].endswith("TwelveLabsDocumentMultimodalEmbedder")
    params = data["init_parameters"]
    assert params["root_path"] == "/data"
    assert params["file_path_meta_field"] == "path"
    assert params["batch_size"] == 8
    restored = TwelveLabsDocumentMultimodalEmbedder.from_dict(data)
    assert restored.root_path == "/data"
    assert restored.file_path_meta_field == "path"
    assert restored.batch_size == 8


def test_run_embeds_documents_and_preserves_originals():
    embedder = TwelveLabsDocumentMultimodalEmbedder(api_key=Secret.from_token("tlk_test"), progress_bar=False)
    docs = [Document(meta={"file_path": "cat.jpg"}), Document(meta={"file_path": "clip.mp4"})]
    with patch(PATCH_TARGET, side_effect=[[0.1] * 4, [0.2] * 4]) as mock_embed:
        result = embedder.run(documents=docs)
    out = result["documents"]
    assert result["meta"] == {"model": "marengo3.0"}
    assert mock_embed.call_count == 2
    mock_embed.assert_any_call("cat.jpg", "image", "marengo3.0", "tlk_test")
    mock_embed.assert_any_call("clip.mp4", "video", "marengo3.0", "tlk_test")
    assert out[0].embedding == [0.1] * 4
    assert out[1].embedding == [0.2] * 4
    assert out[0].meta["embedding_source"] == {"type": "image", "file_path_meta_field": "file_path"}
    assert out[1].meta["embedding_source"]["type"] == "video"
    # Originals are not mutated.
    assert docs[0].embedding is None


def test_run_joins_relative_path_with_root_path(tmp_path):
    root = tmp_path
    (root / "sub").mkdir()
    embedder = TwelveLabsDocumentMultimodalEmbedder(
        api_key=Secret.from_token("tlk_test"), root_path=str(root), progress_bar=False
    )
    docs = [Document(meta={"file_path": "sub/cat.jpg"})]
    with patch(PATCH_TARGET, return_value=[0.1] * 4) as mock_embed:
        embedder.run(documents=docs)
    expected = str(root.resolve() / "sub" / "cat.jpg")
    mock_embed.assert_called_once_with(expected, "image", "marengo3.0", "tlk_test")


def test_run_rejects_traversal_outside_root_path():
    embedder = TwelveLabsDocumentMultimodalEmbedder(
        api_key=Secret.from_token("tlk_test"), root_path="/data", progress_bar=False
    )
    docs = [Document(meta={"file_path": "../../etc/passwd.jpg"})]
    with patch(PATCH_TARGET, return_value=[0.1] * 4), pytest.raises(ValueError):
        embedder.run(documents=docs)


def test_run_rejects_absolute_path_with_root_path():
    embedder = TwelveLabsDocumentMultimodalEmbedder(
        api_key=Secret.from_token("tlk_test"), root_path="/data", progress_bar=False
    )
    docs = [Document(meta={"file_path": "/etc/passwd.jpg"})]
    with patch(PATCH_TARGET, return_value=[0.1] * 4), pytest.raises(ValueError):
        embedder.run(documents=docs)


def test_run_rejects_non_string_modality_meta():
    embedder = TwelveLabsDocumentMultimodalEmbedder(api_key=Secret.from_token("tlk_test"), progress_bar=False)
    docs = [Document(meta={"file_path": "cat.jpg", "modality": 123})]
    with patch(PATCH_TARGET, return_value=[0.1] * 4), pytest.raises(ValueError):
        embedder.run(documents=docs)


def test_run_url_ignores_root_path():
    embedder = TwelveLabsDocumentMultimodalEmbedder(
        api_key=Secret.from_token("tlk_test"), root_path="/data", progress_bar=False
    )
    docs = [Document(meta={"file_path": "https://ex.com/v.mp4"})]
    with patch(PATCH_TARGET, return_value=[0.1] * 4) as mock_embed:
        embedder.run(documents=docs)
    mock_embed.assert_called_once_with("https://ex.com/v.mp4", "video", "marengo3.0", "tlk_test")


def test_run_modality_override_from_meta():
    embedder = TwelveLabsDocumentMultimodalEmbedder(api_key=Secret.from_token("tlk_test"), progress_bar=False)
    docs = [Document(meta={"file_path": "https://ex.com/stream", "modality": "audio"})]
    with patch(PATCH_TARGET, return_value=[0.1] * 4) as mock_embed:
        embedder.run(documents=docs)
    mock_embed.assert_called_once_with("https://ex.com/stream", "audio", "marengo3.0", "tlk_test")


def test_run_missing_media_path_raises():
    embedder = TwelveLabsDocumentMultimodalEmbedder(api_key=Secret.from_token("tlk_test"), progress_bar=False)
    with patch(PATCH_TARGET, return_value=[0.1] * 4), pytest.raises(ValueError):
        embedder.run(documents=[Document(content="no path here")])


def test_run_rejects_non_documents():
    embedder = TwelveLabsDocumentMultimodalEmbedder(api_key=Secret.from_token("tlk_test"))
    with pytest.raises(TypeError):
        embedder.run(documents=["not a document"])


@pytest.mark.asyncio
async def test_run_async_embeds_documents():
    embedder = TwelveLabsDocumentMultimodalEmbedder(api_key=Secret.from_token("tlk_test"), progress_bar=False)
    docs = [Document(meta={"file_path": "cat.jpg"}), Document(meta={"file_path": "song.mp3"})]
    with patch(ASYNC_PATCH_TARGET, new=AsyncMock(side_effect=[[0.1] * 4, [0.2] * 4])) as mock_embed:
        out = (await embedder.run_async(documents=docs))["documents"]
    assert mock_embed.await_count == 2
    assert out[0].embedding == [0.1] * 4
    assert out[1].embedding == [0.2] * 4
    assert out[1].meta["embedding_source"]["type"] == "audio"


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("TWELVELABS_API_KEY"), reason="TWELVELABS_API_KEY env var not set")
def test_live_video_document_embedding():
    sample = Path(__file__).parent / "assets" / "sample_video.mp4"
    embedder = TwelveLabsDocumentMultimodalEmbedder(progress_bar=False)
    docs = [Document(meta={"file_path": str(sample)})]
    out = embedder.run(documents=docs)["documents"]
    assert isinstance(out[0].embedding, list)
    assert len(out[0].embedding) > 0
    assert out[0].meta["embedding_source"]["type"] == "video"
