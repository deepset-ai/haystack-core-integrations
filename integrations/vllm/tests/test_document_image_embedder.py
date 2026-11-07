# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret
from openai import APIError
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage
from PIL import Image

from haystack_integrations.components.embedders.vllm import VLLMDocumentImageEmbedder

MODEL = "TIGER-Lab/VLM2Vec-Full"


def _fake_response(embedding: list[float], prompt_tokens: int = 3, total_tokens: int = 3) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        object="list",
        model="fake-model",
        data=[Embedding(object="embedding", index=0, embedding=embedding)],
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens),
    )


def _api_error() -> APIError:
    return APIError(message="boom", request=MagicMock(), body=None)


@pytest.fixture
def image_files(tmp_path):
    """A PNG image and a 2-page PDF, created on the fly to keep binary files out of the repo."""
    png_path = tmp_path / "red.png"
    Image.new("RGB", (8, 8), color="red").save(png_path)
    pdf_path = tmp_path / "pages.pdf"
    Image.new("RGB", (8, 8), color="blue").save(
        pdf_path, save_all=True, append_images=[Image.new("RGB", (8, 8), color="green")]
    )
    return {"png": png_path, "pdf": pdf_path}


def _image_parts(call) -> list[dict]:
    return call.kwargs["body"]["messages"][0]["content"]


class TestInitializationAndSerialization:
    def test_init_default(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL)
        assert embedder.model == MODEL
        assert embedder.api_key == Secret.from_env_var("VLLM_API_KEY", strict=False)
        assert embedder.api_base_url == "http://localhost:8000/v1"
        assert embedder.file_path_meta_field == "file_path"
        assert embedder.root_path is None
        assert embedder.image_size is None
        assert embedder.instruction is None
        assert embedder.dimensions is None
        assert embedder.batch_size == 8
        assert embedder.progress_bar is True
        assert embedder.raise_on_failure is False
        assert embedder.extra_parameters is None
        assert embedder._client is None
        assert embedder._async_client is None

    def test_to_dict_from_dict_roundtrip_with_non_default_params(self, monkeypatch):
        monkeypatch.setenv("MY_VLLM_KEY", "secret-value")
        embedder = VLLMDocumentImageEmbedder(
            model=MODEL,
            api_key=Secret.from_env_var("MY_VLLM_KEY"),
            api_base_url="http://my-vllm-server:8000/v1",
            file_path_meta_field="image_path",
            root_path="/data",
            image_size=(256, 256),
            instruction="Represent the given image.",
            dimensions=128,
            batch_size=4,
            progress_bar=False,
            timeout=10.0,
            max_retries=2,
            http_client_kwargs={"proxy": "http://localhost:8080"},
            raise_on_failure=True,
            extra_parameters={"truncate_prompt_tokens": 256},
        )
        data = component_to_dict(embedder, "embedder")
        assert data["type"] == (
            "haystack_integrations.components.embedders.vllm.document_image_embedder.VLLMDocumentImageEmbedder"
        )
        assert data["init_parameters"]["api_key"] == {"env_vars": ["MY_VLLM_KEY"], "strict": True, "type": "env_var"}

        restored = component_from_dict(VLLMDocumentImageEmbedder, data, "embedder")
        assert restored.api_key.resolve_value() == "secret-value"
        for attr in (
            "model",
            "api_base_url",
            "file_path_meta_field",
            "root_path",
            "instruction",
            "dimensions",
            "batch_size",
            "progress_bar",
            "timeout",
            "max_retries",
            "http_client_kwargs",
            "raise_on_failure",
            "extra_parameters",
        ):
            assert getattr(restored, attr) == getattr(embedder, attr)
        assert tuple(restored.image_size) == (256, 256)


class TestRun:
    def test_run_embeds_images_and_pdf_pages(self, image_files):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, progress_bar=False)
        embedder._client = MagicMock()
        embedder._client.post.side_effect = [_fake_response([0.1, 0.2]), _fake_response([0.3, 0.4])]

        documents = [
            Document(content="red square", meta={"file_path": str(image_files["png"])}),
            Document(content="pdf page", meta={"file_path": str(image_files["pdf"]), "page_number": 2}),
        ]
        result = embedder.run(documents=documents)

        assert [doc.embedding for doc in result["documents"]] == [[0.1, 0.2], [0.3, 0.4]]
        for doc in result["documents"]:
            assert doc.meta["embedding_source"] == {"type": "image", "file_path_meta_field": "file_path"}
        # input Documents are not modified
        assert documents[0].embedding is None
        assert "embedding_source" not in documents[0].meta
        assert result["meta"] == {"model": "fake-model", "usage": {"prompt_tokens": 6, "total_tokens": 6}}

        first_call, second_call = embedder._client.post.call_args_list
        assert first_call.args == ("/embeddings",)
        assert first_call.kwargs["cast_to"] is CreateEmbeddingResponse
        assert first_call.kwargs["body"]["model"] == MODEL
        assert first_call.kwargs["body"]["encoding_format"] == "float"
        assert _image_parts(first_call)[0]["image_url"]["url"].startswith("data:image/png;base64,")
        # PDF pages are rendered to JPEG
        assert _image_parts(second_call)[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        # no instruction configured, so only the image is sent
        assert len(_image_parts(first_call)) == 1

    def test_run_sends_instruction_dimensions_and_extra_parameters(self, image_files):
        embedder = VLLMDocumentImageEmbedder(
            model=MODEL,
            instruction="Represent the given image.",
            dimensions=64,
            extra_parameters={"truncate_prompt_tokens": 256},
            progress_bar=False,
        )
        embedder._client = MagicMock()
        embedder._client.post.return_value = _fake_response([0.1])

        embedder.run(documents=[Document(meta={"file_path": str(image_files["png"])})])

        call = embedder._client.post.call_args
        assert _image_parts(call)[1] == {"type": "text", "text": "Represent the given image."}
        assert call.kwargs["body"]["dimensions"] == 64
        assert call.kwargs["body"]["truncate_prompt_tokens"] == 256

    def test_run_with_root_path_and_custom_meta_field(self, image_files):
        embedder = VLLMDocumentImageEmbedder(
            model=MODEL,
            file_path_meta_field="image_path",
            root_path=str(image_files["png"].parent),
            progress_bar=False,
        )
        embedder._client = MagicMock()
        embedder._client.post.return_value = _fake_response([0.5])

        result = embedder.run(documents=[Document(meta={"image_path": image_files["png"].name})])

        assert result["documents"][0].embedding == [0.5]
        assert result["documents"][0].meta["embedding_source"]["file_path_meta_field"] == "image_path"

    def test_run_sends_one_request_per_image_across_batches(self, image_files):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, batch_size=2, progress_bar=False)
        embedder._client = MagicMock()
        embedder._client.post.return_value = _fake_response([0.1])

        documents = [Document(content=str(i), meta={"file_path": str(image_files["png"])}) for i in range(5)]
        result = embedder.run(documents=documents)

        assert embedder._client.post.call_count == 5
        assert [doc.content for doc in result["documents"]] == ["0", "1", "2", "3", "4"]

    def test_run_empty_list(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL)
        assert embedder.run(documents=[]) == {"documents": [], "meta": {}}
        assert embedder._client is None

    @pytest.mark.parametrize("wrong_input", ["not a list", ["not a document"]])
    def test_run_wrong_input_type(self, wrong_input):
        embedder = VLLMDocumentImageEmbedder(model=MODEL)
        with pytest.raises(TypeError, match="VLLMImageEmbedder"):
            embedder.run(documents=wrong_input)

    def test_run_pdf_without_page_number(self, image_files):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, progress_bar=False)
        embedder._client = MagicMock()
        with pytest.raises(ValueError, match="page_number"):
            embedder.run(documents=[Document(meta={"file_path": str(image_files["pdf"])})])

    def test_run_api_error_is_logged_and_document_returned_without_embedding(self, image_files, caplog):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, progress_bar=False)
        embedder._client = MagicMock()
        embedder._client.post.side_effect = [_api_error(), _fake_response([0.7])]

        documents = [
            Document(content="fails", meta={"file_path": str(image_files["png"])}),
            Document(content="works", meta={"file_path": str(image_files["png"])}),
        ]
        result = embedder.run(documents=documents)

        assert result["documents"][0].embedding is None
        assert "embedding_source" not in result["documents"][0].meta
        assert result["documents"][1].embedding == [0.7]
        assert "Failed embedding of document" in caplog.text

    def test_run_api_error_raises_when_raise_on_failure(self, image_files):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, raise_on_failure=True, progress_bar=False)
        embedder._client = MagicMock()
        embedder._client.post.side_effect = _api_error()

        with pytest.raises(APIError):
            embedder.run(documents=[Document(meta={"file_path": str(image_files["png"])})])


class TestRunAsync:
    @pytest.mark.asyncio
    async def test_run_async_embeds_images_and_pdf_pages(self, image_files):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, instruction="Represent the given image.", progress_bar=False)
        embedder._async_client = MagicMock()
        embedder._async_client.post = AsyncMock(side_effect=[_fake_response([0.1, 0.2]), _fake_response([0.3, 0.4])])

        documents = [
            Document(meta={"file_path": str(image_files["png"])}),
            Document(meta={"file_path": str(image_files["pdf"]), "page_number": 1}),
        ]
        result = await embedder.run_async(documents=documents)

        assert [doc.embedding for doc in result["documents"]] == [[0.1, 0.2], [0.3, 0.4]]
        assert result["documents"][0].meta["embedding_source"] == {"type": "image", "file_path_meta_field": "file_path"}
        assert result["meta"] == {"model": "fake-model", "usage": {"prompt_tokens": 6, "total_tokens": 6}}
        first_call = embedder._async_client.post.call_args_list[0]
        assert _image_parts(first_call)[1] == {"type": "text", "text": "Represent the given image."}

    @pytest.mark.asyncio
    async def test_run_async_empty_list(self):
        embedder = VLLMDocumentImageEmbedder(model=MODEL)
        assert await embedder.run_async(documents=[]) == {"documents": [], "meta": {}}

    @pytest.mark.asyncio
    async def test_run_async_api_error_is_logged_and_document_returned_without_embedding(self, image_files, caplog):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, progress_bar=False)
        embedder._async_client = MagicMock()
        embedder._async_client.post = AsyncMock(side_effect=[_api_error(), _fake_response([0.7])])

        documents = [
            Document(content="fails", meta={"file_path": str(image_files["png"])}),
            Document(content="works", meta={"file_path": str(image_files["png"])}),
        ]
        result = await embedder.run_async(documents=documents)

        assert result["documents"][0].embedding is None
        assert result["documents"][1].embedding == [0.7]
        assert "Failed embedding of document" in caplog.text

    @pytest.mark.asyncio
    async def test_run_async_api_error_raises_when_raise_on_failure(self, image_files):
        embedder = VLLMDocumentImageEmbedder(model=MODEL, raise_on_failure=True, progress_bar=False)
        embedder._async_client = MagicMock()
        embedder._async_client.post = AsyncMock(side_effect=_api_error())

        with pytest.raises(APIError):
            await embedder.run_async(documents=[Document(meta={"file_path": str(image_files["png"])})])
