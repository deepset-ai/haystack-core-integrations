# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, ClassVar, Literal

from haystack.utils.auth import Secret

from sentence_transformers import MultiVectorEncoder


class _SentenceTransformersMultivectorEmbeddingBackendFactory:
    """
    Factory class to create instances of Sentence Transformers embedding backends.
    """

    _instances: ClassVar[dict[str, "_SentenceTransformersMultivectorEncoderEmbeddingBackend"]] = {}

    @staticmethod
    def get_embedding_backend(
        *,
        model: str,
        device: str | None = None,
        auth_token: Secret | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> "_SentenceTransformersMultivectorEncoderEmbeddingBackend":
        cache_params = {
            "model": model,
            "device": device,
            "auth_token": auth_token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
            "model_kwargs": model_kwargs,
            "tokenizer_kwargs": tokenizer_kwargs,
            "config_kwargs": config_kwargs,
            "backend": backend,
        }

        embedding_backend_id = json.dumps(cache_params, sort_keys=True, default=str)

        if embedding_backend_id in _SentenceTransformersMultivectorEmbeddingBackendFactory._instances:
            return _SentenceTransformersMultivectorEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _SentenceTransformersMultivectorEncoderEmbeddingBackend(
            model=model,
            device=device,
            auth_token=auth_token,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )

        _SentenceTransformersMultivectorEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _SentenceTransformersMultivectorEncoderEmbeddingBackend:
    """
    Class to manage multi-vector embeddings from Sentence Transformers.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str | None = None,
        auth_token: Secret | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> None:
        self.model = MultiVectorEncoder(
            model_name_or_path=model,
            device=device,
            token=auth_token.resolve_value() if auth_token else None,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            model_kwargs=model_kwargs,
            # `tokenizer_kwargs` was renamed to `processor_kwargs` in sentence-transformers 5.4.0
            processor_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )

    def embed(
        self,
        *,
        data: list[str],
        is_query: bool,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[list[list[float]]]:
        """
        Encode texts using the model's query or document settings.
        """
        encode = self.model.encode_query if is_query else self.model.encode_document
        embeddings = encode(data, batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_numpy=True)
        # List input and convert_to_numpy=True guarantee a list of arrays; the SDK's union also includes dicts.
        return [embedding.tolist() for embedding in embeddings]  # type: ignore[union-attr]
