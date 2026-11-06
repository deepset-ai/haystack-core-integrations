# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, ClassVar, Literal

import numpy as np
from haystack.utils.auth import Secret
from PIL.Image import Image

from sentence_transformers import SentenceTransformer, quantize_embeddings


def _resolve_onnx_session_options(model_kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Build an `onnxruntime.SessionOptions` object from a `session_options` dict in `model_kwargs`.

    Each key of the dict is set as an attribute of the `SessionOptions`, e.g. `{"intra_op_num_threads": 1}`.

    :param model_kwargs: The model kwargs passed to the Sentence Transformers model.
    :returns: A copy of `model_kwargs` with `session_options` replaced by the built `SessionOptions` object, or
        `model_kwargs` itself when it has no `session_options` dict.
    """
    if not model_kwargs or not isinstance(model_kwargs.get("session_options"), dict):
        return model_kwargs

    # onnxruntime is only installed for the ONNX backend
    import onnxruntime as ort  # noqa: PLC0415

    session_options = ort.SessionOptions()
    for name, value in model_kwargs["session_options"].items():
        setattr(session_options, name, value)
    return {**model_kwargs, "session_options": session_options}


class _SentenceTransformersEmbeddingBackendFactory:
    """
    Factory class to create instances of Sentence Transformers embedding backends.
    """

    _instances: ClassVar[dict[str, "_SentenceTransformersEmbeddingBackend"]] = {}

    @staticmethod
    def get_embedding_backend(
        *,
        model: str,
        device: str | None = None,
        auth_token: Secret | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        truncate_dim: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> "_SentenceTransformersEmbeddingBackend":
        cache_params = {
            "model": model,
            "device": device,
            "auth_token": auth_token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
            "truncate_dim": truncate_dim,
            "model_kwargs": model_kwargs,
            "tokenizer_kwargs": tokenizer_kwargs,
            "config_kwargs": config_kwargs,
            "backend": backend,
        }

        embedding_backend_id = json.dumps(cache_params, sort_keys=True, default=str)

        if embedding_backend_id in _SentenceTransformersEmbeddingBackendFactory._instances:
            return _SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _SentenceTransformersEmbeddingBackend(
            model=model,
            device=device,
            auth_token=auth_token,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )

        _SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _SentenceTransformersEmbeddingBackend:
    """
    Class to manage Sentence Transformers embeddings.
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
        truncate_dim: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> None:
        self.model = SentenceTransformer(
            model_name_or_path=model,
            device=device,
            token=auth_token.resolve_value() if auth_token else None,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            truncate_dim=truncate_dim,
            model_kwargs=_resolve_onnx_session_options(model_kwargs=model_kwargs)
            if backend == "onnx"
            else model_kwargs,
            # `tokenizer_kwargs` was renamed to `processor_kwargs` in sentence-transformers 5.4.0
            processor_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )

    def embed(self, data: list[str] | list[Image], **kwargs: Any) -> list[list[float]]:
        quantization_ranges = kwargs.pop("quantization_ranges", None)
        precision = kwargs.get("precision", "float32")
        if quantization_ranges is not None and precision in ("int8", "uint8"):
            kwargs["precision"] = "float32"
            embeddings = self.model.encode(data, **kwargs)
            quantized = quantize_embeddings(embeddings, precision=precision, ranges=np.asarray(quantization_ranges))
            if isinstance(quantized, list):
                msg = "Unexpected multi-vector output from quantize_embeddings"
                raise TypeError(msg)
            return quantized.tolist()
        return self.model.encode(data, **kwargs).tolist()
