# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import ComponentDevice, Secret
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs

from haystack_integrations.components.embedders.sentence_transformers.embedding_backend.multivector_backend import (
    _SentenceTransformersMultivectorEmbeddingBackendFactory,
    _SentenceTransformersMultivectorEncoderEmbeddingBackend,
)


@component
class SentenceTransformersMultivectorTextEmbedder:
    """
    Embeds a text query into multiple vectors using Sentence Transformers.

    Multivector embeddings represent a query at the token level, enabling fine-grained comparison with document content
    through late-interaction scoring.

    The query embedding can be passed to a multivector retriever or to `MaxSimRanker` to rank documents with precomputed
    multivector embeddings.

    ### Usage example

    ```python
    from haystack_integrations.components.embedders.sentence_transformers import (
        SentenceTransformersMultivectorTextEmbedder,
    )

    embedder = SentenceTransformersMultivectorTextEmbedder()
    result = embedder.run(text="What is the capital of France?")
    print(result["multivector_embedding"])
    # [[-0.0761594, 0.0336303, -0.0392666, ...], ...]
    ```
    """

    def __init__(
        self,
        *,
        model: str = "answerdotai/answerai-colbert-small-v1",
        device: ComponentDevice | None = None,
        token: Secret | None = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        encode_kwargs: dict[str, Any] | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        revision: str | None = None,
    ) -> None:
        """
        Create a SentenceTransformersMultivectorTextEmbedder component.

        :param model:
            The model to use for calculating multi-vector embeddings.
            Specify the path to a local model or the ID of the model on Hugging Face.
        :param device:
            Device used to load the model. With `None`, Haystack selects an available device.
        :param token:
            Token for private Hugging Face models. Defaults to `HF_API_TOKEN` or `HF_TOKEN`, if set.
            With `None`, authentication follows the Hugging Face Hub defaults.
        :param prefix:
            A string to add at the beginning of each text to embed.
            Sentence Transformers already adds the model's query prompt (for example, the `[Q]` marker of
            ColBERT models) before this prefix, so don't repeat it here.
        :param suffix:
            A string to add at the end of each text to embed.
        :param trust_remote_code:
            If `False`, permits only Hugging Face verified model architectures.
            If `True`, permits custom models and scripts.
        :param local_files_only:
            If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
        :param model_kwargs:
            Additional keyword arguments for the underlying model's `from_pretrained`
            when loading the model. Refer to specific model documentation for available kwargs.
            With `backend="onnx"`, a `session_options` dict is turned into an `onnxruntime.SessionOptions` with
            those attributes set, for example `{"session_options": {"intra_op_num_threads": 1}}` to limit
            the threads each inference call uses.
        :param tokenizer_kwargs:
            Additional keyword arguments forwarded as `processor_kwargs` to the Sentence Transformers model
            when loading it. Refer to the
            [Sentence Transformers documentation](https://sbert.net/docs/package_reference/multi_vector_encoder/model.html)
            for available kwargs.
        :param config_kwargs:
            Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
        :param encode_kwargs:
            Additional keyword arguments for `MultiVectorEncoder.encode_query` when embedding texts.
            This parameter is provided for fine customization. Be careful not to clash with already set parameters and
            avoid passing parameters that change the output type.
        :param backend:
            The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
            Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
            for more information on acceleration and quantization options.
        :param revision:
            The specific model version to use. It can be a branch name, a tag name, or a commit id,
            for a stored model on Hugging Face.
        """

        self.model = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.local_files_only = local_files_only
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.config_kwargs = config_kwargs
        self.encode_kwargs = encode_kwargs
        self.embedding_backend: _SentenceTransformersMultivectorEncoderEmbeddingBackend | None = None
        self.backend = backend

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            model=self.model,
            device=self.device,
            token=self.token,
            prefix=self.prefix,
            suffix=self.suffix,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            local_files_only=self.local_files_only,
            model_kwargs=self.model_kwargs.copy() if self.model_kwargs is not None else None,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs,
            encode_kwargs=self.encode_kwargs,
            backend=self.backend,
        )
        if serialization_dict["init_parameters"].get("model_kwargs") is not None:
            serialize_hf_model_kwargs(serialization_dict["init_parameters"]["model_kwargs"])
        return serialization_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SentenceTransformersMultivectorTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data["init_parameters"]
        if init_params.get("model_kwargs") is not None:
            deserialize_hf_model_kwargs(init_params["model_kwargs"])
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """
        Initializes the component.
        """
        if self.embedding_backend is None:
            self.embedding_backend = _SentenceTransformersMultivectorEmbeddingBackendFactory.get_embedding_backend(
                model=self.model,
                device=self.device.to_torch_str(),
                auth_token=self.token,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
                local_files_only=self.local_files_only,
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.tokenizer_kwargs,
                config_kwargs=self.config_kwargs,
                backend=self.backend,
            )

    @component.output_types(multivector_embedding=list[list[float]])
    def run(self, text: str) -> dict[str, list[list[float]]]:
        """
        Embed a single string.

        :param text:
            Text to embed.

        :returns:
            A dictionary with the following keys:
            - `multivector_embedding`: Token vectors as a `list[list[float]]`.
        :raises TypeError:
            The input is not a string.
        """
        if not isinstance(text, str):
            msg = (
                "SentenceTransformersMultivectorTextEmbedder expects a string as input."
                " In case you want to embed a list of Documents, please use the"
                " SentenceTransformersMultivectorDocumentEmbedder."
            )
            raise TypeError(msg)
        if self.embedding_backend is None:
            self.warm_up()

        text_to_embed = self.prefix + text + self.suffix

        # mypy doesn't know this is set in warm_up
        multivector_embedding = self.embedding_backend.embed(  # type: ignore[union-attr]
            data=[text_to_embed], is_query=True, **(self.encode_kwargs if self.encode_kwargs else {})
        )[0]

        return {"multivector_embedding": multivector_embedding}
