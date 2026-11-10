# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any, Literal

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import ComponentDevice, Secret
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs

from haystack_integrations.components.embedders.sentence_transformers.embedding_backend.multivector_backend import (
    _SentenceTransformersMultivectorEmbeddingBackendFactory,
    _SentenceTransformersMultivectorEncoderEmbeddingBackend,
)


@component
class SentenceTransformersMultivectorDocumentEmbedder:
    """
    Embeds documents into multiple vectors using Sentence Transformers.

    Multivector embeddings represent document content at the token level, enabling fine-grained comparison with a query
    through late-interaction scoring. You can include document metadata in the text to embed.

    Use this component in indexing pipelines to populate each document's `multivector_embedding` field before writing it
    to a document store that supports these embeddings. The embeddings can support multivector retrieval or reranking
    with `MaxSimRanker` after an initial retrieval step.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.sentence_transformers import (
        SentenceTransformersMultivectorDocumentEmbedder,
    )

    embedder = SentenceTransformersMultivectorDocumentEmbedder()
    result = embedder.run(documents=[Document(content="Paris is the capital of France.")])
    print(result["documents"][0].multivector_embedding)
    # [[-0.0796336, 0.0318462, -0.0450180, ...], ...]
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
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        revision: str | None = None,
    ) -> None:
        """
        Creates a SentenceTransformersMultivectorDocumentEmbedder component.

        :param model:
            The model to use for calculating multi-vector embeddings.
            Pass a local path or ID of the model on Hugging Face.
        :param device:
            Device used to load the model. With `None`, Haystack selects an available device.
        :param token:
            Token for private Hugging Face models. Defaults to `HF_API_TOKEN` or `HF_TOKEN`, if set.
            With `None`, authentication follows the Hugging Face Hub defaults.
        :param prefix:
            A string to add at the beginning of each document text.
        :param suffix:
            A string to add at the end of each document text.
        :param batch_size:
            Positive number of documents to embed at once.
        :param progress_bar:
            If `True`, shows a progress bar when embedding documents.
        :param meta_fields_to_embed:
            List of metadata fields to embed along with the document text.
        :param embedding_separator:
            Separator used to concatenate the metadata fields to the document text.
        :param trust_remote_code:
            If `False`, allows only Hugging Face verified model architectures.
            If `True`, allows custom models and scripts.
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
        :param backend:
            The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
            Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
            for more information on acceleration and quantization options.
        :param revision:
            The specific model version to use. It can be a branch name, a tag name, or a commit id,
            for a stored model on Hugging Face.
        :raises ValueError:
            `batch_size` is not positive.
        """

        if batch_size <= 0:
            msg = "batch_size must be greater than zero."
            raise ValueError(msg)

        self.model = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.local_files_only = local_files_only
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.config_kwargs = config_kwargs
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
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            local_files_only=self.local_files_only,
            model_kwargs=self.model_kwargs.copy() if self.model_kwargs is not None else None,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs,
            backend=self.backend,
        )
        if serialization_dict["init_parameters"].get("model_kwargs") is not None:
            serialize_hf_model_kwargs(serialization_dict["init_parameters"]["model_kwargs"])
        return serialization_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SentenceTransformersMultivectorDocumentEmbedder":
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

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Embed a list of documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: A `list[Document]` of copies with `multivector_embedding` populated.
        :raises TypeError:
            The input is not a list of Documents.
        """
        if not isinstance(documents, list) or any(not isinstance(doc, Document) for doc in documents):
            msg = (
                "SentenceTransformersMultivectorDocumentEmbedder expects a list of Documents as input."
                " To embed a string, use SentenceTransformersMultivectorTextEmbedder."
            )
            raise TypeError(msg)
        if not documents:
            return {"documents": []}
        if self.embedding_backend is None:
            self.warm_up()

        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            text_to_embed = (
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            )
            texts_to_embed.append(text_to_embed)

        # mypy doesn't know this is set in warm_up
        embeddings = self.embedding_backend.embed(  # type: ignore[union-attr]
            data=texts_to_embed, is_query=False, batch_size=self.batch_size, show_progress_bar=self.progress_bar
        )

        documents_with_embeddings = []
        for doc, emb in zip(documents, embeddings, strict=True):
            documents_with_embeddings.append(replace(doc, multivector_embedding=emb))

        return {"documents": documents_with_embeddings}
