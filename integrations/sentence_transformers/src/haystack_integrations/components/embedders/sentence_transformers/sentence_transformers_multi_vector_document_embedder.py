# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from haystack import Document, component
from haystack.utils import Secret

from sentence_transformers import MultiVectorEncoder


@component
class SentenceTransformersMultiVectorDocumentEmbedder:
    """Add token vectors to documents for late-interaction scoring."""

    def __init__(
        self,
        *,
        model: str = "answerdotai/answerai-colbert-small-v1",
        device: str | None = None,
        token: Secret | None = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        trust_remote_code: bool = False,
    ) -> None:
        if batch_size <= 0:
            msg = "batch_size must be greater than zero."
            raise ValueError(msg)
        self.model = model
        self.device = device
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.trust_remote_code = trust_remote_code
        self._model: MultiVectorEncoder | None = None

    def warm_up(self) -> None:
        """Load the model once."""
        if self._model is None:
            self._model = MultiVectorEncoder(
                model_name_or_path=self.model,
                device=self.device,
                token=self.token.resolve_value() if self.token else None,
                trust_remote_code=self.trust_remote_code,
            )

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Encode documents and return copies with multi_vector_embedding populated."""
        if not documents:
            return {"documents": []}
        self.warm_up()
        texts = []
        for document in documents:
            metadata = [
                str(document.meta[field]) for field in self.meta_fields_to_embed if document.meta.get(field) is not None
            ]
            texts.append(self.prefix + self.embedding_separator.join([*metadata, document.content or ""]) + self.suffix)

        assert self._model is not None  # noqa: S101 - initialized by warm_up
        embeddings = self._model.encode_document(
            texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return {
            "documents": [
                replace(document, multi_vector_embedding=embedding.tolist())
                for document, embedding in zip(documents, embeddings, strict=True)
            ]
        }
