# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import component
from haystack.utils import Secret

from sentence_transformers import MultiVectorEncoder


@component
class SentenceTransformersMultiVectorTextEmbedder:
    """Embed a text query into token vectors for late-interaction scoring."""

    def __init__(
        self,
        *,
        model: str = "answerdotai/answerai-colbert-small-v1",
        device: str | None = None,
        token: Secret | None = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        trust_remote_code: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
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

    @component.output_types(multi_vector_embedding=list[list[float]])
    def run(self, text: str) -> dict[str, list[list[float]]]:
        """Encode a query using the model's query-specific preprocessing."""
        if not isinstance(text, str):
            msg = "text must be a string."
            raise TypeError(msg)
        self.warm_up()
        assert self._model is not None  # noqa: S101 - initialized by warm_up
        embeddings = self._model.encode_query(
            [self.prefix + text + self.suffix], convert_to_numpy=True, show_progress_bar=False
        )
        return {"multi_vector_embedding": embeddings[0].tolist()}
