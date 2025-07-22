# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Tuple

import requests
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from tqdm import tqdm

JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"


@component
class JinaDocumentEmbedder:
    """
    A component for computing Document embeddings using Jina AI models.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.jina import JinaDocumentEmbedder

    # Make sure that the environment variable JINA_API_KEY is set

    document_embedder = JinaDocumentEmbedder(task="retrieval.query")

    doc = Document(content="I love pizza!")

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("JINA_API_KEY"),  # noqa: B008
        model: str = "jina-embeddings-v3",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        task: Optional[str] = None,
        dimensions: Optional[int] = None,
        late_chunking: Optional[bool] = None,
    ):
        """
        Create a JinaDocumentEmbedder component.

        :param api_key: The Jina API key.
        :param model: The name of the Jina model to use.
            Check the list of available models on [Jina documentation](https://jina.ai/embeddings/).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param batch_size: Number of Documents to encode at once.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
                             to keep the logs clean.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        :param task: The downstream task for which the embeddings will be used.
            The model will return the optimized embeddings for that task.
            Check the list of available tasks on [Jina documentation](https://jina.ai/embeddings/).
        :param dimensions: Number of desired dimension.
            Smaller dimensions are easier to store and retrieve, with minimal performance impact thanks to MRL.
        :param late_chunking: A boolean to enable or disable late chunking.
            Apply the late chunking technique to leverage the model's long-context capabilities for
            generating contextual chunk embeddings.

            The support of `task` and `late_chunking` parameters is only available for jina-embeddings-v3.
        """
        resolved_api_key = api_key.resolve_value()

        self.api_key = api_key
        self.model_name = model
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {resolved_api_key}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )
        self.task = task
        self.dimensions = dimensions
        self.late_chunking = late_chunking

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        :returns:
            Dictionary with serialized data.
        """
        kwargs = {
            "api_key": self.api_key.to_dict(),
            "model": self.model_name,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "batch_size": self.batch_size,
            "progress_bar": self.progress_bar,
            "meta_fields_to_embed": self.meta_fields_to_embed,
            "embedding_separator": self.embedding_separator,
        }
        # Optional parameters, the following two are only supported by embeddings-v3 for now
        if self.task is not None:
            kwargs["task"] = self.task
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        if self.late_chunking is not None:
            kwargs["late_chunking"] = self.late_chunking

        return default_to_dict(self, **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JinaDocumentEmbedder":
        """
        Deserializes the component from a dictionary.
        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            text_to_embed = (
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            )

            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _embed_batch(
        self, texts_to_embed: List[str], batch_size: int, parameters: Optional[Dict] = None
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """

        all_embeddings = []
        metadata = {}
        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]
            response = self._session.post(
                JINA_API_URL,
                json={"input": batch, "model": self.model_name, **(parameters or {})},
            ).json()
            if "data" not in response:
                raise RuntimeError(response["detail"])

            # Sort resulting embeddings by index
            sorted_embeddings = sorted(response["data"], key=lambda e: e["index"])
            embeddings = [result["embedding"] for result in sorted_embeddings]
            all_embeddings.extend(embeddings)
            if "model" not in metadata:
                metadata["model"] = response["model"]
            if "usage" not in metadata:
                metadata["usage"] = dict(response["usage"].items())
            else:
                metadata["usage"]["prompt_tokens"] += response["usage"]["prompt_tokens"]
                metadata["usage"]["total_tokens"] += response["usage"]["total_tokens"]

        return all_embeddings, metadata

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Compute the embeddings for a list of Documents.

        :param documents: A list of Documents to embed.
        :returns: A dictionary with following keys:
            - `documents`: List of Documents, each with an `embedding` field containing the computed embedding.
            - `meta`: A dictionary with metadata including the model name and usage statistics.
        :raises TypeError: If the input is not a list of Documents.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "JinaDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the JinaTextEmbedder."
            )
            raise TypeError(msg)

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)
        parameters: Dict[str, Any] = {}
        if self.task is not None:
            parameters["task"] = self.task
        if self.dimensions is not None:
            parameters["dimensions"] = self.dimensions
        if self.late_chunking is not None:
            parameters["late_chunking"] = self.late_chunking
        embeddings, metadata = self._embed_batch(
            texts_to_embed=texts_to_embed, batch_size=self.batch_size, parameters=parameters
        )

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": metadata}
