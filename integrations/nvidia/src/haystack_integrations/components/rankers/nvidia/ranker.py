# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.rankers.nvidia.truncate import RankerTruncateMode
from haystack_integrations.utils.nvidia import DEFAULT_API_URL, NimBackend, is_hosted, url_validation

logger = logging.getLogger(__name__)


@component
class NvidiaRanker:
    """
    A component for ranking documents using ranking models provided by
    [NVIDIA NIMs](https://ai.nvidia.com).

    Usage example:
    ```python
    from haystack_integrations.components.rankers.nvidia import NvidiaRanker
    from haystack import Document
    from haystack.utils import Secret

    ranker = NvidiaRanker(
        model="nvidia/nv-rerankqa-mistral-4b-v3",
        api_key=Secret.from_env_var("NVIDIA_API_KEY"),
    )
    ranker.warm_up()

    query = "What is the capital of Germany?"
    documents = [
        Document(content="Berlin is the capital of Germany."),
        Document(content="The capital of Germany is Berlin."),
        Document(content="Germany's capital is Berlin."),
    ]

    result = ranker.run(query, documents, top_k=2)
    print(result["documents"])
    ```
    """

    def __init__(
        self,
        model: Optional[str] = None,
        truncate: Optional[Union[RankerTruncateMode, str]] = None,
        api_url: str = os.getenv("NVIDIA_API_URL", DEFAULT_API_URL),
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        top_k: int = 5,
        query_prefix: str = "",
        document_prefix: str = "",
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        timeout: Optional[float] = None,
    ):
        """
        Create a NvidiaRanker component.

        :param model:
            Ranking model to use.
        :param truncate:
            Truncation strategy to use. Can be "NONE", "END", or RankerTruncateMode. Defaults to NIM's default.
        :param api_key:
            API key for the NVIDIA NIM.
        :param api_url:
            Custom API URL for the NVIDIA NIM.
        :param top_k:
            Number of documents to return.
        :param query_prefix:
            A string to add at the beginning of the query text before ranking.
            Use it to prepend the text with an instruction, as required by reranking models like `bge`.
        :param document_prefix:
            A string to add at the beginning of each document before ranking. You can use it to prepend the document
            with an instruction, as required by embedding models like `bge`.
        :param meta_fields_to_embed:
            List of metadata fields to embed with the document.
        :param embedding_separator:
            Separator to concatenate metadata fields to the document.
        :param timeout:
            Timeout for request calls, if not set it is inferred from the `NVIDIA_TIMEOUT` environment variable
            or set to 60 by default.
        """
        if model is not None and not isinstance(model, str):
            msg = "Ranker expects the `model` parameter to be a string."
            raise TypeError(msg)
        if not isinstance(api_url, (str, type(None))):
            msg = "Ranker expects the `api_url` parameter to be a string."
            raise TypeError(msg)
        if truncate is not None and not isinstance(truncate, RankerTruncateMode):
            truncate = RankerTruncateMode.from_str(truncate)
        if not isinstance(top_k, int):
            msg = "Ranker expects the `top_k` parameter to be an integer."
            raise TypeError(msg)

        # todo: detect default in non-hosted case (when api_url is provided)
        self.truncate = truncate
        self.api_key = api_key
        self.model = model
        # if no api_url is provided, we're using a hosted model and can
        #  - assume the default url will work, because there's only one model
        #  - assume we won't call backend.models()
        self.api_url = url_validation(api_url)
        self.top_k = top_k
        self._initialized = False
        self.backend: Optional[Any] = None
        self.is_hosted = is_hosted(api_url)

        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        if timeout is None:
            timeout = float(os.environ.get("NVIDIA_TIMEOUT", "60.0"))
        self.timeout = timeout

    @classmethod
    def class_name(cls) -> str:
        return "NvidiaRanker"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the ranker to a dictionary.

        :returns: A dictionary containing the ranker's attributes.
        """
        return default_to_dict(
            self,
            model=self.model,
            top_k=self.top_k,
            truncate=self.truncate,
            api_url=self.api_url,
            api_key=self.api_key.to_dict() if self.api_key else None,
            query_prefix=self.query_prefix,
            document_prefix=self.document_prefix,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NvidiaRanker":
        """
        Deserialize the ranker from a dictionary.

        :param data: A dictionary containing the ranker's attributes.
        :returns: The deserialized ranker.
        """
        init_parameters = data.get("init_parameters", {})
        if init_parameters:
            deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Initialize the ranker.

        :raises ValueError: If the API key is required for hosted NVIDIA NIMs.
        """
        if not self._initialized:
            model_kwargs = {}
            if self.truncate is not None:
                model_kwargs.update(truncate=str(self.truncate))
            self.backend = NimBackend(
                model=self.model,
                model_type="ranking",
                api_url=self.api_url,
                api_key=self.api_key,
                model_kwargs=model_kwargs,
                timeout=self.timeout,
                client=self.__class__.__name__,
            )
            if not self.is_hosted and not self._model:
                if self.backend.model:
                    self.model = self.backend.model
            self._initialized = True

    def _prepare_documents_to_embed(self, documents: List[Document]) -> List[str]:
        document_texts = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key])
                for key in self.meta_fields_to_embed
                if key in doc.meta and doc.meta[key]  # noqa: RUF019
            ]
            text_to_embed = self.embedding_separator.join([*meta_values_to_embed, doc.content or ""])
            document_texts.append(self.document_prefix + text_to_embed)
        return document_texts

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        """
        Rank a list of documents based on a given query.

        :param query: The query to rank the documents against.
        :param documents: The list of documents to rank.
        :param top_k: The number of documents to return.

        :raises RuntimeError: If the ranker has not been loaded.
        :raises TypeError: If the arguments are of the wrong type.

        :returns: A dictionary containing the ranked documents.
        """
        if not self._initialized:
            msg = "The ranker has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)
        if not isinstance(query, str):
            msg = "NvidiaRanker expects the `query` parameter to be a string."
            raise TypeError(msg)
        if not isinstance(documents, list):
            msg = "NvidiaRanker expects the `documents` parameter to be a list."
            raise TypeError(msg)
        if not all(isinstance(doc, Document) for doc in documents):
            msg = "NvidiaRanker expects the `documents` parameter to be a list of Document objects."
            raise TypeError(msg)
        if top_k is not None and not isinstance(top_k, int):
            msg = "NvidiaRanker expects the `top_k` parameter to be an integer."
            raise TypeError(msg)

        if len(documents) == 0:
            return {"documents": []}

        top_k = top_k if top_k is not None else self.top_k
        if top_k < 1:
            logger.warning("top_k should be at least 1, returning nothing")
            warnings.warn("top_k should be at least 1, returning nothing", stacklevel=2)
            return {"documents": []}

        assert self.backend is not None

        query_text = self.query_prefix + query
        document_texts = self._prepare_documents_to_embed(documents=documents)

        # rank result is list[{index: int, logit: float}] sorted by logit
        sorted_indexes_and_scores = self.backend.rank(query_text=query_text, document_texts=document_texts)
        sorted_documents = []
        for item in sorted_indexes_and_scores[:top_k]:
            # mutate (don't copy) the document because we're only updating the score
            doc = documents[item["index"]]
            doc.score = item["logit"]
            sorted_documents.append(doc)

        return {"documents": sorted_documents}
