# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.rankers.nvidia.truncate import RankerTruncateMode
from haystack_integrations.utils.nvidia import NimBackend, url_validation

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "nvidia/nv-rerankqa-mistral-4b-v3"

_MODEL_ENDPOINT_MAP = {
    "nvidia/nv-rerankqa-mistral-4b-v3": "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking",
    "nvidia/llama-3.2-nv-rerankqa-1b-v1": "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v1/reranking",
}


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
        api_url: Optional[str] = None,
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        top_k: int = 5,
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
        self._model = model or _DEFAULT_MODEL
        self._truncate = truncate
        self._api_key = api_key
        # if no api_url is provided, we're using a hosted model and can
        #  - assume the default url will work, because there's only one model
        #  - assume we won't call backend.models()
        if api_url is not None:
            self._api_url = url_validation(api_url, None, ["v1/ranking"])
            self._endpoint = None  # we let backend.rank() handle the endpoint
        else:
            if self._model not in _MODEL_ENDPOINT_MAP:
                msg = f"Model '{model}' is unknown. Please provide an api_url to access it."
                raise ValueError(msg)
            self._api_url = None  # we handle the endpoint
            self._endpoint = _MODEL_ENDPOINT_MAP[self._model]
            if api_key is None:
                self._api_key = Secret.from_env_var("NVIDIA_API_KEY")
        self._top_k = top_k
        self._initialized = False
        self._backend: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the ranker to a dictionary.

        :returns: A dictionary containing the ranker's attributes.
        """
        return default_to_dict(
            self,
            model=self._model,
            top_k=self._top_k,
            truncate=self._truncate,
            api_url=self._api_url,
            api_key=self._api_key.to_dict() if self._api_key else None,
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
            if self._truncate is not None:
                model_kwargs.update(truncate=str(self._truncate))
            self._backend = NimBackend(
                self._model,
                api_url=self._api_url,
                api_key=self._api_key,
                model_kwargs=model_kwargs,
            )
            if not self._model:
                self._model = _DEFAULT_MODEL
            self._initialized = True

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

        top_k = top_k if top_k is not None else self._top_k
        if top_k < 1:
            logger.warning("top_k should be at least 1, returning nothing")
            warnings.warn("top_k should be at least 1, returning nothing", stacklevel=2)
            return {"documents": []}

        assert self._backend is not None
        # rank result is list[{index: int, logit: float}] sorted by logit
        sorted_indexes_and_scores = self._backend.rank(
            query,
            documents,
            endpoint=self._endpoint,
        )
        sorted_documents = []
        for item in sorted_indexes_and_scores[:top_k]:
            # mutate (don't copy) the document because we're only updating the score
            doc = documents[item["index"]]
            doc.score = item["logit"]
            sorted_documents.append(doc)

        return {"documents": sorted_documents}
