# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

import requests
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

JINA_API_URL: str = "https://api.jina.ai/v1/rerank"


@component
class JinaRanker:
    """
    Ranks Documents based on their similarity to the query using Jina AI models.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.rankers.jina import JinaRanker

    ranker = JinaRanker()
    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "City in Germany"
    result = ranker.run(query=query, documents=docs)
    docs = result["documents"]
    print(docs[0].content)
    ```
    """

    def __init__(
        self,
        model: str = "jina-reranker-v1-base-en",
        api_key: Secret = Secret.from_env_var("JINA_API_KEY"),  # noqa: B008,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ):
        """
        Creates an instance of JinaRanker.

        :param api_key: The Jina API key. It can be explicitly provided or automatically read from the
            environment variable JINA_API_KEY (recommended).
        :param model: The name of the Jina model to use. Check the list of available models on `https://jina.ai/reranker/`
        :param top_k:
            The maximum number of Documents to return per query. If `None`, all documents are returned
        :param score_threshold:
            If provided only returns documents with a score above this threshold.

        :raises ValueError:
            If `top_k` is not > 0.
        """
        # if the user does not provide the API key, check if it is set in the module client
        resolved_api_key = api_key.resolve_value()
        self.api_key = api_key
        self.model = model
        self.top_k = top_k
        self.score_threshold = score_threshold

        if self.top_k is not None and self.top_k <= 0:
            msg = f"top_k must be > 0, but got {top_k}"
            raise ValueError(msg)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {resolved_api_key}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JinaRanker":
        """
        Deserializes the component from a dictionary.
        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ):
        """
        Returns a list of Documents ranked by their similarity to the given query.

        :param query:
            Query string.
        :param documents:
            List of Documents.
        :param top_k:
            The maximum number of Documents you want the Ranker to return.
        :param score_threshold:
            If provided only returns documents with a score above this threshold.
        :returns:
            A dictionary with the following keys:
            - `documents`: List of Documents most similar to the given query in descending order of similarity.

        :raises ValueError:
            If `top_k` is not > 0.
        """
        if not documents:
            return {"documents": []}

        if top_k is not None and top_k <= 0:
            msg = f"top_k must be > 0, but got {top_k}"
            raise ValueError(msg)

        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold

        data = {
            "query": query,
            "documents": [doc.content or "" for doc in documents],
            "model": self.model,
            "top_n": top_k,
        }

        resp = self._session.post(
            JINA_API_URL,
            json=data,
        ).json()

        if "results" not in resp:
            raise RuntimeError(resp["detail"])

        results = resp["results"]

        ranked_docs: List[Document] = []
        for result in results:
            index = result["index"]
            relevance_score = result["relevance_score"]
            doc = documents[index]
            if top_k is None or len(ranked_docs) < top_k:
                doc.score = relevance_score
                if score_threshold is not None:
                    if relevance_score >= score_threshold:
                        ranked_docs.append(doc)
                else:
                    ranked_docs.append(doc)
            else:
                break

        return {"documents": ranked_docs, "meta": {"model": resp["model"], "usage": resp["usage"]}}
