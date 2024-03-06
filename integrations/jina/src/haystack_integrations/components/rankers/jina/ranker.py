# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Dict, List, Optional

import requests
from haystack import Document, component

JINA_API_URL: str = "https://api.jina.ai/v1/rerank"


@component
class JinaRanker:
    """
    Ranks Documents based on their similarity to the query using Jina AI models..

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.rankers import JinaRanker

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
            model_name: str = "jinaai/jina-reranker-v1-base-en",
            api_key: Optional[str] = None,
            top_k: int = 10,
            query_prefix: str = "",
            document_prefix: str = "",
            score_threshold: Optional[float] = None,
    ):
        """
        Creates an instance of JinaRanker.

        :param api_key: The Jina API key. It can be explicitly provided or automatically read from the
            environment variable JINA_API_KEY (recommended).
        :param model_name: The name of the Jina model to use. Check the list of available models on `https://jina.ai/embeddings/`
        :param top_k:
            The maximum number of Documents to return per query.
        :param query_prefix:
            A string to add to the beginning of the query text before ranking.
            Can be used to prepend the text with an instruction, as required by some reranking models, such as bge.
        :param document_prefix:
            A string to add to the beginning of each Document text before ranking. Can be used to prepend the text with
            an instruction, as required by some embedding models, such as bge.
        :param score_threshold:
            If provided only returns documents with a score above this threshold.

        :raises ValueError:
            If `top_k` is not > 0.
            If `scale_score` is True and `calibration_factor` is not provided.
        """
        # if the user does not provide the API key, check if it is set in the module client
        if api_key is None:
            try:
                api_key = os.environ["JINA_API_KEY"]
            except KeyError as e:
                msg = (
                    "JinaRanker expects a Jina API key. "
                    "Set the JINA_API_KEY environment variable (recommended) or pass it explicitly."
                )
                raise ValueError(msg) from e
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.top_k = top_k
        self.score_threshold = score_threshold

        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        # if the user does not provide the API key, check if it is set in the module client
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

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

        top_k = top_k or self.top_k
        score_threshold = score_threshold or self.score_threshold

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")
        resp = self._session.post(  # type: ignore
            JINA_API_URL,
            json={
                "query": query,
                "documents": [doc.content or "" for doc in documents],
                "model": self.model,
                "top_n": top_k,
            },
        ).json()
        if "results" not in resp:
            raise RuntimeError(resp["detail"])

        results = resp["results"]

        ranked_docs = []
        for result in results:
            index = result["index"]
            relevance_score = results["relevance_score"]
            if top_k is None or len(ranked_docs) < top_k:
                if score_threshold is not None:
                    if relevance_score >= score_threshold:
                        ranked_docs.append(documents[index])
                else:
                    ranked_docs.append(documents[index])
            else:
                break

        return {"documents": ranked_docs}