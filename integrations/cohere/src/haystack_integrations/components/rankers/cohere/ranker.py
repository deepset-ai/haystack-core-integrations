from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install cohere'") as cohere_import:
    import cohere
logger = logging.getLogger(__name__)


@component
class CohereRanker:
    """
    A component for performing reranking of documents using Cohere reranking models.

    Reranks retrieved documents based on semantic relevance to a query.
    Documents are indexed from most to least semantically relevant to the query. [Cohere reranker](https://docs.cohere.com/reference/rerank-1)

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.rankers import CohereRanker

    ranker = CohereRanker(model="rerank-english-v2.0", top_k=3)

    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "What is the capital of germany?"
    output = ranker.run(query=query, documents=docs)
    docs = output["documents"]
    ```
    """

    def __init__(
        self,
        model: str = "rerank-english-v2.0",
        top_k: int = 10,
        api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
        api_base_url: Optional[str] = None,
        max_chunks_per_doc: Optional[int] = None,
        meta_fields_to_embed: Optional[List[str]] = None,
        meta_data_separator: Optional[str] = "\n",
    ):
        """
        Creates an instance of the 'CohereRanker'.

        :param model: Cohere model name. Check the list of supported models in the [Cohere documentation](https://docs.cohere.com/docs/models).
        :param top_k: The maximum number of documents to return.
        :param api_key: Cohere API key.
        :param api_base_url: the base URL of the Cohere API.
        :param max_chunks_per_doc: If your document exceeds 512 tokens, this determines the maximum number of
            chunks a document can be split into. If None, the default of 10 is used.
            For example, if your document is 6000 tokens, with the default of 10, the document will be split into 10
            chunks each of 512 tokens and the last 880 tokens will be disregarded.
            Check this [link](https://docs.cohere.com/docs/reranking-best-practices) for more information.
        :param meta_fields_to_embed: List of meta fields that should be concatenated
        with the document content for reranking.
        :param meta_data_separator: Separator to be used to separate the concatenated
        the meta fields and document content.
        """
        cohere_import.check()

        if not api_base_url:
            api_base_url = cohere.COHERE_API_URL
        self.model_name = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.top_k = top_k
        self.max_chunks_per_doc = max_chunks_per_doc
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.meta_data_separator = meta_data_separator
        self.cohere_client = cohere.Client(
            api_key=self.api_key.resolve_value(), api_url=self.api_base_url, client_name="haystack"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model_name,
            api_key=self.api_key.to_dict() if self.api_key else None,
            api_base_url=self.api_base_url,
            top_k=self.top_k,
            max_chunks_per_doc=self.max_chunks_per_doc,
            meta_fields_to_embed=self.meta_fields_to_embed,
            meta_data_separator=self.meta_data_separator,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohereRanker":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_cohere_input_docs(self, documents: List[Document]) -> List[str]:
        """
        Prepare the input by concatenating the document text with the metadata fields specified.
        :param documents: The list of Document objects.

        :return: A list of strings to be input to the cohere model.
        """
        concatenated_input_list = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta.get(key)
            ]
            if self.meta_data_separator is not None:
                concatenated_input = self.meta_data_separator.join([*meta_values_to_embed, doc.content or ""])
            else:
                concatenated_input = "".join([*meta_values_to_embed, doc.content or ""])
            concatenated_input_list.append(concatenated_input)

        return concatenated_input_list

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Use the Cohere Reranker to re-rank the list of documents based on the query.

        :param query: The query string.
        :param documents: List of Document objects to be re-ranked.
        :param top_k: Optional. An integer to override the top_k set during initialization.

        :returns: A dictionary with the following key:
            - `documents`: List of re-ranked Document objects.

        :raises ValueError: If the top_k value is less than or equal to 0.
        """
        if top_k is None:
            top_k = self.top_k
        if top_k <= 0:
            msg = f"top_k must be > 0, but got {top_k}"
            raise ValueError(msg)

        cohere_input_docs = self._prepare_cohere_input_docs(documents)
        max_num_docs_for_cohere_ranker = 1000
        if len(cohere_input_docs) > max_num_docs_for_cohere_ranker:
            logger.warning(
                f"The Cohere reranking endpoint only supports 1000 documents.\
                The number of documents has been truncated to 1000 from {len(cohere_input_docs)}."
            )
            cohere_input_docs = cohere_input_docs[:max_num_docs_for_cohere_ranker]

        response = self.cohere_client.rerank(
            model=self.model_name, query=query, documents=cohere_input_docs, max_chunks_per_doc=self.max_chunks_per_doc
        )
        indices = [output.index for output in response.results]
        scores = [output.relevance_score for output in response.results]
        sorted_docs = []
        for idx, score in zip(indices, scores):
            doc = documents[idx]
            doc.score = score
            sorted_docs.append(documents[idx])
        return {"documents": sorted_docs[:top_k]}
