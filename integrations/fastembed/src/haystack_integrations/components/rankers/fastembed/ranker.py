# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging

from fastembed.rerank.cross_encoder import TextCrossEncoder

logger = logging.getLogger(__name__)


@component
class FastembedRanker:
    """
    Ranks Documents based on their similarity to the query using
    [Fastembed models](https://qdrant.github.io/fastembed/examples/Supported_Models/).

    Documents are indexed from most to least semantically relevant to the query.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.rankers.fastembed import FastembedRanker

    ranker = FastembedRanker(model_name="Xenova/ms-marco-MiniLM-L-6-v2", top_k=2)

    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "What is the capital of germany?"
    output = ranker.run(query=query, documents=docs)
    print(output["documents"][0].content)

    # Berlin
    ```
    """

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        top_k: int = 10,
        cache_dir: str | None = None,
        threads: int | None = None,
        batch_size: int = 64,
        parallel: int | None = None,
        local_files_only: bool = False,
        meta_fields_to_embed: list[str] | None = None,
        meta_data_separator: str = "\n",
    ):
        """
        Creates an instance of the 'FastembedRanker'.

        :param model_name: Fastembed model name. Check the list of supported models in the [Fastembed documentation](https://qdrant.github.io/fastembed/examples/Supported_Models/).
        :param top_k: The maximum number of documents to return.
        :param cache_dir: The path to the cache directory.
                Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                Defaults to `fastembed_cache` in the system's temp directory.
        :param threads: The number of threads single onnxruntime session can use. Defaults to None.
        :param batch_size: Number of strings to encode at once.
        :param parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.
        :param local_files_only: If `True`, only use the model files in the `cache_dir`.
        :param meta_fields_to_embed: List of meta fields that should be concatenated
            with the document content for reranking.
        :param meta_data_separator: Separator used to concatenate the meta fields
            to the Document content.
        """
        if top_k <= 0:
            msg = f"top_k must be > 0, but got {top_k}"
            raise ValueError(msg)

        self.model_name = model_name
        self.top_k = top_k
        self.cache_dir = cache_dir
        self.threads = threads
        self.batch_size = batch_size
        self.parallel = parallel
        self.local_files_only = local_files_only
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.meta_data_separator = meta_data_separator
        self._model: TextCrossEncoder | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model_name=self.model_name,
            top_k=self.top_k,
            cache_dir=self.cache_dir,
            threads=self.threads,
            batch_size=self.batch_size,
            parallel=self.parallel,
            local_files_only=self.local_files_only,
            meta_fields_to_embed=self.meta_fields_to_embed,
            meta_data_separator=self.meta_data_separator,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FastembedRanker":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Initializes the component.
        """
        if self._model is None:
            self._model = TextCrossEncoder(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=self.threads,
                local_files_only=self.local_files_only,
            )

    def _prepare_fastembed_input_docs(self, documents: list[Document]) -> list[str]:
        """
        Prepare the input by concatenating the document text with the metadata fields specified.
        :param documents: The list of Document objects.

        :return: A list of strings to be given as input to Fastembed model.
        """
        concatenated_input_list = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta.get(key)
            ]
            concatenated_input = self.meta_data_separator.join([*meta_values_to_embed, doc.content or ""])
            concatenated_input_list.append(concatenated_input)

        return concatenated_input_list

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document], top_k: int | None = None) -> dict[str, list[Document]]:
        """
        Returns a list of documents ranked by their similarity to the given query, using FastEmbed.

        :param query:
            The input query to compare the documents to.
        :param documents:
            A list of documents to be ranked.
        :param top_k:
            The maximum number of documents to return.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents closest to the query, sorted from most similar to least similar.

        :raises ValueError: If `top_k` is not > 0.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = "FastembedRanker expects a list of Documents as input. "
            raise TypeError(msg)
        if query == "":
            msg = "No query provided"
            raise ValueError(msg)

        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k
        if top_k <= 0:
            msg = f"top_k must be > 0, but got {top_k}"
            raise ValueError(msg)

        if self._model is None:
            self.warm_up()
        assert self._model is not None  
        
        fastembed_input_docs = self._prepare_fastembed_input_docs(documents)

        scores = list(
            self._model.rerank(
                query=query,
                documents=fastembed_input_docs,
                batch_size=self.batch_size,
                parallel=self.parallel,
            )
        )

        # Combine the two lists into a single list of tuples
        doc_scores = list(zip(documents, scores, strict=True))

        # Sort the list of tuples by the score in descending order
        sorted_doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        # Get the top_k documents
        top_k_documents = []
        for doc, score in sorted_doc_scores[:top_k]:
            doc.score = score
            top_k_documents.append(doc)

        return {"documents": top_k_documents}
