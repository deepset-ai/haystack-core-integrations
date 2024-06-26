from typing import Any, Dict, List, Optional

from haystack import Document, component, default_to_dict

from .embedding_backend.fastembed_backend import _FastembedSparseEmbeddingBackendFactory


@component
class FastembedSparseDocumentEmbedder:
    """
    FastembedSparseDocumentEmbedder computes Document embeddings using Fastembed sparse models.

    Usage example:
    ```python
    from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
    from haystack.dataclasses import Document

    sparse_doc_embedder = FastembedSparseDocumentEmbedder(
        model="prithvida/Splade_PP_en_v1",
        batch_size=32,
    )

    sparse_doc_embedder.warm_up()

    # Text taken from PubMed QA Dataset (https://huggingface.co/datasets/pubmed_qa)
    document_list = [
        Document(
            content=("Oxidative stress generated within inflammatory joints can produce autoimmune phenomena and joint "
                     "destruction. Radical species with oxidative activity, including reactive nitrogen species, "
                     "represent mediators of inflammation and cartilage damage."),
            meta={
                "pubid": "25,445,628",
                "long_answer": "yes",
            },
        ),
        Document(
            content=("Plasma levels of pancreatic polypeptide (PP) rise upon food intake. Although other pancreatic "
                     "islet hormones, such as insulin and glucagon, have been extensively investigated, PP secretion "
                     "and actions are still poorly understood."),
            meta={
                "pubid": "25,445,712",
                "long_answer": "yes",
            },
        ),
    ]

    result = sparse_doc_embedder.run(document_list)
    print(f"Document Text: {result['documents'][0].content}")
    print(f"Document Sparse Embedding: {result['documents'][0].sparse_embedding}")
    print(f"Sparse Embedding Dimension: {len(result['documents'][0].sparse_embedding)}")
    ```
    """

    def __init__(
        self,
        model: str = "prithvida/Splade_PP_en_v1",
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        batch_size: int = 32,
        progress_bar: bool = True,
        parallel: Optional[int] = None,
        local_files_only: bool = False,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create an FastembedDocumentEmbedder component.

        :param model: Local path or name of the model in Hugging Face's model hub,
            such as `prithvida/Splade_PP_en_v1`.
        :param cache_dir: The path to the cache directory.
                Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                Defaults to `fastembed_cache` in the system's temp directory.
        :param threads: The number of threads single onnxruntime session can use.
        :param batch_size: Number of strings to encode at once.
        :param progress_bar: If `True`, displays progress bar during embedding.
        :param parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.
        :param local_files_only: If `True`, only use the model files in the `cache_dir`.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document content.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document content.
        """

        self.model_name = model
        self.cache_dir = cache_dir
        self.threads = threads
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.parallel = parallel
        self.local_files_only = local_files_only
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model_name,
            cache_dir=self.cache_dir,
            threads=self.threads,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            parallel=self.parallel,
            local_files_only=self.local_files_only,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    def warm_up(self):
        """
        Initializes the component.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = _FastembedSparseEmbeddingBackendFactory.get_embedding_backend(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=self.threads,
                local_files_only=self.local_files_only,
            )

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            text_to_embed = self.embedding_separator.join([*meta_values_to_embed, doc.content or ""])

            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embeds a list of Documents.

        :param documents: List of Documents to embed.
        :returns: A dictionary with the following keys:
            - `documents`: List of Documents with each Document's `sparse_embedding`
                            field set to the computed embeddings.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            msg = (
                "FastembedSparseDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a list of strings, please use the FastembedTextEmbedder."
            )
            raise TypeError(msg)
        if not hasattr(self, "embedding_backend"):
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)
        embeddings = self.embedding_backend.embed(
            texts_to_embed,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            parallel=self.parallel,
        )

        for doc, emb in zip(documents, embeddings):
            doc.sparse_embedding = emb
        return {"documents": documents}
