from typing import Any, Dict, List, Optional

from haystack import Document, component, default_to_dict

from .embedding_backend.fastembed_backend import _FastembedEmbeddingBackendFactory


@component
class FastembedDocumentEmbedder:
    """
    A component for computing Document embeddings using Fastembed embedding models.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    # To use this component, install the "fastembed-haystack" package.
    # pip install fastembed-haystack

    from haystack_integrations.components.embedders.fastembed import FastembedDocumentEmbedder
    from haystack.dataclasses import Document

    doc_embedder = FastembedDocumentEmbedder(
        model="BAAI/bge-small-en-v1.5",
        batch_size=256,
    )

    doc_embedder.warm_up()

    # Text taken from PubMed QA Dataset (https://huggingface.co/datasets/pubmed_qa)
    document_list = [
        Document(
            content="Oxidative stress generated within inflammatory joints can produce autoimmune phenomena and joint destruction. Radical species with oxidative activity, including reactive nitrogen species, represent mediators of inflammation and cartilage damage.",
            meta={
                "pubid": "25,445,628",
                "long_answer": "yes",
            },
        ),
        Document(
            content="Plasma levels of pancreatic polypeptide (PP) rise upon food intake. Although other pancreatic islet hormones, such as insulin and glucagon, have been extensively investigated, PP secretion and actions are still poorly understood.",
            meta={
                "pubid": "25,445,712",
                "long_answer": "yes",
            },
        ),
    ]

    result = doc_embedder.run(document_list)
    print(f"Document Text: {result['documents'][0].content}")
    print(f"Document Embedding: {result['documents'][0].embedding}")
    print(f"Embedding Dimension: {len(result['documents'][0].embedding)}")
    """  # noqa: E501

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 256,
        progress_bar: bool = True,
        parallel: Optional[int] = None,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create an FastembedDocumentEmbedder component.

        :param model: Local path or name of the model in Hugging Face's model hub,
            such as ``'BAAI/bge-small-en-v1.5'``.
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param batch_size: Number of strings to encode at once.
        :param progress_bar: If true, displays progress bar during embedding.
        :param parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document content.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document content.
        """

        self.model_name = model
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.parallel = parallel
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model_name,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            parallel=self.parallel,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    def warm_up(self):
        """
        Load the embedding backend.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = _FastembedEmbeddingBackendFactory.get_embedding_backend(model_name=self.model_name)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            text_to_embed = [
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix,
            ]

            texts_to_embed.append(text_to_embed[0])
        return texts_to_embed

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.
        The embedding of each Document is stored in the `embedding` field of the Document.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            msg = (
                "FastembedDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a list of strings, please use the FastembedTextEmbedder."
            )
            raise TypeError(msg)
        if not hasattr(self, "embedding_backend"):
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        # TODO: once non textual Documents are properly supported, we should also prepare them for embedding here

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)
        embeddings = self.embedding_backend.embed(
            texts_to_embed,
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            parallel=self.parallel,
        )

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents}
