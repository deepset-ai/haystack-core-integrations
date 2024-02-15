# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from .embedding_backend.instructor_backend import _InstructorEmbeddingBackendFactory


@component
class InstructorDocumentEmbedder:
    """
    A component for computing Document embeddings using INSTRUCTOR embedding models.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    # To use this component, install the "instructor-embedders-haystack" package.
    # pip install instructor-embedders-haystack

    from instructor_embedders_haystack.instructor_document_embedder import InstructorDocumentEmbedder
    from haystack.dataclasses import Document


    doc_embedding_instruction = "Represent the Medical Document for retrieval:"

    doc_embedder = InstructorDocumentEmbedder(
        model="hkunlp/instructor-base",
        instruction=doc_embedding_instruction,
        batch_size=32,
        device="cpu",
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
        model: str = "hkunlp/instructor-base",
        device: Optional[str] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),  # noqa: B008
        instruction: str = "Represent the document",
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create an InstructorDocumentEmbedder component.

        :param model: Local path or name of the model in Hugging Face's model hub,
            such as ``'hkunlp/instructor-base'``.
        :param device: Device (like 'cuda' / 'cpu') that should be used for computation.
            If None, checks if a GPU can be used.
        :param use_auth_token: An API token used to download private models from Hugging Face.
            If this parameter is set to `True`, then the token generated when running
            `transformers-cli login` (stored in ~/.huggingface) will be used.
        :param instruction: The instruction string to be used while computing domain-specific embeddings.
            The instruction follows the unified template of the form:
            "Represent the 'domain' 'text_type' for 'task_objective'", where:
            - "domain" is optional, and it specifies the domain of the text, e.g., science, finance, medicine, etc.
            - "text_type" is required, and it specifies the encoding unit, e.g., sentence, document, paragraph, etc.
            - "task_objective" is optional, and it specifies the objective of embedding, e.g., retrieve a document,
             classify the sentence, etc.
            Check some examples of instructions here: https://github.com/xlang-ai/instructor-embedding#use-cases.
        :param batch_size: Number of strings to encode at once.
        :param progress_bar: If true, displays progress bar during embedding.
        :param normalize_embeddings: If set to true, returned vectors will have the length of 1.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document content.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document content.
        """

        self.model = model
        # TODO: remove device parameter and use Haystack's device management once migrated
        self.device = device or "cpu"
        self.token = token
        self.instruction = instruction
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model,
            device=self.device,
            token=self.token.to_dict() if self.token else None,
            instruction=self.instruction,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructorDocumentEmbedder":
        """
        Deserialize this component from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Load the embedding backend.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = _InstructorEmbeddingBackendFactory.get_embedding_backend(
                model=self.model, device=self.device, token=self.token
            )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.
        The embedding of each Document is stored in the `embedding` field of the Document.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            msg = (
                "InstructorDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a list of strings, please use the InstructorTextEmbedder."
            )
            raise TypeError(msg)
        if not hasattr(self, "embedding_backend"):
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        # TODO: once non textual Documents are properly supported, we should also prepare them for embedding here

        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            text_to_embed = [
                self.instruction,
                self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]),
            ]
            texts_to_embed.append(text_to_embed)

        embeddings = self.embedding_backend.embed(
            texts_to_embed,
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents}
