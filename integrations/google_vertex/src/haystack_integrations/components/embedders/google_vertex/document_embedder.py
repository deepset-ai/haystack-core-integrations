import math
import time
from typing import Any, Dict, List, Literal, Optional

import vertexai
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

logger = logging.getLogger(__name__)

SUPPORTED_EMBEDDING_MODELS = [
    "text-embedding-004",
    "text-embedding-005",
    "textembedding-gecko-multilingual@001",
    "text-multilingual-embedding-002",
    "text-embedding-large-exp-03-07",
]

logger = logging.getLogger(__name__)


@component
class VertexAIDocumentEmbedder:
    """
    Embed text using Vertex AI Embeddings API.

    See available models in the official
    [Google documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#syntax).

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.google_vertex import VertexAIDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = VertexAIDocumentEmbedder(model="text-embedding-005")

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)
    # [-0.044606007635593414, 0.02857724390923977, -0.03549133986234665,
    ```
    """

    def __init__(
        self,
        model: Literal[
            "text-embedding-004",
            "text-embedding-005",
            "textembedding-gecko-multilingual@001",
            "text-multilingual-embedding-002",
            "text-embedding-large-exp-03-07",
        ],
        task_type: Literal[
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
            "CODE_RETRIEVAL_QUERY",
        ] = "RETRIEVAL_DOCUMENT",
        gcp_region_name: Optional[Secret] = Secret.from_env_var("GCP_DEFAULT_REGION", strict=False),  # noqa: B008
        gcp_project_id: Optional[Secret] = Secret.from_env_var("GCP_PROJECT_ID", strict=False),  # noqa: B008
        batch_size: int = 32,
        max_tokens_total: int = 20000,
        time_sleep: int = 30,  # seconds
        retries: int = 3,
        progress_bar: bool = True,
        truncate_dim: Optional[int] = None,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ) -> None:
        """
        Generate Document Embedder using a Google Vertex AI model.

        Authenticates using Google Cloud Application Default Credentials (ADCs).
        For more information see the official [Google documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc).

        :param model: Name of the model to use.
        :param task_type: The type of task for which the embeddings are being generated.
                        For more information see the official [Google documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#tasktype).
        :param gcp_region_name: The default location to use when making API calls, if not set uses us-central-1.
        :param gcp_project_id: ID of the GCP project to use. By default, it is set during Google Cloud authentication.
        :param batch_size: The number of documents to process in a single batch.
        :param max_tokens_total: The maximum number of tokens to process in total.
        :param time_sleep: The time to sleep between retries in seconds.
        :param retries: The number of retries in case of failure.
        :param progress_bar: Whether to display a progress bar during processing.
        :param truncate_dim: The dimension to truncate the embeddings to, if specified.
        :param meta_fields_to_embed: A list of metadata fields to include in the embeddings.
        :param embedding_separator: The separator to use between different embeddings.

        :raises ValueError: If the provided model is not in the list of supported models.
        """
        soft_deprecation_msg = (
            "This component uses a deprecated SDK. We recommend using the GoogleGenAIDocumentEmbedder instead."
            "Documentation is available at https://docs.haystack.deepset.ai/docs/googlegenaidocumentembedder."
        )
        logger.warning(soft_deprecation_msg)

        if meta_fields_to_embed is None:
            meta_fields_to_embed = []
        if not model or model not in SUPPORTED_EMBEDDING_MODELS:
            msg = "Please provide a valid model from the list of supported models: " + ", ".join(
                SUPPORTED_EMBEDDING_MODELS
            )
            raise ValueError(msg)

        self.model = model
        self.batch_size = batch_size
        self.max_tokens_total = max_tokens_total
        self.time_sleep = time_sleep

        self.progress_bar = progress_bar
        self.truncate_dim = truncate_dim
        self.meta_fields_to_embed = meta_fields_to_embed
        self.embedding_separator = embedding_separator

        self.gcp_project_id = gcp_project_id
        self.gcp_region_name = gcp_region_name

        self.retries = retries

        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        vertexai.init(project=resolve_secret(self.gcp_project_id), location=resolve_secret(self.gcp_region_name))
        self.embedder = TextEmbeddingModel.from_pretrained(self.model)
        self.task_type = task_type

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [str(doc.meta[key]) for key in self.meta_fields_to_embed if doc.meta.get(key)]  # type: ignore

            text_to_embed = self.embedding_separator.join([*meta_values_to_embed, doc.content or ""])

            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def get_text_embedding_input(self, batch: List[Document]) -> List[TextEmbeddingInput]:
        """
        Converts a batch of Document objects into a list of TextEmbeddingInput objects.

        Args:
            batch (List[Document]): A list of Document objects to be converted.

        Returns:
            List[TextEmbeddingInput]: A list of TextEmbeddingInput objects created from the input documents.
        """
        texts_to_embed = self._prepare_texts_to_embed(documents=batch)
        return [TextEmbeddingInput(text=content, task_type=self.task_type) for content in texts_to_embed]

    def embed_batch_by_smaller_batches(self, batch: List[str], subbatch=1) -> List[List[float]]:
        """
        Embeds a batch of text strings by dividing them into smaller sub-batches.
        Args:
            batch (List[str]): A list of text strings to be embedded.
            subbatch (int, optional): The size of the smaller sub-batches. Defaults to 1.
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        Raises:
            Exception: If embedding fails at the item level, an exception is raised with the error details.
        """

        logger.debug(f"Embedding by smaller batches {subbatch}")
        embeddings_batch = []
        try:
            for i in range(0, len(batch), subbatch):
                text_embedding_input = self.get_text_embedding_input(batch[i : i + subbatch])
                embeddings = [
                    item.values
                    for item in self.embedder.get_embeddings(text_embedding_input, auto_truncate=self.truncate_dim)
                ]
                embeddings_batch.extend(embeddings)
        except Exception:
            try:
                return self.embed_batch_by_smaller_batches(batch, subbatch=1)
            except Exception as e:
                logger.info("Failing item per item ")
                logger.info(e)
                raise Exception(e) from e

        return embeddings_batch

    def embed_batch(self, batch: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of text strings.

        Args:
            batch (List[str]): A list of text strings to be embedded.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        text_embedding_input = self.get_text_embedding_input(batch)
        embeddings = [item.values for item in self.embedder.get_embeddings(text_embedding_input)]

        return embeddings

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Processes all documents in batches while adhering to the API's token limit per request.

        :param documents: A list of documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
        """
        # Initial batch size
        all_embeddings = []
        logger.info(f"Starting embedding process for {len(documents)} documents with a batch size of {self.batch_size}")

        # for enum_batch, i in enumerate(range(0, len(documents), batch_size)):
        i = 0
        batch_number = 0
        while i < len(documents):
            batch_size = self.batch_size  # reset to 30
            batch = documents[i : i + batch_size]
            batch_text = self._prepare_texts_to_embed(batch)
            total_tokens = self.embedder.count_tokens(batch_text).total_tokens
            logger.debug(f"Batch {batch_number}, batch_size: {batch_size} total_tokens {total_tokens}")

            # Adjust batch size dynamically based on token count
            while total_tokens > self.max_tokens_total:
                batch_size = max(batch_size - 3, 1)  # math.ceil(batch_size / 2)
                logger.debug(f"Batch {batch_number} Reducing batch size to {batch_size}")
                logger.debug(f"due to token limit (total_tokens = {total_tokens})")
                batch = documents[i : i + batch_size]
                batch_text = self._prepare_texts_to_embed(batch)
                total_tokens = self.embedder.count_tokens(batch_text).total_tokens

                if batch_size == 1:
                    break

            retries = self.retries
            while retries > 0:
                try:
                    all_embeddings.extend(self.embed_batch(batch))

                    break
                except Exception as e:
                    logger.debug(f"Batch {batch_number} Error tokens, {e}")
                    logger.debug(f"Batch {batch_number} Embedding by smaller batchers")
                    logger.debug(f"batch_size reduced to {math.ceil(batch_size / 2)})")

                    try:
                        all_embeddings.extend(
                            self.embed_batch_by_smaller_batches(batch, subbatch=math.ceil(batch_size / 2))
                        )
                        break

                    except Exception as e:
                        logger.info(f"{retries}/3: Maximum 500 quota achieved")
                        logger.info(f"Waiting {self.time_sleep} seconds before retrying")
                        logger.info(e)
                        retries -= 1
                        if retries == 0:
                            time.sleep(self.time_sleep)  # Wait before retrying
                exception_msg = "Exceeded maximum retries for API call"
                raise Exception(exception_msg)
            i += batch_size
            batch_number += 1

        for doc, embeddings in zip(documents, all_embeddings):
            doc.embedding = embeddings

        return {"documents": documents}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            gcp_project_id=self.gcp_project_id.to_dict() if self.gcp_project_id else None,
            gcp_region_name=self.gcp_region_name.to_dict() if self.gcp_region_name else None,
            model=self.model,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            truncate_dim=self.truncate_dim,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            max_tokens_total=self.max_tokens_total,
            task_type=self.task_type,
            time_sleep=self.time_sleep,
            retries=self.retries,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAIDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["gcp_project_id", "gcp_region_name"],
        )
        return default_from_dict(cls, data)
