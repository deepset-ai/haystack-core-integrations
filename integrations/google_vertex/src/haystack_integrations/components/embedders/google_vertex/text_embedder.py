

import os
from typing import Any, Dict, List, Literal, Optional, Union
from haystack import component, default_from_dict, default_to_dict, logging

import vertexai
from haystack import Document, component
from tqdm import tqdm
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from haystack.utils.auth import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

@component
class VertexAITextEmbedder:
    """
    Embed text using VertexAI Text Embedder API

    Available models found here:
    https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#syntax
    """

    def __init__(
        self,
        model: Literal["text-embedding-004",
                       "text-embedding-005",
                       "textembedding-gecko-multilingual@001",
                       "text-multilingual-embedding-002",
                       "text-embedding-large-exp-03-07"],
        task_type: Literal["RETRIEVAL_DOCUMENT",
                           "RETRIEVAL_QUERY",
                           "SEMANTIC_SIMILARITY", "CLASSIFICATION", "CLUSTERING", "QUESTION_ANSWERING", "FACT_VERIFICATION", "CODE_RETRIEVAL_QUERY"],  # link https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#tasktype
        gcp_region_name: Optional[Secret] = Secret.from_env_var("GCP_DEFAULT_REGION", strict=False),  # noqa: B008
        gcp_project_id: Optional[Secret] = Secret.from_env_var("GCP_PROJECT_ID", strict=False),  # noqa: B008
        progress_bar: bool = True,
        truncate_dim: Optional[int] = None,
    ) -> None:
        self.model = model
        self.progress_bar = progress_bar
        self.truncate_dim = truncate_dim

        self.gcp_project_id = gcp_project_id
        self.gcp_region_name = gcp_region_name


        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None
        
        vertexai.init(project=resolve_secret(self.gcp_project_id), location=resolve_secret(self.gcp_region_name))        
        self.embedder = TextEmbeddingModel.from_pretrained(self.model)
        self.task_type = task_type

    @component.output_types(embedding=List[float])
    def run(self, text: Union[List[Document], List[str], str]):
        if not isinstance(text, str):
            msg = (
                "FastembedTextEmbedder expects a string as input. "
                "In case you want to embed a list of Documents, please use the VertexAIDocumentEmbedder."
            )
            raise TypeError(msg)
        
        text_embed_input = [TextEmbeddingInput(text=text, task_type=self.task_type)]
        embeddings = self.embedder.get_embeddings(text_embed_input)[0].values
        return {"embedding": embeddings}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            aws_access_key_id=self.aws_access_key_id.to_dict() if self.aws_access_key_id else None,
            aws_secret_access_key=self.aws_secret_access_key.to_dict() if self.aws_secret_access_key else None,
            aws_session_token=self.aws_session_token.to_dict() if self.aws_session_token else None,
            aws_region_name=self.aws_region_name.to_dict() if self.aws_region_name else None,
            aws_profile_name=self.aws_profile_name.to_dict() if self.aws_profile_name else None,
            model=self.model,
            boto3_config=self.boto3_config,
            **self.kwargs,
        )


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
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAITextEmbedder":
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