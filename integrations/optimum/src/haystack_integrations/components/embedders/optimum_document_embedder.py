from typing import Any, Dict, List, Optional

import numpy as np
import torch
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.hf import HFModelType, check_valid_model, deserialize_hf_model_kwargs, serialize_hf_model_kwargs
from optimum.onnxruntime import ORTModelForFeatureExtraction
from tqdm import tqdm
from transformers import AutoTokenizer


class OptimumDocumentEmbedder:
    """
    A component for computing Document embeddings using models loaded with the HuggingFace Optimum library.
    This component is designed to seamlessly inference models using the high speed ONNX runtime.

    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack.dataclasses import Document
    from haystack_integrations.components.embedders import OptimumDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = OptimumDocumentEmbedder(model="BAAI/bge-small-en-v1.5")
    document_embedder.warm_up()

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```

    Key Features and Compatibility:
        - **Primary Compatibility**: Designed to work seamlessly with any embedding model present on the Hugging Face
        Hub.
        - **Conversion to ONNX**: The models are converted to ONNX using the HuggingFace Optimum library. This is
        performed in real-time, during the warm-up step.
        - **Accelerated Inference on GPU**: Supports using different execution providers such as CUDA and TensorRT, to
        accelerate ONNX Runtime inference on GPUs.
        Simply pass the execution provider as the onnx_execution_provider parameter. Additonal parameters can be passed
        to the model using the model_kwargs parameter.
        For more details refer to the HuggingFace documentation:
        https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),  # noqa: B008
        prefix: str = "",
        suffix: str = "",
        normalize_embeddings: bool = True,
        onnx_execution_provider: str = "CPUExecutionProvider",
        model_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create a OptimumDocumentEmbedder component.

        :param model: A string representing the model id on HF Hub. Default is "BAAI/bge-small-en-v1.5".
        :param token: The HuggingFace token to use as HTTP bearer authorization.
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param normalize_embeddings: Whether to normalize the embeddings to unit length.
        :param onnx_execution_provider: The execution provider to use for ONNX models. Defaults to
        "CPUExecutionProvider".
        :param model_kwargs: Dictionary containing additional keyword arguments to pass to the model.
            In case of duplication, these kwargs override `model`, `onnx_execution_provider`, and `token` initialization
            parameters.
        :param batch_size: Number of Documents to encode at once.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
            to keep the logs clean.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        """
        check_valid_model(model, HFModelType.EMBEDDING, token)
        self.model = model

        self.token = token
        token = token.resolve_value() if token else None

        self.prefix = prefix
        self.suffix = suffix
        self.normalize_embeddings = normalize_embeddings
        self.onnx_execution_provider = onnx_execution_provider
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

        model_kwargs = model_kwargs or {}

        # Check if the model_kwargs contain the parameters, otherwise, populate them with values from init parameters
        model_kwargs.setdefault("model_id", model)
        model_kwargs.setdefault("provider", onnx_execution_provider)
        model_kwargs.setdefault("use_auth_token", token)

        self.model_kwargs = model_kwargs
        self.embedding_model = None
        self.tokenizer = None

    def warm_up(self):
        """
        Convert the model to ONNX.

        The model is cached if the "TensorrtExecutionProvider" is used, since it takes a while to to build the TensorRT
        engine.
        """
        if self.embedding_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)

            if self.onnx_execution_provider == "TensorrtExecutionProvider":
                # Cache engine for TensorRT
                provider_options = {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": f"tmp/trt_cache_{self.model}",
                }
                self.embedding_model = ORTModelForFeatureExtraction.from_pretrained(
                    **self.model_kwargs, use_cache=False, provider_options=provider_options
                )
            else:
                # export=True converts the model to ONNX on the fly
                self.embedding_model = ORTModelForFeatureExtraction.from_pretrained(**self.model_kwargs, export=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        serialization_dict = default_to_dict(
            self,
            model=self.model,
            prefix=self.prefix,
            suffix=self.suffix,
            normalize_embeddings=self.normalize_embeddings,
            onnx_execution_provider=self.onnx_execution_provider,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            model_kwargs=self.model_kwargs,
            token=self.token.to_dict() if self.token else None,
        )

        model_kwargs = serialization_dict["init_parameters"]["model_kwargs"]
        model_kwargs.pop("token", None)

        serialize_hf_model_kwargs(model_kwargs)
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimumDocumentEmbedder":
        """
        Deserialize this component from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        deserialize_hf_model_kwargs(data["init_parameters"]["model_kwargs"])
        return default_from_dict(cls, data)

    def mean_pooling(self, model_output: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        """
        Perform Mean Pooling on the output of the Embedding model.

        :param model_output: The output of the embedding model.
        :param attention_mask: The attention mask of the tokenized text.
        :return: The embeddings of the text after mean pooling.
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]

            text_to_embed = (
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            )

            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> List[List[float]]:
        """
        Embed a list of texts in batches.
        """
        # Determine device for tokenizer output
        device = (
            "cuda"
            if self.onnx_execution_provider
            in ["CUDAExecutionProvider", "ROCMExecutionProvider", "TensorrtExecutionProvider"]
            else "cpu"
        )

        # Sorting by length
        length_sorted_idx = np.argsort([-len(sen) for sen in texts_to_embed])
        sentences_sorted = [texts_to_embed[idx] for idx in length_sorted_idx]

        all_embeddings = []
        for i in tqdm(
            range(0, len(sentences_sorted), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = sentences_sorted[i : i + batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)  # type: ignore

            # Compute token embeddings
            with torch.no_grad():
                model_output = self.embedding_model(**encoded_input)  # type: ignore

            # Perform mean pooling
            sentence_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"].cpu())

            all_embeddings.extend(sentence_embeddings.tolist())

        # Reorder embeddings according to original order
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        # Normalize all embeddings
        if self.normalize_embeddings:
            all_embeddings = torch.nn.functional.normalize(torch.tensor(all_embeddings), p=2, dim=1).tolist()

        return all_embeddings

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.
        The embedding of each Document is stored in the `embedding` field of the Document.

        :param documents: A list of Documents to embed.
        :return: A dictionary containing the updated Documents with their embeddings.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            msg = (
                "OptimumDocumentEmbedder expects a list of Documents as input."
                " In case you want to embed a string, please use the OptimumTextEmbedder."
            )
            raise TypeError(msg)

        if not (self.embedding_model and self.tokenizer):
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        # Return empty list if no documents
        if not documents:
            return {"documents": []}

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents}
