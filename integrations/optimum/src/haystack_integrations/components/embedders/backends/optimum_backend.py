from typing import Any, ClassVar, Dict, List, Optional, Union

import numpy as np
import torch
from haystack.utils.auth import Secret
from optimum.onnxruntime import ORTModelForFeatureExtraction
from tqdm import tqdm
from transformers import AutoTokenizer


class _OptimumEmbeddingBackendFactory:
    """
    Factory class to create instances of Sentence Transformers embedding backends.
    """

    _instances: ClassVar[Dict[str, "_OptimumEmbeddingBackend"]] = {}

    @staticmethod
    def get_embedding_backend(
        model: str, token: Optional[Secret] = None, model_kwargs: Optional[Dict[str, Any]] = None
    ):
        embedding_backend_id = f"{model}{token}"

        if embedding_backend_id in _OptimumEmbeddingBackendFactory._instances:
            return _OptimumEmbeddingBackendFactory._instances[embedding_backend_id]
        embedding_backend = _OptimumEmbeddingBackend(model=model, token=token, model_kwargs=model_kwargs)
        _OptimumEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _OptimumEmbeddingBackend:
    """
    Class to manage Optimum embeddings.
    """

    def __init__(self, model: str, token: Optional[Secret] = None, model_kwargs: Optional[Dict[str, Any]] = None):
        # export=True converts the model to ONNX on the fly
        self.model = ORTModelForFeatureExtraction.from_pretrained(**model_kwargs, export=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=token)

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

    def embed(
        self,
        texts_to_embed: Union[str, List[str]],
        normalize_embeddings: bool,
        progress_bar: bool = False,
        batch_size: int = 1,
    ) -> Union[List[List[float]], List[float]]:
        """
        Embed text or list of texts using the Optimum model.

        :param texts_to_embed: T
        :param normalize_embeddings: Whether to normalize the embeddings to unit length.
        :param progress_bar: Whether to show a progress bar or not, defaults to False.
        :param batch_size: Batch size to use, defaults to 1.
        :return: A single embedding if the input is a single string. A list of embeddings if the input is a list of
            strings.
        """
        if isinstance(texts_to_embed, str):
            texts = [texts_to_embed]
        else:
            texts = texts_to_embed

        # Determine device for tokenizer output
        device = self.model.device

        # Sorting by length
        length_sorted_idx = np.argsort([-len(sen) for sen in texts])
        sentences_sorted = [texts[idx] for idx in length_sorted_idx]

        all_embeddings = []
        for i in tqdm(
            range(0, len(sentences_sorted), batch_size), disable=not progress_bar, desc="Calculating embeddings"
        ):
            batch = sentences_sorted[i : i + batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

            # Only pass required inputs otherwise onnxruntime can raise an error
            inputs_to_remove = set(encoded_input.keys()).difference(self.model.inputs_names)
            for key in inputs_to_remove:
                encoded_input.pop(key)

            # Compute token embeddings
            model_output = self.model(**encoded_input)

            # Perform mean pooling
            sentence_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"].to(device))

            all_embeddings.extend(sentence_embeddings.tolist())

        # Reorder embeddings according to original order
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        # Normalize all embeddings
        if normalize_embeddings:
            all_embeddings = torch.nn.functional.normalize(torch.tensor(all_embeddings), p=2, dim=1).tolist()

        if isinstance(texts_to_embed, str):
            # Return the embedding if only one text was passed
            all_embeddings = all_embeddings[0]

        return all_embeddings
