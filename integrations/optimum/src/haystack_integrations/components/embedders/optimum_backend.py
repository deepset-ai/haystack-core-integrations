from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from haystack.utils.auth import Secret
from haystack_integrations.components.embedders.pooling import Pooling, PoolingMode
from optimum.onnxruntime import ORTModelForFeatureExtraction
from tqdm import tqdm
from transformers import AutoTokenizer


class OptimumEmbeddingBackend:
    """
    Class to manage Optimum embeddings.
    """

    def __init__(self, model: str, model_kwargs: Dict[str, Any], token: Optional[Secret] = None):
        """
        Create an instance of OptimumEmbeddingBackend.

        :param model: A string representing the model id on HF Hub.
        :param model_kwargs: Keyword arguments to pass to the model.
        :param token: The HuggingFace token to use as HTTP bearer authorization.
        """
        # export=True converts the model to ONNX on the fly
        self.model = ORTModelForFeatureExtraction.from_pretrained(**model_kwargs, export=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=token)

    def embed(
        self,
        texts_to_embed: Union[str, List[str]],
        normalize_embeddings: bool,
        pooling_mode: PoolingMode = PoolingMode.MEAN,
        progress_bar: bool = False,
        batch_size: int = 1,
    ) -> Union[List[List[float]], List[float]]:
        """
        Embed text or list of texts using the Optimum model.

        :param texts_to_embed: The text or list of texts to embed.
        :param normalize_embeddings: Whether to normalize the embeddings to unit length.
        :param pooling_mode: The pooling mode to use.
        :param progress_bar: Whether to show a progress bar or not.
        :param batch_size: Batch size to use.
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
            model_output = self.model(**encoded_input)

            # Pool Embeddings
            pooling = Pooling(
                pooling_mode=pooling_mode,
                attention_mask=encoded_input["attention_mask"].to(device),
                model_output=model_output,
            )
            sentence_embeddings = pooling.pool_embeddings()
            all_embeddings.append(sentence_embeddings)

        embeddings = torch.cat(all_embeddings, dim=0)

        # Normalize all embeddings
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        embeddings = embeddings.tolist()

        # Reorder embeddings according to original order
        reordered_embeddings: List[List[float]] = [[]] * len(texts)
        for embedding, idx in zip(embeddings, length_sorted_idx):
            reordered_embeddings[idx] = embedding

        if isinstance(texts_to_embed, str):
            # Return the embedding if only one text was passed
            return reordered_embeddings[0]

        return reordered_embeddings
