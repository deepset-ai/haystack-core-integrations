import json
from enum import Enum
from typing import Optional

import torch
from haystack.utils import Secret
from huggingface_hub import hf_hub_download
from sentence_transformers.models import Pooling as PoolingLayer


class PoolingMode(Enum):
    """
    Pooling Modes support by the Optimum Embedders.
    """

    CLS = "cls"
    MEAN = "mean"
    MAX = "max"
    MEAN_SQRT_LEN = "mean_sqrt_len"
    WEIGHTED_MEAN = "weighted_mean"
    LAST_TOKEN = "last_token"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "PoolingMode":
        """
        Create a pooling mode from a string.

        :param string:
            The string to convert.
        :returns:
            The pooling mode.
        """
        enum_map = {e.value: e for e in PoolingMode}
        pooling_mode = enum_map.get(string)
        if pooling_mode is None:
            msg = f"Unknown Pooling mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return pooling_mode


POOLING_MODES_MAP = {
    "pooling_mode_cls_token": PoolingMode.CLS,
    "pooling_mode_mean_tokens": PoolingMode.MEAN,
    "pooling_mode_max_tokens": PoolingMode.MAX,
    "pooling_mode_mean_sqrt_len_tokens": PoolingMode.MEAN_SQRT_LEN,
    "pooling_mode_weightedmean_tokens": PoolingMode.WEIGHTED_MEAN,
    "pooling_mode_lasttoken": PoolingMode.LAST_TOKEN,
}

INVERSE_POOLING_MODES_MAP = {mode: name for name, mode in POOLING_MODES_MAP.items()}


class HFPoolingMode:
    """
    Gets the pooling mode of the model from the Hugging Face Hub.
    """

    @staticmethod
    def get_pooling_mode(model: str, token: Optional[Secret] = None) -> Optional[PoolingMode]:
        """
        Gets the pooling mode of the model from the Hugging Face Hub.

        :param model:
            The model to get the pooling mode for.
        :param token:
            The HuggingFace token to use as HTTP bearer authorization.
        :returns:
            The pooling mode.
        """
        try:

            with open(pooling_config_path) as f:
                pooling_config = json.load(f)

            # Filter only those keys that start with "pooling_mode" and are True
            true_pooling_modes = [
                key for key, value in pooling_config.items() if key.startswith("pooling_mode") and value
            ]

            # If exactly one True pooling mode is found, return it
            if len(true_pooling_modes) == 1:
                pooling_mode_from_config = true_pooling_modes[0]
                pooling_mode = POOLING_MODES_MAP.get(pooling_mode_from_config)
            # If no True pooling modes or more than one True pooling mode is found, return None
            else:
                pooling_mode = None
            return pooling_mode
        except Exception as e:
            msg = f"An error occurred while inferring the pooling mode from the model config: {e}"
            raise ValueError(msg) from e


class Pooling:
    """
    Class to manage pooling of the embeddings.

    :param pooling_mode: The pooling mode to use.
    :param attention_mask: The attention mask of the tokenized text.
    :param model_output: The output of the embedding model.
    """

    def __init__(self, pooling_mode: PoolingMode, attention_mask: torch.Tensor, model_output: torch.Tensor):
        self.pooling_mode = pooling_mode
        self.attention_mask = attention_mask
        self.model_output = model_output

    def pool_embeddings(self) -> torch.Tensor:
        """
        Perform pooling on the output of the embedding model.

        :param pooling_mode: The pooling mode to use.
        :param attention_mask: The attention mask of the tokenized text.
        :param model_output: The output of the embedding model.
        :return: The embeddings of the text after pooling.
        """
        pooling_func_map = {
            INVERSE_POOLING_MODES_MAP[self.pooling_mode]: True,
        }
        # By default, sentence-transformers uses mean pooling
        # If multiple pooling methods are specified, the output dimension of the embeddings is scaled by the number of
        # pooling methods selected
        if self.pooling_mode != PoolingMode.MEAN:
            pooling_func_map[INVERSE_POOLING_MODES_MAP[PoolingMode.MEAN]] = False

        # First element of model_output contains all token embeddings
        token_embeddings = self.model_output[0]
        word_embedding_dimension = token_embeddings.size(dim=2)
        pooling = PoolingLayer(word_embedding_dimension=word_embedding_dimension, **pooling_func_map)
        features = {"token_embeddings": token_embeddings, "attention_mask": self.attention_mask}
        pooled_outputs = pooling.forward(features)
        embeddings = pooled_outputs["sentence_embedding"]

        return embeddings
