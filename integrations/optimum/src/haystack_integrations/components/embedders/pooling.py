import json
from enum import Enum
from typing import Optional

import torch
from haystack.utils import Secret
from huggingface_hub import hf_hub_download


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


class HFPoolingMode:
    """
    Gets the pooling mode of the Sentence Transformer model from the Hugging Face Hub.
    """

    @staticmethod
    def get_pooling_mode(model: str, token: Optional[Secret] = None) -> Optional[PoolingMode]:
        try:
            pooling_config_path = hf_hub_download(repo_id=model, token=token, filename="1_Pooling/config.json")

            with open(pooling_config_path) as f:
                pooling_config = json.load(f)

            # Filter only those keys that start with "pooling_mode" and are True
            true_pooling_modes = [
                key for key, value in pooling_config.items() if key.startswith("pooling_mode") and value
            ]

            pooling_modes_map = {
                "pooling_mode_cls_token": PoolingMode.CLS,
                "pooling_mode_mean_tokens": PoolingMode.MEAN,
                "pooling_mode_max_tokens": PoolingMode.MAX,
                "pooling_mode_mean_sqrt_len_tokens": PoolingMode.MEAN_SQRT_LEN,
                "pooling_mode_weightedmean_tokens": PoolingMode.WEIGHTED_MEAN,
                "pooling_mode_last_token": PoolingMode.LAST_TOKEN,
            }

            # If exactly one True pooling mode is found, return it
            if len(true_pooling_modes) == 1:
                pooling_mode_from_config = true_pooling_modes[0]
                pooling_mode = pooling_modes_map.get(pooling_mode_from_config)
            # If no True pooling modes or more than one True pooling mode is found, return None
            else:
                pooling_mode = None
            return pooling_mode
        except Exception:
            return None


class Pooling:
    """
    Class to manage pooling of the embeddings.

    :param pooling_mode: The pooling mode to use.
    :param attention_mask: The attention mask of the tokenized text.
    :param model_output: The output of the embedding model.
    """

    def __init__(self, pooling_mode: PoolingMode, attention_mask: torch.tensor, model_output: torch.tensor):
        self.pooling_mode = pooling_mode
        self.attention_mask = attention_mask
        self.model_output = model_output

    def _cls_pooling(self, token_embeddings: torch.tensor) -> torch.tensor:
        """
        Perform CLS Pooling on the output of the embedding model. Uses the first token (CLS token) as text
        representations.

        :param model_output: The output of the embedding model.
        :param attention_mask: The attention mask of the tokenized text.
        :return: The embeddings of the text after mean pooling.
        """
        embeddings = token_embeddings[:, 0]
        return embeddings

    def _max_pooling(self, token_embeddings: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        """
        Perform Max Pooling on the output of the embedding model. Uses max in each dimension over all tokens.

        :param model_output: The output of the embedding model.
        :param attention_mask: The attention mask of the tokenized text.
        :return: The embeddings of the text after mean pooling.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Set padding tokens to large negative value
        token_embeddings[input_mask_expanded == 0] = -1e9
        embeddings = torch.max(token_embeddings, 1)[0]
        return embeddings

    def _mean_pooling(self, token_embeddings: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        """
        Perform Mean Pooling on the output of the embedding model.

        :param model_output: The output of the embedding model.
        :param attention_mask: The attention mask of the tokenized text.
        :return: The embeddings of the text after mean pooling.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _mean_sqrt_len_pooling(self, token_embeddings: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        """
        Perform mean-pooling on the output of the embedding model, but divide by sqrt(input_length).

        :param model_output: The output of the embedding model.
        :param attention_mask: The attention mask of the tokenized text.
        :return: The embeddings of the text after mean pooling.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / torch.sqrt(sum_mask)

    def _weighted_mean_pooling(self, token_embeddings: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        """
        Perform Weighted (position) Mean Pooling on the output of the embedding model.
        See https://arxiv.org/abs/2202.08904.

        :param model_output: The output of the embedding model.
        :param attention_mask: The attention mask of the tokenized text.
        :return: The embeddings of the text after mean pooling.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # token_embeddings shape: bs, seq, hidden_dim
        weights = (
            torch.arange(start=1, end=token_embeddings.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
            .to(token_embeddings.device)
        )
        input_mask_expanded = input_mask_expanded * weights
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _last_token_pooling(self, token_embeddings: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        """
        Perform Last Token Pooling on the output of the embedding model. See https://arxiv.org/abs/2202.08904 &
        https://arxiv.org/abs/2201.10005.

        :param model_output: The output of the embedding model.
        :param attention_mask: The attention mask of the tokenized text.
        :return: The embeddings of the text after mean pooling.
        """
        bs, seq_len, hidden_dim = token_embeddings.shape
        # attention_mask shape: (bs, seq_len)
        # Get shape [bs] indices of the last token (i.e. the last token for each batch item)
        # argmin gives us the index of the first 0 in the attention mask; We get the last 1 index by subtracting 1
        # Any sequence where min == 1, we use the entire sequence length since argmin = 0
        values, indices = torch.min(attention_mask, 1, keepdim=False)
        gather_indices = torch.where(values == 0, indices, seq_len) - 1  # Shape [bs]
        # There are empty sequences, where the index would become -1 which will crash
        gather_indices = torch.clamp(gather_indices, min=0)

        # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
        gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
        gather_indices = gather_indices.unsqueeze(1)

        # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
        # Actually no need for the attention mask as we gather the last token where attn_mask = 1
        # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
        # use the attention mask to ignore them again
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.gather(token_embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        return embeddings

    def pool_embeddings(self) -> torch.tensor:
        """
        Perform pooling on the output of the embedding model.

        :param pooling_mode: The pooling mode to use.
        :param attention_mask: The attention mask of the tokenized text.
        :param model_output: The output of the embedding model.
        :return: The embeddings of the text after pooling.
        """
        pooling_func_map = {
            PoolingMode.CLS: self._cls_pooling,
            PoolingMode.MEAN: self._mean_pooling,
            PoolingMode.MAX: self._max_pooling,
            PoolingMode.MEAN_SQRT_LEN: self._mean_sqrt_len_pooling,
            PoolingMode.WEIGHTED_MEAN: self._weighted_mean_pooling,
            PoolingMode.LAST_TOKEN: self._last_token_pooling,
        }
        self._pooling_function = pooling_func_map[self.pooling_mode]

        # First element of model_output contains all token embeddings
        token_embeddings = self.model_output[0]

        embeddings = (
            self._pooling_function(token_embeddings, self.attention_mask)  # type: ignore
            if self._pooling_function != self._cls_pooling
            else self._pooling_function(token_embeddings)  # type: ignore
        )

        return embeddings
