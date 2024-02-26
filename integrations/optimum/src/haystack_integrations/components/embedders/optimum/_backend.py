import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.hf import HFModelType, check_valid_model, deserialize_hf_model_kwargs, serialize_hf_model_kwargs
from huggingface_hub import hf_hub_download
from sentence_transformers.models import Pooling as SentenceTransformerPoolingLayer
from tqdm import tqdm
from transformers import AutoTokenizer

from optimum.onnxruntime import ORTModelForFeatureExtraction

from .pooling import OptimumEmbedderPooling


@dataclass
class _EmbedderParams:
    model: str
    token: Optional[Secret]
    prefix: str
    suffix: str
    normalize_embeddings: bool
    onnx_execution_provider: str
    batch_size: int
    progress_bar: bool
    pooling_mode: Optional[Union[str, OptimumEmbedderPooling]]
    model_kwargs: Optional[Dict[str, Any]]

    def serialize(self) -> Dict[str, Any]:
        out = {}
        for field in self.__dataclass_fields__.keys():
            out[field] = copy.deepcopy(getattr(self, field))

        # Fixups.
        assert isinstance(self.pooling_mode, OptimumEmbedderPooling)
        out["pooling_mode"] = self.pooling_mode.value
        out["token"] = self.token.to_dict() if self.token else None
        out["model_kwargs"].pop("use_auth_token", None)
        serialize_hf_model_kwargs(out["model_kwargs"])
        return out

    @classmethod
    def deserialize_inplace(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        data["pooling_mode"] = OptimumEmbedderPooling.from_str(data["pooling_mode"])
        deserialize_secrets_inplace(data, keys=["token"])
        deserialize_hf_model_kwargs(data["model_kwargs"])
        return data


class _EmbedderBackend:
    def __init__(self, params: _EmbedderParams):
        check_valid_model(params.model, HFModelType.EMBEDDING, params.token)
        resolved_token = params.token.resolve_value() if params.token else None

        if isinstance(params.pooling_mode, str):
            params.pooling_mode = OptimumEmbedderPooling.from_str(params.pooling_mode)
        elif params.pooling_mode is None:
            params.pooling_mode = _pooling_from_model_config(params.model, resolved_token)

        if params.pooling_mode is None:
            modes = {e.value: e for e in OptimumEmbedderPooling}
            msg = (
                f"Pooling mode not found in model config and not specified by user."
                f" Supported modes are: {list(modes.keys())}"
            )
            raise ValueError(msg)

        params.model_kwargs = params.model_kwargs or {}

        # Check if the model_kwargs contain the parameters, otherwise, populate them with values from init parameters
        params.model_kwargs.setdefault("model_id", params.model)
        params.model_kwargs.setdefault("provider", params.onnx_execution_provider)
        params.model_kwargs.setdefault("use_auth_token", resolved_token)

        self.params = params
        self.model = None
        self.tokenizer = None
        self.pooling_layer = None

    def warm_up(self):
        self.model = ORTModelForFeatureExtraction.from_pretrained(**self.params.model_kwargs, export=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.params.model, token=self.params.token.resolve_value() if self.params.token else None
        )

        # We need the width of the embeddings to initialize the pooling layer
        # so we do a dummy forward pass with the model.
        dummy_input = self.tokenizer(["dummy input"], padding=True, truncation=True, return_tensors="pt").to(
            self.model.device
        )
        dummy_output = self.model(input_ids=dummy_input["input_ids"], attention_mask=dummy_input["attention_mask"])
        width = dummy_output[0].size(dim=2)  # BaseModelOutput.last_hidden_state

        self.pooling_layer = SentenceTransformerPoolingLayer(
            width,
            pooling_mode_cls_token=self.params.pooling_mode == OptimumEmbedderPooling.CLS,
            pooling_mode_max_tokens=self.params.pooling_mode == OptimumEmbedderPooling.MAX,
            pooling_mode_mean_tokens=self.params.pooling_mode == OptimumEmbedderPooling.MEAN,
            pooling_mode_mean_sqrt_len_tokens=self.params.pooling_mode == OptimumEmbedderPooling.MEAN_SQRT_LEN,
            pooling_mode_weightedmean_tokens=self.params.pooling_mode == OptimumEmbedderPooling.WEIGHTED_MEAN,
            pooling_mode_lasttoken=self.params.pooling_mode == OptimumEmbedderPooling.LAST_TOKEN,
        )

    @property
    def parameters(self) -> _EmbedderParams:
        return self.params

    def pool_embeddings(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        assert self.pooling_layer is not None
        features = {"token_embeddings": model_output, "attention_mask": attention_mask}
        pooled_outputs = self.pooling_layer.forward(features)
        return pooled_outputs["sentence_embedding"]

    def embed_texts(
        self,
        texts_to_embed: Union[str, List[str]],
    ) -> Union[List[List[float]], List[float]]:
        assert self.model is not None
        assert self.tokenizer is not None

        if isinstance(texts_to_embed, str):
            texts = [texts_to_embed]
        else:
            texts = texts_to_embed

        device = self.model.device

        # Sorting by length
        length_sorted_idx = np.argsort([-len(sen) for sen in texts])
        sentences_sorted = [texts[idx] for idx in length_sorted_idx]

        all_embeddings = []
        for i in tqdm(
            range(0, len(sentences_sorted), self.params.batch_size),
            disable=not self.params.progress_bar,
            desc="Calculating embeddings",
        ):
            batch = sentences_sorted[i : i + self.params.batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            model_output = self.model(
                input_ids=encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"]
            )
            sentence_embeddings = self.pool_embeddings(model_output[0], encoded_input["attention_mask"].to(device))
            all_embeddings.append(sentence_embeddings)

        embeddings = torch.cat(all_embeddings, dim=0)

        if self.params.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        embeddings = embeddings.tolist()

        # Reorder embeddings according to original order
        reordered_embeddings: List[List[float]] = [None] * len(texts)  # type: ignore
        for embedding, idx in zip(embeddings, length_sorted_idx):
            reordered_embeddings[idx] = embedding

        if isinstance(texts_to_embed, str):
            return reordered_embeddings[0]
        else:
            return reordered_embeddings


def _pooling_from_model_config(model: str, token: Optional[str] = None) -> Optional[OptimumEmbedderPooling]:
    try:
        pooling_config_path = hf_hub_download(repo_id=model, token=token, filename="1_Pooling/config.json")
    except Exception as e:
        msg = f"An error occurred while downloading the model config: {e}"
        raise ValueError(msg) from e

    with open(pooling_config_path) as f:
        pooling_config = json.load(f)

    # Filter only those keys that start with "pooling_mode" and are True
    true_pooling_modes = [key for key, value in pooling_config.items() if key.startswith("pooling_mode") and value]

    # If exactly one True pooling mode is found, return it
    # If no True pooling modes or more than one True pooling mode is found, return None
    if len(true_pooling_modes) == 1:
        pooling_mode_from_config = true_pooling_modes[0]
        pooling_mode = _POOLING_MODES_MAP.get(pooling_mode_from_config)
    else:
        pooling_mode = None
    return pooling_mode


_POOLING_MODES_MAP = {
    "pooling_mode_cls_token": OptimumEmbedderPooling.CLS,
    "pooling_mode_mean_tokens": OptimumEmbedderPooling.MEAN,
    "pooling_mode_max_tokens": OptimumEmbedderPooling.MAX,
    "pooling_mode_mean_sqrt_len_tokens": OptimumEmbedderPooling.MEAN_SQRT_LEN,
    "pooling_mode_weightedmean_tokens": OptimumEmbedderPooling.WEIGHTED_MEAN,
    "pooling_mode_lasttoken": OptimumEmbedderPooling.LAST_TOKEN,
}
