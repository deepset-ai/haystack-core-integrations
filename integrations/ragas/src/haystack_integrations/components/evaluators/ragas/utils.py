# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import importlib
from typing import Any

from openai import AsyncOpenAI

from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.base import SimpleBaseMetric


def _serialize_metric(metric: SimpleBaseMetric) -> dict[str, Any]:
    """
    Serialize a `SimpleBaseMetric` to a JSON-compatible dict.

    Stores the class path, metric name, and — when present — the LLM and
    embeddings configuration (provider and model name).

    :param metric: The metric instance to serialize.
    :returns: A dict suitable for storage in a pipeline YAML or `to_dict` output.
    """
    metric_cls = type(metric)
    serialized: dict[str, Any] = {
        "type": f"{metric_cls.__module__}.{metric_cls.__qualname__}",
        "name": metric.name,
    }
    llm = getattr(metric, "llm", None)
    if llm is not None:
        serialized["llm"] = {"model": llm.model, "provider": llm.provider}
    embeddings = getattr(metric, "embeddings", None)
    if embeddings is not None:
        serialized["embeddings"] = {"model": embeddings.model, "provider": embeddings.PROVIDER_NAME}
    return serialized


def _deserialize_metric(data: dict[str, Any]) -> SimpleBaseMetric:
    """
    Reconstruct a `SimpleBaseMetric` from a serialized dict.

    Imports the metric class from the stored `type` path and rebuilds any LLM
    or embeddings using the stored provider and model name. Only the `openai`
    provider is supported for automatic reconstruction; the API key is read from
    the `OPENAI_API_KEY` environment variable at deserialization time.

    :param data: Dict produced by `_serialize_metric`.
    :returns: A fully constructed `SimpleBaseMetric` instance.
    :raises ValueError: If a non-`openai` provider is encountered.
    """
    type_path = data["type"]
    module_path, class_name = type_path.rsplit(".", 1)
    metric_cls = getattr(importlib.import_module(module_path), class_name)

    kwargs: dict[str, Any] = {}

    if "llm" in data:
        llm_data = data["llm"]
        if llm_data["provider"] != "openai":
            msg = f"Automatic deserialization only supports the 'openai' provider; got '{llm_data['provider']}'."
            raise ValueError(msg)
        kwargs["llm"] = llm_factory(llm_data["model"], client=AsyncOpenAI())

    if "embeddings" in data:
        emb_data = data["embeddings"]
        if emb_data["provider"] != "openai":
            msg = f"Automatic deserialization only supports the 'openai' provider; got '{emb_data['provider']}'."
            raise ValueError(msg)
        kwargs["embeddings"] = embedding_factory("openai", model=emb_data["model"], client=AsyncOpenAI())

    if "name" in data:
        kwargs["name"] = data["name"]

    return metric_cls(**kwargs)
