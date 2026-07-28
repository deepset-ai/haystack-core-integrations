# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from haystack.core.serialization import import_class_by_name
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

    The metric class is imported through Haystack's gated `import_class_by_name`, so with
    `haystack-ai` >= 3.0 the module it lives in must be on the deserialization allowlist. Metrics
    from `ragas` itself or from your own package therefore need to be trusted explicitly, e.g. via
    `Pipeline.load(..., allowed_modules=["ragas.*"])`, `allow_deserialization_module("ragas.*")` or
    the `HAYSTACK_DESERIALIZATION_ALLOWLIST` environment variable.

    :param data: Dict produced by `_serialize_metric`.
    :returns: A fully constructed `SimpleBaseMetric` instance.
    :raises ValueError: If a non-`openai` provider is encountered.
    :raises DeserializationError: If the metric class is not on the deserialization allowlist.
    """
    type_path = data["type"]
    # `import_class_by_name` returns `type[object]`; annotate as `Any` so that calling it with the
    # metric's own keyword arguments below type-checks.
    metric_cls: Any = import_class_by_name(type_path)

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
