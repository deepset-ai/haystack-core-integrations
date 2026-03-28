# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging

import cognee  # type: ignore[import-untyped]

from ._utils import run_sync

logger = logging.getLogger(__name__)


@component
class CogneeCognifier:
    """
    Processes previously added data through Cognee's knowledge engine.

    Wraps `cognee.cognify()` as a standalone pipeline step. Cognify takes raw data
    that was previously added via `cognee.add()` and transforms it into a structured
    knowledge graph. The process includes:

    1. **Document classification** — identifies the type and structure of the input data.
    2. **Text chunking** — splits documents into semantically meaningful segments.
    3. **Entity extraction** — uses an LLM to identify entities and their properties.
    4. **Relationship detection** — discovers connections between extracted entities.
    5. **Graph construction** — builds a knowledge graph with embeddings for vector search.
    6. **Summarization** — generates hierarchical summaries of the processed content.

    After cognification, the data becomes searchable via `cognee.search()` using various
    strategies (graph traversal, vector similarity, summaries, etc.).

    This component is useful when you want to separate the add and cognify phases —
    for example, batch-add documents first with `CogneeWriter(auto_cognify=False)`,
    then cognify once.

    Usage:
    ```python
    from haystack import Pipeline
    from haystack_integrations.components.writers.cognee import CogneeWriter
    from haystack_integrations.components.connectors.cognee import CogneeCognifier

    pipeline = Pipeline()
    pipeline.add_component("writer", CogneeWriter(dataset_name="my_data", auto_cognify=False))
    pipeline.add_component("cognifier", CogneeCognifier(dataset_name="my_data"))
    pipeline.connect("writer.documents_written", "cognifier.documents_written")

    result = pipeline.run({"writer": {"documents": docs}})
    ```
    """

    def __init__(self, dataset_name: str | list[str] | None = None):
        """
        :param dataset_name: Optional Cognee dataset name(s) to cognify. Accepts a single
            name, a list of names, or None to cognify all pending datasets.
        """
        self.dataset_name = dataset_name

    @component.output_types(cognified=bool)
    def run(self, documents_written: int | None = None) -> dict[str, Any]:
        """
        Run cognee.cognify() to process added data into the knowledge graph.

        :param documents_written: Optional number of documents written by a preceding
            CogneeWriter. Used as a pipeline connection point; cognify runs regardless
            of the value, as long as data has been previously added.
        :returns: Dictionary with key `cognified` set to True on success.
        """
        cognify_kwargs: dict[str, Any] = {}
        if self.dataset_name:
            datasets = [self.dataset_name] if isinstance(self.dataset_name, str) else self.dataset_name
            cognify_kwargs["datasets"] = datasets

        logger.info("Running cognee.cognify()")
        run_sync(cognee.cognify(**cognify_kwargs))
        return {"cognified": True}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, dataset_name=self.dataset_name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CogneeCognifier":
        return default_from_dict(cls, data)
