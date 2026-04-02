# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging

import cognee  # type: ignore[import-untyped]
from haystack_integrations.components.connectors.cognee._utils import run_sync

logger = logging.getLogger(__name__)


@component
class CogneeWriter:
    """
    Adds Haystack Documents to Cognee's memory.

    Wraps `cognee.add()` and optionally `cognee.cognify()` to ingest documents
    and build a knowledge engine in a single pipeline step.

    Usage:
    ```python
    from haystack import Document
    from haystack_integrations.components.writers.cognee import CogneeWriter

    writer = CogneeWriter(dataset_name="my_dataset", auto_cognify=True)
    writer.run(documents=[Document(content="Cognee builds AI memory.")])
    ```
    """

    def __init__(self, *, dataset_name: str = "haystack", auto_cognify: bool = True):
        """
        Initialize the CogneeWriter.

        :param dataset_name: Name of the Cognee dataset to add documents to.
        :param auto_cognify: If True, automatically runs `cognee.cognify()` after adding
            documents to process them into the knowledge engine.
        """
        self.dataset_name = dataset_name
        self.auto_cognify = auto_cognify

    @component.output_types(documents_written=int)
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Add documents to Cognee and optionally cognify them.

        :param documents: List of Haystack Documents to add.
        :returns: Dictionary with key `documents_written` indicating how many
            documents were successfully added.
        """
        texts = [doc.content for doc in documents if doc.content]
        skipped = len(documents) - len(texts)
        if skipped > 0:
            logger.warning("Skipping {count} document(s) with empty content", count=skipped)

        if texts:
            run_sync(cognee.add(texts, dataset_name=self.dataset_name))

        written = len(texts)

        if self.auto_cognify and written > 0:
            logger.info(
                "Cognifying {count} documents in dataset '{dataset}'",
                count=written,
                dataset=self.dataset_name,
            )
            run_sync(cognee.cognify(datasets=[self.dataset_name]))

        return {"documents_written": written}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            dataset_name=self.dataset_name,
            auto_cognify=self.auto_cognify,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CogneeWriter":
        """Deserialize a component from a dictionary."""
        return default_from_dict(cls, data)
