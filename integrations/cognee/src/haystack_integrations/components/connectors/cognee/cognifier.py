# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import cognee
from haystack import component, default_from_dict, default_to_dict, logging

from ._utils import run_sync

logger = logging.getLogger(__name__)


@component
class CogneeCognifier:
    """
    Processes previously added data in Cognee's knowledge engine.

    Wraps `cognee.cognify()` as a standalone pipeline step, useful when you
    want to separate the add and cognify phases (e.g., batch-add first, then
    cognify once).

    Usage:
    ```python
    from haystack_integrations.components.connectors.cognee import CogneeCognifier

    cognifier = CogneeCognifier()
    result = cognifier.run()
    assert result["cognified"] is True
    ```
    """

    @component.output_types(cognified=bool)
    def run(self) -> dict[str, Any]:
        """
        Run cognee.cognify() to process added data into the memory.

        :returns: Dictionary with key `cognified` set to True on success.
        """
        logger.info("Running cognee.cognify()")
        run_sync(cognee.cognify())
        return {"cognified": True}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CogneeCognifier":
        return default_from_dict(cls, data)
