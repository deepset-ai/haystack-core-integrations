# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.document import Document
from haystack.document_stores.protocol import DuplicatePolicy


class PgvectorDocumentStore:

    def __init__(
        self
    ):
        pass


    def count_documents(self) -> int:
        return 0

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:  # noqa: ARG002
        return []

    def write_documents(
        self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE  # noqa: ARG002
    ) -> int:
        return 0

    def delete_documents(self, document_ids: List[str]) -> None:  # noqa: ARG002
        return
