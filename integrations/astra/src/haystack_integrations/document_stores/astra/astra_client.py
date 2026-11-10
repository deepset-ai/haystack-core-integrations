# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from astrapy import DataAPIClient
from astrapy.api_options import APIOptions, SerdesOptions
from astrapy.info import CollectionDescriptor
from haystack import Document, logging
from haystack.version import __version__ as integration_version

logger = logging.getLogger(__name__)

NON_INDEXED_FIELDS = ["metadata._node_content", "content"]
CALLER_NAME = "haystack"
# Preserve embedding precision and plain Python types when reading with AstraPy 2.
_API_OPTIONS = APIOptions(serdes_options=SerdesOptions(binary_encode_vectors=False, custom_datatypes_in_reading=False))


def _data_api_client() -> DataAPIClient:
    return DataAPIClient(callers=[(CALLER_NAME, integration_version)], api_options=_API_OPTIONS)


def _collection_definition(embedding_dimension: int, similarity: str) -> dict[str, Any]:
    return {
        "vector": {"dimension": embedding_dimension, "metric": similarity},
        "indexing": {"deny": NON_INDEXED_FIELDS},
    }


def _find_collection(collection_name: str, collections: list[CollectionDescriptor]) -> CollectionDescriptor | None:
    return next((descriptor for descriptor in collections if descriptor.name == collection_name), None)


def _collection_indexing_warning(descriptor: CollectionDescriptor) -> str | None:
    indexing = descriptor.definition.indexing or {}
    if not indexing:
        return (
            f"Collection '{descriptor.name}' is detected as having indexing turned on for all fields "
            "(either created manually or by older versions of this plugin). This implies stricter "
            "limitations on the amount of text each entry can store. Consider indexing anew on a "
            "fresh collection to be able to store longer texts."
        )
    if indexing != {"deny": NON_INDEXED_FIELDS}:
        return (
            f"Collection '{descriptor.name}' has unexpected 'indexing' settings "
            f"(options.indexing = {json.dumps(indexing)}). This can result in odd behaviour when running "
            "metadata filtering and/or unwarranted limitations on storing long texts. "
            "Consider indexing anew on a fresh collection."
        )
    return None


def _find_kwargs(
    filters: dict[str, Any] | None, *, vector: list[float] | None = None, limit: int | None = None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"filter": filters, "limit": limit, "projection": {"*": 1}}
    if vector is not None:
        kwargs |= {"sort": {"$vector": vector}, "include_similarity": True}
    return kwargs


def _to_documents(responses: list[dict[str, Any]]) -> list[Document]:
    if not responses:
        logger.warning("No documents found.")
    documents = []
    for response in responses:
        fields = dict(response)
        documents.append(
            Document(
                id=fields.pop("_id"),
                content=fields.pop("content", None),
                embedding=fields.pop("$vector", None),
                blob=fields.pop("blob", None),
                meta=fields.pop("meta", {}),
                score=fields.pop("$similarity", None),
            )
        )
    return documents
