# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError

logger = logging.getLogger(__name__)


class FAISSDocumentStore:
    """
    A Document Store using FAISS for vector search and a simple JSON file for metadata storage.

    This Document Store is suitable for small to medium-sized datasets where simplicity is preferred over scalability.
    It supports basic persistence by saving the FAISS index to a `.faiss` file and documents to a `.json` file.
    """

    def __init__(
        self,
        index_path: str | None = None,
        index_string: str = "Flat",
        embedding_dim: int = 768,
    ):
        """
        Initializes the FAISSDocumentStore.

        :param index_path: Path to save/load the index and documents. If None, the store is in-memory only.
        :param index_string: The FAISS index factory string. Default is "Flat".
        :param embedding_dim: The dimension of the embeddings. Default is 768.
        :raises DocumentStoreError: If the FAISS index cannot be initialized.
        :raises ValueError: If `index_path` points to a missing `.faiss` file when loading persisted data.
        """
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.index_string = index_string

        # Initialize in-memory storage
        self.documents: dict[str, Document] = {}
        self.id_map: dict[int, str] = {}  # Map integer IDs (FAISS) to string IDs (Documents)
        self.inverse_id_map: dict[str, int] = {}  # Map string IDs to integer IDs
        self._next_id = 0

        # Initialize FAISS index
        self.index: faiss.Index | None = None
        if self.index_path and Path(self.index_path).with_suffix(".faiss").exists():
            self.load(self.index_path)
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Creates a new FAISS index."""
        try:
            # We use IndexIDMap to support add_with_ids
            base_index = faiss.index_factory(self.embedding_dim, self.index_string)
            self.index = faiss.IndexIDMap(base_index)
        except RuntimeError as e:
            msg = f"Could not create FAISS index with factory string '{self.index_string}': {e}"
            raise DocumentStoreError(msg) from e

    def _get_index_or_raise(self) -> Any:
        """Return the FAISS index or raise if it is unexpectedly missing."""
        if self.index is None:
            msg = "FAISS index has not been initialized."
            raise DocumentStoreError(msg)
        return self.index

    def count_documents(self) -> int:
        """
        Returns the number of documents in the store.
        """
        return len(self.documents)

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns documents that match the provided filters.

        :param filters: A dictionary of filters to apply.
        :return: A list of matching Documents.
        :raises FilterError: If the filter structure is invalid.
        """
        if not filters:
            return list(self.documents.values())

        filtered_docs = []
        for doc in self.documents.values():
            if self._matches_filters(doc, filters):
                filtered_docs.append(doc)
        return filtered_docs

    def _matches_filters(self, doc: Document, filters: dict[str, Any]) -> bool:
        """
        Checks if a document matches the given filters.

        Currently, supports simple equality and comparison checks.
        """
        return self._check_condition(doc, filters)

    @staticmethod
    def _get_doc_value(doc: Document, field: str) -> Any:
        """Helper to get value from doc, handling 'meta.' prefix."""
        if field == "content":
            return doc.content
        if field == "id":
            return doc.id
        if field.startswith("meta."):
            key = field[5:]
            return doc.meta.get(key)
        # Fallback: check top level attributes then meta
        if hasattr(doc, field):
            return getattr(doc, field)
        return doc.meta.get(field)

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int:
        """
        Writes documents to the store.

        :param documents: The list of documents to write.
        :param policy: The policy to handle duplicate documents.
        :return: The number of documents written.
        :raises ValueError: If `documents` is not an iterable of `Document` objects.
        :raises DuplicateDocumentError: If a duplicate document is found and `policy` is `DuplicatePolicy.FAIL`.
        :raises DocumentStoreError: If the FAISS index is unexpectedly unavailable when adding embeddings.
        """
        if not isinstance(documents, Iterable) or isinstance(documents, (str, bytes)):
            msg = "param 'documents' must contain an iterable of objects of type Document."
            raise ValueError(msg)

        if any(not isinstance(doc, Document) for doc in documents):
            msg = "param 'documents' must contain an iterable of objects of type Document."
            raise ValueError(msg)

        if not documents:
            return 0

        # Check for duplicates first if policy is FAIL
        if policy == DuplicatePolicy.FAIL:
            for doc in documents:
                if doc.id in self.documents:
                    msg = f"Document with id '{doc.id}' already exists."
                    raise DuplicateDocumentError(msg)

        # Process documents
        ids_to_add_to_index = []
        vectors_to_add = []

        docs_written = 0

        for doc in documents:
            if policy == DuplicatePolicy.SKIP and doc.id in self.documents:
                continue

            # Handle overwrite or new
            if doc.id in self.documents:
                # If overwriting, we need to remove the old vector from index first?
                # FAISS doesn't support easy update. We'd have to remove and add.
                # For MVP, let's implement remove then add for overwrite.
                self.delete_documents([doc.id])

            self.documents[doc.id] = doc

            if doc.embedding is not None:
                # Assign a new integer ID
                int_id = self._next_id
                self._next_id += 1

                self.id_map[int_id] = doc.id
                self.inverse_id_map[doc.id] = int_id

                ids_to_add_to_index.append(int_id)
                vectors_to_add.append(doc.embedding)

            docs_written += 1

        # Add to FAISS
        if vectors_to_add:
            vectors = np.array(vectors_to_add, dtype="float32")
            ids = np.array(ids_to_add_to_index, dtype="int64")
            index = self._get_index_or_raise()
            index.add_with_ids(vectors, ids)

        return docs_written

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents from the store.

        :raises DocumentStoreError: If the FAISS index is unexpectedly unavailable when removing embeddings.
        """
        if not document_ids:
            return

        ids_to_remove_from_index = []

        for doc_id in document_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]

                if doc_id in self.inverse_id_map:
                    int_id = self.inverse_id_map.pop(doc_id)
                    del self.id_map[int_id]
                    ids_to_remove_from_index.append(int_id)

        index = self._get_index_or_raise()
        if ids_to_remove_from_index and index.ntotal > 0:
            ids_array = np.array(ids_to_remove_from_index, dtype="int64")
            index.remove_ids(ids_array)

    def delete_all_documents(self) -> None:
        """
        Deletes all documents from the store.
        """
        self.documents = {}
        self.id_map = {}
        self.inverse_id_map = {}
        self._next_id = 0
        self._create_new_index()

    def search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[Document]:
        """
        Performs a vector search.

        :param query_embedding: The query embedding.
        :param top_k: The number of results to return.
        :param filters: Filters to apply.
        :return: A list of matching Documents.
        :raises FilterError: If the filter structure is invalid.
        """
        if not self.index or self.index.ntotal == 0:
            return []

        # Ensure embedding format
        query_vec = np.array([query_embedding], dtype="float32")

        # Search in FAISS
        # Valid strategy for pre-filtering vs post-filtering:
        # Since FAISS `IndexIDMap` doesn't support pre-filtering natively comfortably
        # without `RangeSearch` or specialized impls, we fetch more and filter post-retrieval.

        fetch_k = top_k
        if filters:
            fetch_k = min(self.index.ntotal, top_k * 10)  # Simple heuristic

        distances, indices = self.index.search(query_vec, fetch_k)

        results = []
        for dist, int_id in zip(distances[0], indices[0], strict=False):
            if int_id == -1:
                continue

            doc_id = self.id_map.get(int_id)
            if not doc_id or doc_id not in self.documents:
                continue

            doc = self.documents[doc_id]

            if filters and not self._matches_filters(doc, filters):
                continue

            # Build a new instance instead of mutating score in place.
            score = float(1 / (1 + dist)) if self.index_string == "Flat" else float(dist)
            result_doc = replace(doc, score=score)

            results.append(result_doc)

            if len(results) >= top_k:
                break

        return results

    def _check_condition(self, doc: Document, condition: dict[str, Any]) -> bool:
        if "operator" not in condition and "conditions" not in condition:
            # This might be a legacy or malformed filter from tests like test_missing_top_level_operator_key
            # The standard Haystack filter structure enforces keys.
            # On failure to parse standard structure, we should raise FilterError as per tests?
            # Actually, looking at the tests (e.g. TestFAISSDocumentStore.test_missing_top_level_operator_key),
            # they expect FilterError if "operator" is missing from a condition block.
            msg = "Filter condition missing 'operator'"
            raise FilterError(msg)

        operator = condition.get("operator", "==")

        if operator == "AND":
            if "conditions" not in condition:
                msg = "Missing 'conditions' for AND operator"
                raise FilterError(msg)
            return all(self._check_condition(doc, cond) for cond in condition.get("conditions", []))
        elif operator == "OR":
            if "conditions" not in condition:
                msg = "Missing 'conditions' for OR operator"
                raise FilterError(msg)
            return any(self._check_condition(doc, cond) for cond in condition.get("conditions", []))
        elif operator == "NOT":
            if "conditions" not in condition:
                msg = "Missing 'conditions' for NOT operator"
                raise FilterError(msg)
            conditions = condition.get("conditions")
            if not isinstance(conditions, list) or not conditions:
                msg = "NOT operator expects at least one condition"
                raise FilterError(msg)
            return not all(self._check_condition(doc, cond) for cond in conditions)

        # Leaf condition
        if "field" not in condition:
            msg = "Missing 'field' in filter condition"
            raise FilterError(msg)
        field = condition.get("field")
        if not isinstance(field, str):
            msg = "'field' in filter condition must be a string"
            raise FilterError(msg)
        if "value" not in condition:
            msg = "Missing 'value' in filter condition"
            raise FilterError(msg)
        value = condition.get("value")

        doc_val = FAISSDocumentStore._get_doc_value(doc, field)

        # Type check for comparison operators
        if operator in [">", ">=", "<", "<="]:
            if doc_val is None:
                # Haystack specific: if field is missing/None, and we compare, it generally shouldn't match.
                return False

            if value is None:
                # Comparing anything with None using inequalities is invalid,
                # but tests expect efficient handling (no match)
                return False

            # Check for compatibility
            # We allow int/float comparison
            is_number_doc = isinstance(doc_val, (int, float))
            is_number_val = isinstance(value, (int, float))

            if is_number_doc and is_number_val:
                # Compatible
                pass
            elif type(doc_val) is not type(value):
                # Incompatible types for inequality implementation (like str vs int, or list vs int)
                msg = f"Type mismatch: cannot compare {type(doc_val)} with {type(value)}"
                raise FilterError(msg)

            try:
                if operator == ">":
                    return doc_val > value
                if operator == ">=":
                    return doc_val >= value
                if operator == "<":
                    return doc_val < value
                if operator == "<=":
                    return doc_val <= value
            except TypeError as e:
                msg = f"Type mismatch in filter: {e}"
                raise FilterError(msg) from e

        if operator == "==":
            return doc_val == value
        elif operator == "!=":
            return doc_val != value
        elif operator == "in":
            if not isinstance(value, list):
                msg = "Value for 'in' must be a list"
                raise FilterError(msg)
            return doc_val in value
        elif operator == "not in":
            if not isinstance(value, list):
                msg = "Value for 'not in' must be a list"
                raise FilterError(msg)
            return doc_val not in value

        return False

    # Mixin implementations

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Deletes documents that match the provided filters from the store.

        :param filters: A dictionary of filters to apply to find documents to delete.
        :returns: The number of documents deleted.
        :raises FilterError: If the filter structure is invalid.
        :raises DocumentStoreError: If the FAISS index is unexpectedly unavailable when removing embeddings.
        """
        docs_to_delete = self.filter_documents(filters)
        ids = [doc.id for doc in docs_to_delete]
        self.delete_documents(ids)
        return len(ids)

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Returns the number of documents that match the provided filters.

        :param filters: A dictionary of filters to apply.
        :returns: The number of matching documents.
        :raises FilterError: If the filter structure is invalid.
        """
        return len(self.filter_documents(filters))

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Updates documents that match the provided filters with the new metadata.

        Note: Updates are performed in-memory only. To persist these changes,
        you must explicitly call `save()` after updating.

        :param filters: A dictionary of filters to apply to find documents to update.
        :param meta: A dictionary of metadata key-value pairs to update in the matching documents.
        :returns: The number of documents updated.
        :raises FilterError: If the filter structure is invalid.
        """
        docs_to_update = self.filter_documents(filters)
        for doc in docs_to_update:
            doc.meta.update(meta)
            # In this dict implementation, it's updated in place in memory.
        return len(docs_to_update)

    def get_metadata_fields_info(self) -> dict[str, dict[str, Any]]:
        """
        Infers and returns the types of all metadata fields from the stored documents.

        :returns: A dictionary mapping field names to dictionaries with a "type" key
            (e.g. `{"field": {"type": "long"}}`).
        """
        fields_idx = {}
        for doc in self.documents.values():
            for key, value in doc.meta.items():
                if key not in fields_idx:
                    type_name = type(value).__name__
                    if type_name == "str":
                        type_name = "keyword"
                    elif type_name == "int":
                        type_name = "long"
                    elif type_name == "bool":
                        type_name = "boolean"
                    fields_idx[key] = {"type": type_name}
        return fields_idx

    def get_metadata_field_min_max(self, field_name: str) -> dict[str, Any]:
        """
        Returns the minimum and maximum values for a specific metadata field.

        :param field_name: The name of the metadata field.
        :returns: A dictionary with keys "min" and "max" containing the respective min and max values.
        """
        values = []
        for doc in self.documents.values():
            val = (
                FAISSDocumentStore._get_doc_value(doc, field_name)
                if not field_name.startswith("meta.")
                else doc.meta.get(field_name[5:])
            )
            if val is not None:
                values.append(val)

        if not values:
            return {"min": None, "max": None}

        return {"min": min(values), "max": max(values)}

    def get_metadata_field_unique_values(self, field_name: str) -> list[Any]:
        """
        Returns all unique values for a specific metadata field.

        :param field_name: The name of the metadata field.
        :returns: A list of unique values for the specified field.
        """
        values = set()
        for doc in self.documents.values():
            val = (
                FAISSDocumentStore._get_doc_value(doc, field_name)
                if not field_name.startswith("meta.")
                else doc.meta.get(field_name[5:])
            )
            if val is not None:
                values.add(val)
        return list(values)

    def count_unique_metadata_by_filter(self, filters: dict[str, Any], fields: list[str]) -> dict[str, int]:
        """
        Returns a count of unique values for multiple metadata fields, optionally scoped by a filter.

        :param filters: A dictionary of filters to apply.
        :param fields: A list of metadata field names to count unique values for.
        :returns: A dictionary mapping each field name to the count of its unique values.
        """
        filtered_docs = self.filter_documents(filters)
        counts = {}

        for field in fields:
            unique_vals = set()
            for doc in filtered_docs:
                val = FAISSDocumentStore._get_doc_value(doc, field)
                if val is not None:
                    unique_vals.add(val)
            counts[field] = len(unique_vals)

        return dict(counts)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the store to a dictionary.
        """
        return default_to_dict(
            self,
            index_path=self.index_path,
            index_string=self.index_string,
            embedding_dim=self.embedding_dim,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FAISSDocumentStore":
        """
        Deserializes the store from a dictionary.
        """
        return default_from_dict(cls, data)

    def save(self, index_path: str | Path) -> None:
        """
        Saves the index and documents to disk.

        :raises DocumentStoreError: If the FAISS index is unexpectedly unavailable.
        """
        path = Path(index_path)
        faiss.write_index(self._get_index_or_raise(), str(path.with_suffix(".faiss")))

        # Save documents and ID mapping
        data = {
            "documents": [doc.to_dict() for doc in self.documents.values()],
            "id_map": self.id_map,
            "inverse_id_map": self.inverse_id_map,
            "next_id": self._next_id,
        }

        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, index_path: str | Path) -> None:
        """
        Loads the index and documents from disk.

        :raises ValueError: If the `.faiss` file does not exist.
        """
        path = Path(index_path)
        if not path.with_suffix(".faiss").exists():
            msg = f"File not found: {path.with_suffix('.faiss')}"
            raise ValueError(msg)

        self.index = faiss.read_index(str(path.with_suffix(".faiss")))

        with open(path.with_suffix(".json"), encoding="utf-8") as f:
            data = json.load(f)

        self.documents = {doc_dict["id"]: Document.from_dict(doc_dict) for doc_dict in data["documents"]}
        self.id_map = {int(k): v for k, v in data["id_map"].items()}
        # inverse_id_map keys are strings, values are ints. JSON keys are strings.
        self.inverse_id_map = data["inverse_id_map"]
        self._next_id = data["next_id"]

        # Verify sync
        if len(self.documents) != len(self.id_map):
            logger.warning(
                "Loaded %d documents but %d ID mappings. Index might be out of sync.",
                len(self.documents),
                len(self.id_map),
            )
