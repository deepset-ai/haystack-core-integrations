import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
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
        sql_url: str = "sqlite:///:memory:",  # Kept for compatibility but unused
        index_path: str | None = None,
        index_string: str = "Flat",
        embedding_dim: int = 768,
        faiss_index_factory_str: str | None = None,  # Legacy parameter support
        similarity: str = "dot_product",  # Legacy parameter
        isolation_level: str | None = None,  # Legacy parameter
        duplicate_documents: str = "overwrite",  # Legacy parameter
        return_embedding: bool = True,
        progress_bar: bool = True,  # Legacy parameter
    ):
        """
        Initializes the FAISSDocumentStore.

        :param index_path: Path to save/load the index and documents. If None, the store is in-memory only.
        :param index_string: The FAISS index factory string. Default is "Flat".
        :param embedding_dim: The dimension of the embeddings. Default is 768.
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
            raise DocumentStoreError(
                f"Could not create FAISS index with factory string '{self.index_string}': {e}"
            ) from e

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
        Currently supports simple equality check for 'field' == 'value', and logical operators AND/OR/NOT are NOT fully implemented in this MVP helper.
        Wait, Haystack 2.x filters are complex. We should use a proper filter parser or a simple recusive check if we want to support full syntax.
        For MVP, let's implement basic filtering logic.
        """
        if "operator" not in filters:
            # Simple legacy style or simple dict: {"field": "value"} - NOT standard 2.x but often used in tests?
            # Standard 2.x filters usually have an operator at the top level (AND/OR) or a comparison.
            # If it's a leaf node {field, operator, value}:
            pass

        return self._check_condition(doc, filters)

    def _get_doc_value(self, doc: Document, field: str) -> Any:
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
        """
        if not documents:
            return 0

        # Check for duplicates first if policy is FAIL
        if policy == DuplicatePolicy.FAIL:
            for doc in documents:
                if doc.id in self.documents:
                    raise DuplicateDocumentError(f"Document with id '{doc.id}' already exists.")

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
            self.index.add_with_ids(vectors, ids)

        return docs_written

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents from the store.
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

        if ids_to_remove_from_index and self.index.ntotal > 0:
            ids_array = np.array(ids_to_remove_from_index, dtype="int64")
            self.index.remove_ids(ids_array)

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
        """
        if not self.index or self.index.ntotal == 0:
            return []

        # Ensure embedding format
        query_vec = np.array([query_embedding], dtype="float32")

        # Search in FAISS
        # Valid strategy for pre-filtering vs post-filtering:
        # Since FAISS `IndexIDMap` doesn't support pre-filtering natively comfortably without `RangeSearch` or specialized impls,
        # we usually fetch more (k * scale_factor) and filter post-retrieval.

        fetch_k = top_k
        if filters:
            fetch_k = min(self.index.ntotal, top_k * 10)  # Simple heuristic

        distances, indices = self.index.search(query_vec, fetch_k)

        results = []
        for dist, int_id in zip(distances[0], indices[0]):
            if int_id == -1:
                continue

            doc_id = self.id_map.get(int_id)
            if not doc_id or doc_id not in self.documents:
                continue

            doc = self.documents[doc_id]

            if filters and not self._matches_filters(doc, filters):
                continue

            # Set score. Calculate from distance if needed.
            # For FlatL2, distance is L2 distance. score = 1 / (1 + distance) is common in Haystack.
            # But the Protocol expects the Document to be returned, user can inspect score.
            # We should probably clone the document to avoid modifying the stored one.
            result_doc = Document.from_dict(doc.to_dict())
            result_doc.score = (
                float(1 / (1 + dist)) if self.index_string == "Flat" else float(dist)
            )  # Simplified score handling

            results.append(result_doc)

            if len(results) >= top_k:
                break

        return results

    def _get_result_to_documents(self, result) -> list[Document]:
        # Compatibility/Helper if matching Chroma approach
        return []

    def _check_condition(self, doc: Document, condition: dict[str, Any]) -> bool:
        if "operator" not in condition and "conditions" not in condition:
            # This might be a legacy or malformed filter from tests like test_missing_top_level_operator_key
            # The standard Haystack filter structure enforces keys.
            # On failure to parse standard structure, we should raise FilterError as per tests?
            # Actually, looking at the tests (e.g. TestFAISSDocumentStore.test_missing_top_level_operator_key),
            # they expect FilterError if "operator" is missing from a condition block.
            raise FilterError("Filter condition missing 'operator'")

        operator = condition.get("operator", "==")

        if operator == "AND":
            if "conditions" not in condition:
                raise FilterError("Missing 'conditions' for AND operator")
            return all(self._check_condition(doc, cond) for cond in condition.get("conditions", []))
        elif operator == "OR":
            if "conditions" not in condition:
                raise FilterError("Missing 'conditions' for OR operator")
            return any(self._check_condition(doc, cond) for cond in condition.get("conditions", []))
        elif operator == "NOT":
            if "conditions" not in condition:
                raise FilterError("Missing 'conditions' for NOT operator")
            return not self._check_condition(doc, condition.get("conditions", [])[0])

        # Leaf condition
        if "field" not in condition:
            raise FilterError("Missing 'field' in filter condition")
        field = condition.get("field")
        if "value" not in condition:
            raise FilterError("Missing 'value' in filter condition")
        value = condition.get("value")

        doc_val = self._get_doc_value(doc, field)

        # Type check for comparison operators
        if operator in [">", ">=", "<", "<="]:
            if doc_val is None:
                # Haystack specific: if field is missing/None, and we compare, it generally shouldn't match.
                return False

            if value is None:
                # Comparing anything with None using inequalities is invalid, but tests expect efficient handling (no match)
                return False

            # Check for compatibility
            # We allow int/float comparison
            is_number_doc = isinstance(doc_val, (int, float))
            is_number_val = isinstance(value, (int, float))

            if is_number_doc and is_number_val:
                # Compatible
                pass
            elif type(doc_val) != type(value):
                # Incompatible types for inequality implementation (like str vs int, or list vs int)
                raise FilterError(f"Type mismatch: cannot compare {type(doc_val)} with {type(value)}")

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
                raise FilterError(f"Type mismatch in filter: {e}") from e

        if operator == "==":
            return doc_val == value
        elif operator == "!=":
            return doc_val != value
        elif operator == "in":
            if not isinstance(value, list):
                raise FilterError("Value for 'in' must be a list")
            return doc_val in value
        elif operator == "not in":
            if not isinstance(value, list):
                raise FilterError("Value for 'not in' must be a list")
            return doc_val not in value

        return False

    # Mixin implementations

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        docs_to_delete = self.filter_documents(filters)
        ids = [doc.id for doc in docs_to_delete]
        self.delete_documents(ids)
        return len(ids)

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        return len(self.filter_documents(filters))

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        docs_to_update = self.filter_documents(filters)
        for doc in docs_to_update:
            doc.meta.update(meta)
            # Re-write to ensure persistence if we had auto-save (we don't yet)
            # In this dict implementation, it's updated in place in memory.
        return len(docs_to_update)

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
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
        values = []
        for doc in self.documents.values():
            val = (
                self._get_doc_value(doc, field_name)
                if not field_name.startswith("meta.")
                else doc.meta.get(field_name[5:])
            )
            if val is not None:
                values.append(val)

        if not values:
            return {"min": None, "max": None}

        return {"min": min(values), "max": max(values)}

    def get_metadata_field_unique_values(self, field_name: str) -> list[Any]:
        values = set()
        for doc in self.documents.values():
            val = (
                self._get_doc_value(doc, field_name)
                if not field_name.startswith("meta.")
                else doc.meta.get(field_name[5:])
            )
            if val is not None:
                values.add(val)
        return list(values)

    def count_unique_metadata_by_filter(self, filters: dict[str, Any], fields: list[str]) -> dict[str, int]:
        filtered_docs = self.filter_documents(filters)
        counts = defaultdict(int)
        # Wait, the return type is Dict[str, int] mapping field -> unique_count?
        # Specification says: "Returns a dictionary mapping each field name to the count of its unique values."

        for field in fields:
            unique_vals = set()
            for doc in filtered_docs:
                val = self._get_doc_value(doc, field)
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
        """
        path = Path(index_path)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))

        # Save documents and ID mapping
        data = {
            "documents": [doc.to_dict() for doc in self.documents.values()],
            "id_map": self.id_map,
            "inverse_id_map": self.inverse_id_map,
            "next_id": self._next_id,
        }

        with open(path.with_suffix(".json"), "w") as f:
            json.dump(data, f)

    def load(self, index_path: str | Path) -> None:
        """
        Loads the index and documents from disk.
        """
        path = Path(index_path)
        if not path.with_suffix(".faiss").exists():
            raise ValueError(f"File not found: {path.with_suffix('.faiss')}")

        self.index = faiss.read_index(str(path.with_suffix(".faiss")))

        with open(path.with_suffix(".json")) as f:
            data = json.load(f)

        self.documents = {doc_dict["id"]: Document.from_dict(doc_dict) for doc_dict in data["documents"]}
        self.id_map = {int(k): v for k, v in data["id_map"].items()}
        # inverse_id_map keys are strings, values are ints. JSON keys are strings.
        self.inverse_id_map = data["inverse_id_map"]
        self._next_id = data["next_id"]

        # Verify sync
        if len(self.documents) != len(self.id_map):
            logger.warning(
                f"Loaded {len(self.documents)} documents but {len(self.id_map)} ID mappings. Index might be out of sync."
            )
