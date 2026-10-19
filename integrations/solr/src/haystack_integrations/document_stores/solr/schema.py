# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Mapping between Haystack `Document`s and Solr documents.

Solr is strongly typed: a field's type is fixed the first time the field is created and a value of the
wrong type is rejected. Haystack metadata, on the other hand, is an arbitrary `dict[str, Any]` whose
value types are only known at write time. To reconcile the two, every metadata entry is stored in a
Solr field whose name encodes the Python type of the value:

    meta.page  = "100"  ->  meta_s_page = "100"     (string)
    meta.page  = 100    ->  meta_l_page = 100       (plong)

The type code lives in the *prefix* rather than the suffix because Solr dynamic field patterns accept
only a leading or a trailing wildcard - `meta_*_s` is not a legal pattern, while `meta_s_*` is.

Encoding the type in the field name buys two properties that a single JSON blob or Solr's schemaless
type inference cannot provide:

- metadata round-trips with its Python type intact, so `{"page": "100"}` never comes back as
  `{"page": 100}`;
- values that merely share a string form stay distinct, so the int `1`, the str `"1"`, the float `1.0`
  and the bool `True` occupy four different fields and are reported as four distinct values.
"""

import json
from typing import Any

from haystack.dataclasses import ByteStream, Document

# Solr fields holding the non-metadata parts of a Document.
ID_FIELD = "id"
CONTENT_FIELD = "content"
EMBEDDING_FIELD = "embedding"
BLOB_FIELD = "blob"

#: Prefix shared by every metadata field.
META_PREFIX = "meta_"

#: Solr internal fields that come back in query responses and must never be mistaken for content.
RESERVED_FIELDS = frozenset({"_version_", "_root_", "_text_", "score"})

#: Scalar type code -> Solr field type. `j` is the JSON fallback for values with no Solr equivalent.
SCALAR_TYPE_CODES: dict[str, str] = {"s": "string", "b": "boolean", "l": "plong", "d": "pdouble"}

#: Multi-valued type codes, one per scalar code, used for homogeneous lists.
LIST_TYPE_CODES: dict[str, str] = {f"{code}s": solr_type for code, solr_type in SCALAR_TYPE_CODES.items()}

JSON_TYPE_CODE = "j"

#: Every type code, in the order the existence union in `filters.py` emits them.
ALL_TYPE_CODES: tuple[str, ...] = (*SCALAR_TYPE_CODES, *LIST_TYPE_CODES, JSON_TYPE_CODE)

#: Solr field type -> the Python type its values decode to. Used by `get_metadata_fields_info`.
SOLR_TYPE_TO_PYTHON: dict[str, str] = {
    "string": "str",
    "boolean": "bool",
    "plong": "int",
    "pdouble": "float",
}


def _scalar_type_code(value: Any) -> str:
    """Return the type code for a scalar metadata value."""
    # bool is a subclass of int, so it has to be tested first.
    if isinstance(value, bool):
        return "b"
    if isinstance(value, int):
        return "l"
    if isinstance(value, float):
        return "d"
    if isinstance(value, str):
        return "s"
    return JSON_TYPE_CODE


def type_code_for_value(value: Any) -> str:
    """
    Return the type code under which `value` is stored.

    Homogeneous lists of scalars use the multi-valued code for their element type. Everything else -
    dicts, mixed lists, nested structures - falls back to a JSON-encoded string.

    :param value: the metadata value to classify.
    :returns: one of the codes in `ALL_TYPE_CODES`.
    """
    if isinstance(value, list):
        if not value:
            # An empty list has no element type to key on, and Solr cannot distinguish an empty
            # multi-valued field from a missing one, so JSON keeps the round-trip exact.
            return JSON_TYPE_CODE
        element_codes = {_scalar_type_code(element) for element in value}
        if len(element_codes) == 1:
            element_code = element_codes.pop()
            if element_code != JSON_TYPE_CODE:
                return f"{element_code}s"
        return JSON_TYPE_CODE
    return _scalar_type_code(value)


def meta_field_name(key: str, type_code: str) -> str:
    """
    Build the Solr field name holding metadata `key` at `type_code`.

    :param key: the Haystack metadata key.
    :param type_code: one of the codes in `ALL_TYPE_CODES`.
    :returns: the Solr field name, e.g. `meta_s_page`.
    """
    return f"{META_PREFIX}{type_code}_{key}"


def parse_meta_field_name(field: str) -> tuple[str, str] | None:
    """
    Invert `meta_field_name`.

    :param field: a Solr field name.
    :returns: a `(type_code, key)` pair, or `None` if `field` is not a metadata field. Type codes
        contain no underscore, so a single split is unambiguous even when the key does.
    """
    if not field.startswith(META_PREFIX):
        return None
    remainder = field[len(META_PREFIX) :]
    type_code, separator, key = remainder.partition("_")
    if not separator or not key or type_code not in ALL_TYPE_CODES:
        return None
    return type_code, key


def validate_meta_keys(meta: dict[str, Any]) -> None:
    """
    Reject metadata keys that cannot be expressed as a Solr field name.

    :param meta: the metadata of a single document.
    :raises ValueError: if any key contains a character outside `[A-Za-z0-9_]`. Silently rewriting
        such keys would let two distinct keys collide, so the write is refused instead.
    """
    invalid = sorted(key for key in meta if not key.replace("_", "").isalnum() or not key)
    if invalid:
        msg = (
            f"Metadata keys must contain only letters, digits and underscores to be stored as Solr "
            f"fields, but got: {invalid}. Rename these keys before writing the documents."
        )
        raise ValueError(msg)


def document_to_solr(document: Document) -> dict[str, Any]:
    """
    Convert a Haystack `Document` into a Solr document.

    :param document: the document to convert.
    :returns: a JSON-serializable dict ready to be posted to Solr's update handler.
    :raises ValueError: if a metadata key cannot be expressed as a Solr field name.
    """
    validate_meta_keys(document.meta)

    solr_document: dict[str, Any] = {ID_FIELD: document.id}
    if document.content is not None:
        solr_document[CONTENT_FIELD] = document.content
    if document.embedding is not None:
        solr_document[EMBEDDING_FIELD] = list(document.embedding)
    if document.blob is not None:
        solr_document[BLOB_FIELD] = json.dumps(document.blob.to_dict())

    for key, value in document.meta.items():
        if value is None:
            # Solr has no null: a missing field already means "no value", and storing one would make
            # `== None` and `!= None` disagree with the round-tripped metadata.
            continue
        type_code = type_code_for_value(value)
        stored = json.dumps(value) if type_code == JSON_TYPE_CODE else value
        solr_document[meta_field_name(key, type_code)] = stored

    return solr_document


def solr_to_document(solr_document: dict[str, Any], *, score: float | None = None) -> Document:
    """
    Convert a Solr document back into a Haystack `Document`.

    :param solr_document: a single entry from a Solr query response.
    :param score: the relevance score to attach, when the document came from a retrieval query.
    :returns: the reconstructed document.
    """
    meta: dict[str, Any] = {}
    content: str | None = None
    embedding: list[float] | None = None
    blob: ByteStream | None = None

    for field, value in solr_document.items():
        if field in RESERVED_FIELDS or field == ID_FIELD:
            continue
        if field == CONTENT_FIELD:
            content = value
            continue
        if field == EMBEDDING_FIELD:
            embedding = list(value) if value is not None else None
            continue
        if field == BLOB_FIELD:
            blob = ByteStream.from_dict(json.loads(value))
            continue
        parsed = parse_meta_field_name(field)
        if parsed is None:
            continue
        type_code, key = parsed
        meta[key] = json.loads(value) if type_code == JSON_TYPE_CODE else value

    return Document(id=solr_document[ID_FIELD], content=content, meta=meta, embedding=embedding, blob=blob, score=score)


def vector_field_type_name(embedding_dim: int) -> str:
    """
    Return the name of the `DenseVectorField` type backing embeddings of `embedding_dim` dimensions.

    :param embedding_dim: the embedding dimension.
    :returns: the Solr field type name.
    """
    return f"haystack_knn_{embedding_dim}"


def schema_payload(
    *,
    embedding_dim: int,
    similarity_function: str,
    existing_field_types: set[str],
    existing_fields: set[str],
    existing_dynamic_fields: set[str],
    vector_field_type_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build an idempotent Schema API payload creating only what the core is missing.

    :param embedding_dim: dimension of the `DenseVectorField` backing embeddings.
    :param similarity_function: `cosine`, `dot_product` or `euclidean`.
    :param existing_field_types: field type names already defined in the core.
    :param existing_fields: field names already defined in the core.
    :param existing_dynamic_fields: dynamic field patterns already defined in the core.
    :param vector_field_type_params: extra attributes for the vector field type, for example
        `{"hnswM": 32}` on Solr 10 or `{"hnswMaxConnections": 32}` on Solr 9. Left unset by default so
        that one payload is valid on both major versions, which renamed these attributes.
    :returns: the Schema API payload. Empty when the core already has everything.
    """
    payload: dict[str, Any] = {}

    vector_type = vector_field_type_name(embedding_dim)
    if vector_type not in existing_field_types:
        payload["add-field-type"] = {
            "name": vector_type,
            "class": "solr.DenseVectorField",
            "vectorDimension": embedding_dim,
            "similarityFunction": similarity_function,
            **(vector_field_type_params or {}),
        }

    wanted_fields = [
        {"name": CONTENT_FIELD, "type": "text_general", "indexed": True, "stored": True, "multiValued": False},
        {"name": EMBEDDING_FIELD, "type": vector_type, "indexed": True, "stored": True},
        {"name": BLOB_FIELD, "type": "string", "indexed": False, "stored": True, "multiValued": False},
    ]
    missing_fields = [field for field in wanted_fields if field["name"] not in existing_fields]
    if missing_fields:
        payload["add-field"] = missing_fields

    wanted_dynamic_fields = []
    for type_code, solr_type in {**SCALAR_TYPE_CODES, **LIST_TYPE_CODES}.items():
        wanted_dynamic_fields.append(
            {
                "name": f"{META_PREFIX}{type_code}_*",
                "type": solr_type,
                "indexed": True,
                "stored": True,
                "docValues": True,
                "multiValued": type_code in LIST_TYPE_CODES,
            }
        )
    wanted_dynamic_fields.append(
        # The JSON fallback is opaque to Solr, so there is nothing to gain from indexing it.
        {
            "name": f"{META_PREFIX}{JSON_TYPE_CODE}_*",
            "type": "string",
            "indexed": False,
            "stored": True,
            "docValues": False,
            "multiValued": False,
        }
    )
    missing_dynamic_fields = [field for field in wanted_dynamic_fields if field["name"] not in existing_dynamic_fields]
    if missing_dynamic_fields:
        payload["add-dynamic-field"] = missing_dynamic_fields

    return payload
