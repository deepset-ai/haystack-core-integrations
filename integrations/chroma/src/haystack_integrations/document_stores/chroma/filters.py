from collections import defaultdict
from typing import Any, Dict, List, Tuple

from chromadb.api.types import validate_where, validate_where_document

from .errors import ChromaDocumentStoreFilterError

OPERATORS = {
    "==": "$eq",
    "!=": "$ne",
    ">": "$gt",
    ">=": "$gte",
    "<": "$lt",
    "<=": "$lte",
    "in": "$in",
    "not in": "$nin",
    "AND": "$and",
    "OR": "$or",
    "contains": "$contains",
    "not contains": "$not_contains",
}


def _convert_filters(filters: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    """
    Convert haystack filters to chroma filter string capturing all boolean operators
    """

    filters = _normalize_filters(filters)

    ids = []
    where: Dict[str, Any] = defaultdict(list)
    where_document: Dict[str, Any] = defaultdict(list)

    for field, value in filters.items():
        if value is None:
            continue
        where_document.update(create_where_document_filter(field, value))
        if where_document:  # if where_document is not empty, skip the rest of the loop
            continue
        elif field == "id":
            if not value["$eq"]:
                msg = f"id filter only supports '==' operator, got {value}"
                raise ChromaDocumentStoreFilterError(msg)
            ids.append(value["$eq"])
        else:
            where[field] = value

    try:
        if where_document:
            validate_where_document(where_document)
        elif where:
            validate_where(where)
    except ValueError as e:
        raise ChromaDocumentStoreFilterError(e) from e

    return ids, where, where_document


def _normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts Haystack filters to Chroma compatible filters.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise ChromaDocumentStoreFilterError(msg)

    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


# Method to covnert haystack filters with "content" field to chroma document filters
def create_where_document_filter(field, value) -> Dict[str, Any]:
    where_document: Dict[str, Any] = defaultdict(list)
    document_filters = []

    if value is None:
        return where_document
    if field == "content":
        return value
    if field in ["$and", "$or"] and value[0].get("content"):
        # Use list comprehension to populate the field without modifying the original structure
        document_filters = [
            create_where_document_filter(k, v) for v in value if isinstance(v, dict) for k, v in v.items()
        ]
    if document_filters:
        where_document[field] = document_filters
    return where_document


def _parse_logical_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise ChromaDocumentStoreFilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise ChromaDocumentStoreFilterError(msg)

    operator = condition["operator"]
    conditions = [_normalize_filters(c) for c in condition["conditions"]]

    if operator not in OPERATORS:
        msg = f"Unknown operator {operator}"
        raise ChromaDocumentStoreFilterError(msg)
    return {OPERATORS[operator]: conditions}


def _parse_comparison_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    if "field" not in condition:
        msg = f"'field' key missing in {condition}"
        raise ChromaDocumentStoreFilterError(msg)
    field: str = ""
    # remove the "meta." prefix from the field name
    if condition["field"].startswith("meta."):
        field = condition["field"].split(".")[-1]
    else:
        field = condition["field"]

    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise ChromaDocumentStoreFilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise ChromaDocumentStoreFilterError(msg)
    operator: str = condition["operator"]
    value: Any = condition["value"]

    return {field: {OPERATORS[operator]: value}}
