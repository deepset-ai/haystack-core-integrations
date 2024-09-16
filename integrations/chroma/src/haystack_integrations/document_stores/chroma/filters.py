from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

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


def _convert_filters(
    filters: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    """
    Converts Haystack filters into a format compatible with Chroma, separating them into ids, metadata filters,
    and content filters to be passed to chroma as ids, where, and where_document clauses respectively.

    """

    filters = _normalize_filters(filters)

    ids = []
    where: Dict[str, Any] = defaultdict(list)
    where_document: Dict[str, Any] = defaultdict(list)

    if isinstance(filters, dict):  # if filters is a dict, convert it to a list
        filters = [filters]

    for clause in filters:
        for field, value in clause.items():
            if value is None:
                continue
            where_document.update(create_where_document_filter(field, value))
            # if where_document is not empty, current clause is a content filter and we can skip rest of the loop
            if where_document:
                continue
            # if field is "id", it'll be passed to Chroma's ids filter
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


def _normalize_filters(filters: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Converts Haystack filters to Chroma compatible filters.
    """
    normalized_filters = {}

    if not isinstance(filters, list):
        filters = [filters]
    for condition in filters:
        if "field" in condition:
            normalized_filters.update(_parse_comparison_condition(condition))
        else:
            normalized_filters.update(_parse_logical_condition(condition))

    return normalized_filters


def create_where_document_filter(field, value) -> Dict[str, Any]:
    """
    Method to convert Haystack filters with the "content" field to Chroma-compatible document filters

    """
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
