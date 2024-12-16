from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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


@dataclass
class ChromaFilter:
    """
    Dataclass to store the converted filter structure used in Chroma queries.

    Following filter criteria are supported:
    - `ids`: A list of document IDs to filter by in Chroma collection.
    - `where`: A dictionary of metadata filters applied to the documents.
    - `where_document`: A dictionary of content-based filters applied to the documents' content.
    """

    ids: List[str]
    where: Optional[Dict[str, Any]]
    where_document: Optional[Dict[str, Any]]


def _convert_filters(filters: Dict[str, Any]) -> ChromaFilter:
    """
    Converts Haystack filters into a format compatible with Chroma, separating them into ids, metadata filters,
    and content filters to be passed to chroma as ids, where, and where_document clauses respectively.
    """

    ids = []
    where: Dict[str, Any] = defaultdict(list)
    where_document: Dict[str, Any] = defaultdict(list)

    converted_filters = _convert_filter_clause(filters)
    for field, value in converted_filters.items():
        if value is None:
            continue

        # Chroma differentiates between metadata filters and content filters,
        # with each filter applying to only one type.
        # If 'where_document' is populated, it's a content filter.
        # In this case, we skip further processing of metadata filters for this field.
        where_document.update(_create_where_document_filter(field, value))
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
            test_clause = "document content filter"
            validate_where_document(where_document)
        elif where:
            test_clause = "metadata filter"
            validate_where(where)
    except ValueError as e:
        msg = f"Invalid '{test_clause}' : {e}"
        raise ChromaDocumentStoreFilterError(msg) from e

    return ChromaFilter(ids=ids, where=where or None, where_document=where_document or None)


def _convert_filter_clause(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts Haystack filters to Chroma compatible filters.
    """
    converted_clauses = {}

    if "field" in filters:
        converted_clauses.update(_parse_comparison_condition(filters))
    else:
        converted_clauses.update(_parse_logical_condition(filters))

    return converted_clauses


def _create_where_document_filter(field: str, value: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Method to check if given haystack filter is a document filter
    and converts it to Chroma-compatible where_document filter.

    """
    where_document: Dict[str, List[Any]] = defaultdict(list)

    # Create a single document filter for the content field
    if field == "content":
        return value
    # In case of a logical operator, check if the given filters contain "content"
    # Then combine the filters into a single where_document filter to pass to Chroma
    if field in ["$and", "$or"] and value[0].get("content"):
        # Use list comprehension to populate the field without modifying the original structure
        document_filters = [
            _create_where_document_filter(k, v) for v in value if isinstance(v, dict) for k, v in v.items()
        ]
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
    conditions = [_convert_filter_clause(c) for c in condition["conditions"]]

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

    if operator not in OPERATORS:
        msg = f"Unknown operator {operator}. Valid operators are: {list(OPERATORS.keys())}"
        raise ChromaDocumentStoreFilterError(msg)
    return {field: {OPERATORS[operator]: value}}
