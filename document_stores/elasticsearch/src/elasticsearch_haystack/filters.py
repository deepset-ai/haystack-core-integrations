from typing import Any, Dict, List, Union

from haystack.preview.errors import FilterError
from pandas import DataFrame


def _normalize_filters(filters: Union[List[Dict], Dict], logical_condition="") -> Dict[str, Any]:
    """
    Converts Haystack filters in ElasticSearch compatible filters.
    """
    if not isinstance(filters, dict) and not isinstance(filters, list):
        msg = "Filters must be either a dictionary or a list"
        raise FilterError(msg)
    conditions = []
    if isinstance(filters, dict):
        filters = [filters]
    for filter_ in filters:
        for operator, value in filter_.items():
            if operator in ["$not", "$and", "$or"]:
                # Logical operators
                conditions.append(_normalize_filters(value, operator))
            else:
                # Comparison operators
                conditions.extend(_parse_comparison(operator, value))

    if len(conditions) > 1:
        conditions = _normalize_ranges(conditions)
    else:
        conditions = conditions[0]

    if logical_condition == "$not":
        return {"bool": {"must_not": conditions}}
    elif logical_condition == "$or":
        return {"bool": {"should": conditions}}

    # If no logical condition is specified we default to "$and"
    return {"bool": {"must": conditions}}


def _parse_comparison(field: str, comparison: Union[Dict, List, str, float]) -> List:
    result: List[Dict[str, Any]] = []
    if isinstance(comparison, dict):
        for comparator, val in comparison.items():
            if isinstance(val, DataFrame):
                val = val.to_json()
            if comparator == "$eq":
                if isinstance(val, list):
                    result.append(
                        {
                            "terms_set": {
                                field: {
                                    "terms": val,
                                    "minimum_should_match_script": {
                                        "source": f"Math.max(params.num_terms, doc['{field}'].size())"
                                    },
                                }
                            }
                        }
                    )
                result.append({"term": {field: val}})
            elif comparator == "$ne":
                if isinstance(val, list):
                    result.append({"bool": {"must_not": {"terms": {field: val}}}})
                else:
                    result.append(
                        {"bool": {"must_not": {"match": {field: {"query": val, "minimum_should_match": "100%"}}}}}
                    )
            elif comparator == "$in":
                if not isinstance(val, list):
                    msg = f"{field}'s value must be a list when using '{comparator}' comparator"
                    raise FilterError(msg)
                result.append({"terms": {field: val}})
            elif comparator == "$nin":
                if not isinstance(val, list):
                    msg = f"{field}'s value must be a list when using '{comparator}' comparator"
                    raise FilterError(msg)
                result.append({"bool": {"must_not": {"terms": {field: val}}}})
            elif comparator in ["$gt", "$gte", "$lt", "$lte"]:
                if not isinstance(val, str) and not isinstance(val, int) and not isinstance(val, float):
                    msg = f"{field}'s value must be 'str', 'int', 'float' types when using '{comparator}' comparator"
                    raise FilterError(msg)
                result.append({"range": {field: {comparator[1:]: val}}})
            elif comparator in ["$not", "$or"]:
                if isinstance(val, list):
                    # This handles corner cases like this:
                    # `{"name": {"$or": [{"$eq": "name_0"}, {"$eq": "name_1"}]}}`
                    # If we don't handle it like this we'd lose the "name" field and the
                    # generated query would be wrong and return unexpected results.
                    comparisons = [_parse_comparison(field, v)[0] for v in val]
                    if comparator == "$not":
                        result.append({"bool": {"must_not": comparisons}})
                    elif comparator == "$or":
                        result.append({"bool": {"should": comparisons}})
                else:
                    result.append(_normalize_filters(val, comparator))
            elif comparator == "$and" and isinstance(val, list):
                # We're assuming there are no duplicate items in the list
                flat_filters = {k: v for d in val for k, v in d.items()}
                result.extend(_parse_comparison(field, flat_filters))
            elif comparator == "$and":
                result.append(_normalize_filters({field: val}, comparator))
            else:
                msg = f"Unknown comparator '{comparator}'"
                raise FilterError(msg)
    elif isinstance(comparison, list):
        result.append({"terms": {field: comparison}})
    elif isinstance(comparison, DataFrame):
        result.append({"match": {field: {"query": comparison.to_json(), "minimum_should_match": "100%"}}})
    elif isinstance(comparison, str):
        # We can't use "term" for text fields as ElasticSearch changes the value of text.
        # More info here:
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-term-query.html#query-dsl-term-query
        result.append({"match": {field: {"query": comparison, "minimum_should_match": "100%"}}})
    else:
        result.append({"term": {field: comparison}})
    return result


def _normalize_ranges(conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges range conditions acting on a same field.

    Example usage:

    ```python
    conditions = [
        {"range": {"date": {"lt": "2021-01-01"}}},
        {"range": {"date": {"gte": "2015-01-01"}}},
    ]
    conditions = _normalize_ranges(conditions)
    assert conditions == [
        {"range": {"date": {"lt": "2021-01-01", "gte": "2015-01-01"}}},
    ]
    ```
    """
    range_conditions = [next(iter(c["range"].items())) for c in conditions if "range" in c]
    if range_conditions:
        conditions = [c for c in conditions if "range" not in c]
        range_conditions_dict: Dict[str, Any] = {}
        for field_name, comparison in range_conditions:
            if field_name not in range_conditions_dict:
                range_conditions_dict[field_name] = {}
            range_conditions_dict[field_name].update(comparison)

        for field_name, comparisons in range_conditions_dict.items():
            conditions.append({"range": {field_name: comparisons}})
    return conditions
