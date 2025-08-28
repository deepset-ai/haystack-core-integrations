from typing import Any, Dict, List, Union


def remove_key_from_schema(
    schema: Union[Dict[str, Any], List[Any], Any], target_key: str
) -> Union[Dict[str, Any], List[Any], Any]:
    """
    Recursively traverse a schema and remove all occurrences of the target key.


    :param schema: The schema dictionary/list/value to process
    :param target_key: The key to remove from all dictionaries in the schema

    :returns: The schema with the target key removed from all nested dictionaries
    """
    if isinstance(schema, dict):
        # Create a new dict without the target key
        result = {}
        for k, v in schema.items():
            if k != target_key:
                result[k] = remove_key_from_schema(v, target_key)
        return result

    elif isinstance(schema, list):
        return [remove_key_from_schema(item, target_key) for item in schema]

    return schema
