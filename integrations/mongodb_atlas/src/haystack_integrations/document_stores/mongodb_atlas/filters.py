from typing import Any, Dict, Optional


def haystack_filters_to_mongo(filters: Optional[Dict[str, Any]]):
    # TODO
    if filters:
        msg = "Filtering not yet implemented for MongoDBAtlasDocumentStore"
        raise ValueError(msg)
    return {}
