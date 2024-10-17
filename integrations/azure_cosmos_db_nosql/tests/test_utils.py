import random
from typing import Any, Dict, List

from haystack.dataclasses import Document


def get_vector_indexing_policy(embedding_type: str) -> Dict[str, Any]:
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": embedding_type}],
    }


def get_vector_embedding_policy(
        data_type: str, dimensions: int, distance_function: str
) -> Dict[str, Any]:
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": data_type,
                "dimensions": dimensions,
                "distanceFunction": distance_function,
            }
        ]
    }


def get_documents() -> List[Document]:
    documents = [Document(
        content=f"Document A",
        meta={
            "name": f"name_A",
            "page": "100",
            "chapter": "intro",
            "number": 2,
            "date": "1969-07-21T20:17:40",
        },
        embedding=_random_embeddings(768),
    ), Document(
        content=f"Document B",
        meta={
            "name": f"name_B",
            "page": "123",
            "chapter": "abstract",
            "number": -2,
            "date": "1972-12-11T19:54:58",
        },
        embedding=_random_embeddings(768),
    ), Document(
        content=f"Document C",
        meta={
            "name": f"name_C",
            "page": "90",
            "chapter": "conclusion",
            "number": -10,
            "date": "1989-11-09T17:53:00",
        },
        embedding=_random_embeddings(768),
    )]
    return documents


def _random_embeddings(n):
    return [random.random() for _ in range(n)]
