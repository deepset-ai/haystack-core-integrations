# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import pytest
from astrapy.data.utils.collection_converters import postprocess_collection_response
from astrapy.info import CollectionDefinition, CollectionDescriptor
from astrapy.utils.api_options import defaultAPIOptions
from haystack import Document

from haystack_integrations.document_stores.astra.astra_client import (
    _API_OPTIONS,
    _collection_indexing_warning,
    _find_kwargs,
    _to_documents,
)


def test_api_options_read_plain_python_types():
    serdes = defaultAPIOptions(environment="prod").with_override(_API_OPTIONS).serdes_options
    doc = postprocess_collection_response(
        {"_id": "1", "$vector": [0.12345678901234568, 0.2], "meta": {"d": {"$date": 0}}}, options=serdes
    )
    assert isinstance(doc["$vector"], list)
    assert doc["$vector"] == [0.12345678901234568, 0.2]
    assert isinstance(doc["meta"]["d"], datetime)


@pytest.mark.parametrize(
    "indexing,expected",
    [
        ({"deny": ["metadata._node_content", "content"]}, None),
        (None, "having indexing turned on"),
        ({"deny": ["something_else"]}, "unexpected 'indexing' settings"),
    ],
)
def test_collection_indexing_warning(indexing, expected):
    descriptor = CollectionDescriptor(name="c", definition=CollectionDefinition(indexing=indexing), raw_descriptor={})
    warning = _collection_indexing_warning(descriptor)
    assert warning is None if expected is None else expected in warning


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"limit": 2}, {"filter": {"k": 1}, "limit": 2, "projection": {"*": 1}}),
        (
            {"vector": [0.1], "limit": 3},
            {
                "filter": {"k": 1},
                "limit": 3,
                "projection": {"*": 1},
                "sort": {"$vector": [0.1]},
                "include_similarity": True,
            },
        ),
    ],
)
def test_find_kwargs(kwargs, expected):
    assert _find_kwargs({"k": 1}, **kwargs) == expected


def test_to_documents_maps_fields_without_mutating_input():
    response = {"_id": "1", "$similarity": 0.5, "content": "hi", "$vector": [0.1], "meta": {"k": "v"}, "extra": 1}
    assert _to_documents([response]) == [Document(id="1", content="hi", embedding=[0.1], meta={"k": "v"}, score=0.5)]
    assert "_id" in response


def test_to_documents_warns_when_empty(caplog):
    assert _to_documents([]) == []
    assert "No documents found" in caplog.text
