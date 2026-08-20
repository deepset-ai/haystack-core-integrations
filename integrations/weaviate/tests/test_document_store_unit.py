# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import weaviate
from haystack.dataclasses import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.weaviate.document_store import (
    DOCUMENT_COLLECTION_PROPERTIES,
    WeaviateDocumentStore,
)

STORE_MODULE = "haystack_integrations.document_stores.weaviate.document_store"


def _property(name: str, data_type: str = "text") -> SimpleNamespace:
    """A collection schema property, as Weaviate's config.get() reports it."""
    return SimpleNamespace(name=name, data_type=SimpleNamespace(value=data_type))


def _config(*properties: SimpleNamespace) -> SimpleNamespace:
    special = [
        SimpleNamespace(name=prop["name"], data_type=SimpleNamespace(value="text"))
        for prop in DOCUMENT_COLLECTION_PROPERTIES
    ]
    return SimpleNamespace(properties=[*special, *properties])


def _groups(*values) -> SimpleNamespace:
    return SimpleNamespace(
        groups=[SimpleNamespace(grouped_by=SimpleNamespace(value=v)) for v in values],
        total_count=len(values),
    )


@pytest.fixture
def store() -> WeaviateDocumentStore:
    """
    A store with its collection injected.

    The `collection` property returns `_collection` when it is already set, so no
    Weaviate client is ever created.
    """
    with patch(f"{STORE_MODULE}.weaviate"):
        store = WeaviateDocumentStore(url="http://localhost:8080")
    store._collection = MagicMock()
    return store


@pytest.fixture
def async_store() -> WeaviateDocumentStore:
    with patch(f"{STORE_MODULE}.weaviate"):
        store = WeaviateDocumentStore(url="http://localhost:8080")
    collection = MagicMock()
    collection.aggregate.over_all = AsyncMock()
    collection.config.get = AsyncMock()
    store._async_collection = collection
    return store


class TestCleanConnectionSettings:
    @pytest.mark.parametrize(
        ("settings", "expected_class", "expected_properties"),
        [
            ({"class": "lowercase_name"}, "Lowercase_name", DOCUMENT_COLLECTION_PROPERTIES),
            ({}, "Default", DOCUMENT_COLLECTION_PROPERTIES),
            (
                {"class": "Custom", "properties": [{"name": "custom", "dataType": ["text"]}]},
                "Custom",
                [{"name": "custom", "dataType": ["text"]}],
            ),
        ],
        ids=["capitalizes-the-class", "defaults-the-class", "keeps-given-properties"],
    )
    def test_normalizes_the_collection_settings(self, settings, expected_class, expected_properties):
        with patch(f"{STORE_MODULE}.weaviate"):
            store = WeaviateDocumentStore(url="http://localhost:8080", collection_settings=settings)

        assert store._collection_settings["class"] == expected_class
        assert store._collection_settings["properties"] == expected_properties


class TestComputeFieldUniqueValues:
    def test_returns_one_value_per_group(self):
        values, total = WeaviateDocumentStore._compute_field_unique_values(
            _groups("book", "paper"), search_term=None, from_=0, size=10
        )

        assert values == ["book", "paper"]
        assert total == 2

    def test_sorts_the_values_so_pagination_is_stable(self):
        values, _ = WeaviateDocumentStore._compute_field_unique_values(
            _groups("paper", "book"), search_term=None, from_=0, size=10
        )

        assert values == ["book", "paper"]

    def test_matches_the_search_term_case_insensitively(self):
        # Weaviate's own `like` filter is case-sensitive, which is why this happens in Python.
        values, total = WeaviateDocumentStore._compute_field_unique_values(
            _groups("Book", "Paper"), search_term="boo", from_=0, size=10
        )

        assert values == ["Book"]
        assert total == 1

    def test_paginates_after_filtering(self):
        values, total = WeaviateDocumentStore._compute_field_unique_values(
            _groups("a", "b", "c"), search_term=None, from_=1, size=1
        )

        assert values == ["b"]
        assert total == 3

    def test_handles_a_result_without_groups(self):
        assert WeaviateDocumentStore._compute_field_unique_values(
            SimpleNamespace(groups=None), search_term=None, from_=0, size=10
        ) == ([], 0)


class TestCountDocuments:
    def test_count_documents(self, store):
        store._collection.aggregate.over_all.return_value = SimpleNamespace(total_count=7)

        assert store.count_documents() == 7

    def test_count_documents_treats_a_missing_total_as_zero(self, store):
        store._collection.aggregate.over_all.return_value = SimpleNamespace(total_count=None)

        assert store.count_documents() == 0

    def test_count_documents_by_filter_passes_the_converted_filter(self, store):
        store._collection.aggregate.over_all.return_value = SimpleNamespace(total_count=2)

        assert store.count_documents_by_filter({"field": "meta.kind", "operator": "==", "value": "book"}) == 2
        assert store._collection.aggregate.over_all.call_args.kwargs["filters"] is not None

    def test_count_documents_by_filter_rejects_a_malformed_filter(self, store):
        with pytest.raises(ValueError, match="Invalid filter syntax"):
            store.count_documents_by_filter({"field": "meta.kind"})


class TestGetMetadataFieldsInfo:
    def test_reports_the_user_metadata_fields_with_their_types(self, store):
        store._collection.config.get.return_value = _config(_property("year", "int"), _property("kind", "text"))

        assert store.get_metadata_fields_info() == {"year": {"type": "int"}, "kind": {"type": "text"}}

    def test_leaves_out_the_fields_the_store_itself_owns(self, store):
        store._collection.config.get.return_value = _config()

        assert store.get_metadata_fields_info() == {}


class TestGetMetadataFieldMinMax:
    @pytest.mark.parametrize("data_type", ["int", "number", "date"])
    def test_supports_the_aggregatable_types(self, store, data_type):
        store._collection.config.get.return_value = _config(_property("year", data_type))
        store._collection.aggregate.over_all.return_value = SimpleNamespace(
            total_count=2, properties={"year": SimpleNamespace(minimum=2020, maximum=2024)}
        )

        assert store.get_metadata_field_min_max("meta.year") == {"min": 2020, "max": 2024}

    def test_rejects_a_type_that_cannot_be_aggregated(self, store):
        store._collection.config.get.return_value = _config(_property("kind", "text"))

        with pytest.raises(ValueError, match="doesn't support min/max aggregation"):
            store.get_metadata_field_min_max("kind")

    def test_rejects_a_field_that_is_not_in_the_schema(self, store):
        store._collection.config.get.return_value = _config()

        with pytest.raises(ValueError, match="not found in collection schema"):
            store.get_metadata_field_min_max("meta.absent")

    def test_returns_nothing_for_an_empty_collection(self, store):
        store._collection.config.get.return_value = _config(_property("year", "int"))
        store._collection.aggregate.over_all.return_value = SimpleNamespace(total_count=0, properties={})

        assert store.get_metadata_field_min_max("year") == {"min": None, "max": None}


class TestGetMetadataFieldUniqueValues:
    def test_returns_the_unique_values(self, store):
        store._collection.config.get.return_value = _config(_property("kind"))
        store._collection.aggregate.over_all.return_value = _groups("book", "paper")

        assert store.get_metadata_field_unique_values("meta.kind") == (["book", "paper"], 2)

    def test_rejects_a_field_that_is_not_in_the_schema(self, store):
        store._collection.config.get.return_value = _config()

        with pytest.raises(ValueError, match="not found in collection schema"):
            store.get_metadata_field_unique_values("absent")

    def test_passes_the_filters_to_the_aggregation(self, store):
        store._collection.config.get.return_value = _config(_property("kind"))
        store._collection.aggregate.over_all.return_value = _groups("book")

        store.get_metadata_field_unique_values("kind", filters={"field": "meta.year", "operator": "==", "value": 2024})

        assert store._collection.aggregate.over_all.call_args.kwargs["filters"] is not None


class TestCountUniqueMetadataByFilter:
    def test_counts_the_groups_per_field(self, store):
        store._collection.config.get.return_value = _config(_property("kind"), _property("year", "int"))
        store._collection.aggregate.over_all.return_value = _groups("book", "paper")

        counts = store.count_unique_metadata_by_filter({}, ["meta.kind", "meta.year"])

        assert counts == {"kind": 2, "year": 2}

    def test_rejects_fields_that_are_not_in_the_schema(self, store):
        store._collection.config.get.return_value = _config(_property("kind"))

        with pytest.raises(ValueError, match="Fields not found in collection schema"):
            store.count_unique_metadata_by_filter({}, ["absent"])


class TestAsyncMetadataPaths:
    """
    The async metadata reads. Their aggregation logic is shared with the sync twins
    above, so only the await plumbing and the schema guards are checked here.
    """

    async def test_count_documents_async(self, async_store):
        async_store._async_collection.aggregate.over_all = AsyncMock(return_value=SimpleNamespace(total_count=3))

        assert await async_store.count_documents_async() == 3

    async def test_count_documents_by_filter_async(self, async_store):
        async_store._async_collection.aggregate.over_all = AsyncMock(return_value=SimpleNamespace(total_count=2))

        assert (
            await async_store.count_documents_by_filter_async({"field": "meta.kind", "operator": "==", "value": "book"})
            == 2
        )

    async def test_get_metadata_fields_info_async(self, async_store):
        async_store._async_collection.config.get = AsyncMock(return_value=_config(_property("year", "int")))

        assert await async_store.get_metadata_fields_info_async() == {"year": {"type": "int"}}

    async def test_get_metadata_field_min_max_async(self, async_store):
        async_store._async_collection.config.get = AsyncMock(return_value=_config(_property("year", "int")))
        async_store._async_collection.aggregate.over_all = AsyncMock(
            return_value=SimpleNamespace(
                total_count=2, properties={"year": SimpleNamespace(minimum=2020, maximum=2024)}
            )
        )

        assert await async_store.get_metadata_field_min_max_async("meta.year") == {"min": 2020, "max": 2024}

    async def test_get_metadata_field_unique_values_async(self, async_store):
        async_store._async_collection.config.get = AsyncMock(return_value=_config(_property("kind")))
        async_store._async_collection.aggregate.over_all = AsyncMock(return_value=_groups("book", "paper"))

        assert await async_store.get_metadata_field_unique_values_async("kind") == (["book", "paper"], 2)

    async def test_count_unique_metadata_by_filter_async(self, async_store):
        async_store._async_collection.config.get = AsyncMock(return_value=_config(_property("kind")))
        async_store._async_collection.aggregate.over_all = AsyncMock(return_value=_groups("book", "paper"))

        assert await async_store.count_unique_metadata_by_filter_async({}, ["kind"]) == {"kind": 2}

    @pytest.mark.parametrize(
        ("method", "args", "error_match"),
        [
            ("get_metadata_field_min_max_async", ("absent",), "not found in collection schema"),
            ("get_metadata_field_unique_values_async", ("absent",), "not found in collection schema"),
            ("count_unique_metadata_by_filter_async", ({}, ["absent"]), "Fields not found in collection schema"),
        ],
    )
    async def test_fields_outside_the_schema_are_rejected(self, async_store, method, args, error_match):
        async_store._async_collection.config.get = AsyncMock(return_value=_config())

        with pytest.raises(ValueError, match=error_match):
            await getattr(async_store, method)(*args)


UUIDS = [f"00000000-0000-4000-8000-00000000000{i}" for i in range(10)]


def _object(properties=None, vector=None, uuid=UUIDS[0], metadata=None) -> SimpleNamespace:
    """A Weaviate result object, as fetch_objects and the iterator return them."""
    return SimpleNamespace(
        properties=properties if properties is not None else {"_original_id": "doc-1", "content": "alpha"},
        vector=vector,
        uuid=uuid,
        metadata=metadata,
    )


def _status_error(message: str) -> weaviate.exceptions.UnexpectedStatusCodeError:
    return weaviate.exceptions.UnexpectedStatusCodeError(message, httpx.Response(500, text=message))


class TestToDataObject:
    def test_moves_the_document_id_out_of_the_way_of_the_weaviate_uuid(self):
        data = WeaviateDocumentStore._to_data_object(Document(id="doc-1", content="alpha"))

        assert data["_original_id"] == "doc-1"
        assert "id" not in data

    def test_the_embedding_is_stored_separately_from_the_properties(self):
        data = WeaviateDocumentStore._to_data_object(Document(content="alpha", embedding=[0.1, 0.2]))

        assert "embedding" not in data

    def test_a_blob_is_split_into_base64_data_and_a_mime_type(self):
        document = Document(content="alpha", blob=ByteStream(data=b"bytes", mime_type="text/plain"))

        data = WeaviateDocumentStore._to_data_object(document)

        assert base64.b64decode(data["blob_data"]) == b"bytes"
        assert data["blob_mime_type"] == "text/plain"
        assert "blob" not in data


class TestToDocument:
    def test_restores_the_original_document_id(self):
        document = WeaviateDocumentStore._to_document(_object())

        assert document.id == "doc-1"
        assert document.content == "alpha"

    @pytest.mark.parametrize(
        ("vector", "expected"),
        [
            ([0.1, 0.2], [0.1, 0.2]),
            ({"default": [0.3]}, [0.3]),
            (None, None),
        ],
        ids=["list", "named-vector", "missing"],
    )
    def test_reads_the_embedding_from_either_vector_shape(self, vector, expected):
        assert WeaviateDocumentStore._to_document(_object(vector=vector)).embedding == expected

    def test_rebuilds_the_blob_from_its_base64_representation(self):
        properties = {
            "_original_id": "doc-1",
            "content": "alpha",
            "blob_data": base64.b64encode(b"bytes").decode(),
            "blob_mime_type": "text/plain",
        }

        document = WeaviateDocumentStore._to_document(_object(properties=properties))

        assert document.blob.data == b"bytes"
        assert document.blob.mime_type == "text/plain"

    def test_datetimes_are_rendered_as_iso_strings(self):
        properties = {
            "_original_id": "doc-1",
            "content": "alpha",
            "published": datetime.datetime(2024, 1, 15, 10, 30, 0, tzinfo=datetime.timezone.utc),
        }

        document = WeaviateDocumentStore._to_document(_object(properties=properties))

        assert document.meta["published"] == "2024-01-15T10:30:00Z"

    def test_the_bm25_score_is_used_when_present(self):
        metadata = SimpleNamespace(score=0.8, certainty=None)

        assert WeaviateDocumentStore._to_document(_object(metadata=metadata)).score == 0.8

    def test_the_embedding_certainty_is_used_when_there_is_no_score(self):
        metadata = SimpleNamespace(score=None, certainty=0.9)

        assert WeaviateDocumentStore._to_document(_object(metadata=metadata)).score == 0.9

    def test_no_score_when_the_query_reported_neither(self):
        metadata = SimpleNamespace(score=None, certainty=None)

        assert WeaviateDocumentStore._to_document(_object(metadata=metadata)).score is None


class TestFilterDocuments:
    def test_without_filters_it_iterates_the_whole_collection(self, store):
        store._collection.config.get.return_value = _config()
        store._collection.iterator.return_value = [_object()]

        documents = store.filter_documents()

        assert [doc.id for doc in documents] == ["doc-1"]

    def test_with_filters_it_pages_through_fetch_objects(self, store):
        store._collection.config.get.return_value = _config()
        store._collection.query.fetch_objects.return_value = SimpleNamespace(objects=[_object()])

        documents = store.filter_documents({"field": "meta.kind", "operator": "==", "value": "book"})

        assert [doc.id for doc in documents] == ["doc-1"]
        assert store._collection.query.fetch_objects.call_args.kwargs["filters"] is not None

    def test_rejects_a_malformed_filter(self, store):
        with pytest.raises(ValueError, match="Invalid filter syntax"):
            store.filter_documents({"field": "meta.kind"})

    def test_a_query_error_becomes_a_document_store_error(self, store):
        store._collection.config.get.return_value = _config()
        store._collection.iterator.side_effect = weaviate.exceptions.WeaviateQueryError("boom", "GRPC")

        with pytest.raises(DocumentStoreError, match="Failed to query documents"):
            store.filter_documents()

    def test_a_filtered_query_error_becomes_a_document_store_error(self, store):
        store._collection.config.get.return_value = _config()
        store._collection.query.fetch_objects.side_effect = weaviate.exceptions.WeaviateQueryError("boom", "GRPC")

        with pytest.raises(DocumentStoreError, match="Failed to query documents"):
            store.filter_documents({"field": "meta.kind", "operator": "==", "value": "book"})


class TestWriteDocuments:
    def test_the_none_policy_writes_in_batches(self, store):
        store._client = MagicMock()
        store._client.batch.failed_objects = []

        written = store.write_documents([Document(id="a", content="alpha")])

        assert written == 1

    def test_skip_leaves_an_existing_document_alone(self, store):
        store._collection.data.exists.return_value = True

        written = store.write_documents([Document(id="a", content="alpha")], policy=DuplicatePolicy.SKIP)

        assert written == 0
        store._collection.data.insert.assert_not_called()

    def test_fail_reports_every_duplicate_id(self, store):
        store._collection.data.exists.return_value = False
        store._collection.data.insert.side_effect = _status_error("boom")

        with pytest.raises(DuplicateDocumentError, match="already exist in the document store"):
            store.write_documents([Document(id="a", content="alpha")], policy=DuplicatePolicy.FAIL)

    @pytest.mark.parametrize("policy", [DuplicatePolicy.NONE, DuplicatePolicy.SKIP])
    def test_only_documents_are_accepted(self, store, policy):
        store._client = MagicMock()
        store._client.batch.failed_objects = []

        with pytest.raises(ValueError, match="Expected a Document"):
            store.write_documents(["not a document"], policy=policy)


class TestDeleteDocuments:
    def test_deletes_by_the_weaviate_uuids_derived_from_the_document_ids(self, store):
        store.delete_documents(["a", "b"])

        store._collection.data.delete_many.assert_called_once()

    def test_delete_by_filter_returns_the_number_deleted(self, store):
        store._collection.data.delete_many.return_value = SimpleNamespace(successful=3)

        assert store.delete_by_filter({"field": "meta.kind", "operator": "==", "value": "book"}) == 3

    def test_delete_by_filter_rejects_a_malformed_filter(self, store):
        with pytest.raises(ValueError, match="Invalid filter syntax"):
            store.delete_by_filter({"field": "meta.kind"})

    def test_delete_all_documents_deletes_the_remaining_ids_in_a_final_batch(self, store):
        store._collection.iterator.return_value = [_object(uuid=UUIDS[0]), _object(uuid=UUIDS[1])]
        store._collection.data.delete_many.return_value = SimpleNamespace(successful=2)

        store.delete_all_documents()

        store._collection.data.delete_many.assert_called_once()

    def test_delete_all_documents_deletes_in_batches_of_the_given_size(self, store):
        store._collection.iterator.return_value = [_object(uuid=UUIDS[i]) for i in range(4)]
        store._collection.data.delete_many.return_value = SimpleNamespace(successful=2)

        store.delete_all_documents(batch_size=2)

        assert store._collection.data.delete_many.call_count == 2

    def test_delete_all_documents_warns_when_a_batch_is_only_partly_deleted(self, store, caplog):
        store._collection.iterator.return_value = [_object(uuid=UUIDS[0]), _object(uuid=UUIDS[1])]
        store._collection.data.delete_many.return_value = SimpleNamespace(successful=1)

        store.delete_all_documents(batch_size=2)

        assert "Not all documents" in caplog.text

    def test_delete_all_documents_can_recreate_the_collection(self, store):
        store._client = MagicMock()
        store._client.collections.get.return_value.config.get.return_value.to_dict.return_value = {"class": "Default"}

        store.delete_all_documents(recreate_index=True)

        store._client.collections.delete.assert_called_once_with("Default")
        store._client.collections.create_from_dict.assert_called_once_with({"class": "Default"})


class TestUpdateByFilter:
    def test_merges_the_new_metadata_into_the_matching_objects(self, store):
        store._collection.config.get.return_value = _config()
        store._collection.query.fetch_objects.return_value = SimpleNamespace(
            objects=[_object(properties={"_original_id": "doc-1", "content": "alpha"}, vector=[0.1])]
        )

        updated = store.update_by_filter({"field": "meta.kind", "operator": "==", "value": "book"}, {"seen": True})

        assert updated == 1
        kwargs = store._collection.data.replace.call_args.kwargs
        assert kwargs["properties"]["seen"] is True
        assert kwargs["vector"] == [0.1]

    def test_a_filter_that_matches_nothing_updates_nothing(self, store):
        store._collection.config.get.return_value = _config()
        store._collection.query.fetch_objects.return_value = SimpleNamespace(objects=[])

        assert store.update_by_filter({"field": "meta.kind", "operator": "==", "value": "book"}, {}) == 0

    def test_requires_the_metadata_to_be_a_dictionary(self, store):
        with pytest.raises(ValueError, match="Meta must be a dictionary"):
            store.update_by_filter({"field": "meta.kind", "operator": "==", "value": "book"}, "not a dict")


class TestRetrieval:
    def test_bm25_retrieval_passes_the_query_and_top_k(self, store):
        store._collection.config.get.return_value = _config()
        store._collection.query.bm25.return_value = SimpleNamespace(objects=[_object()])

        documents = store._bm25_retrieval("a query", top_k=5)

        assert [doc.id for doc in documents] == ["doc-1"]
        kwargs = store._collection.query.bm25.call_args.kwargs
        assert kwargs["query"] == "a query"
        assert kwargs["limit"] == 5
        assert kwargs["filters"] is None

    def test_embedding_retrieval_passes_the_vector(self, store):
        store._collection.config.get.return_value = _config()
        store._collection.query.near_vector.return_value = SimpleNamespace(objects=[_object()])

        documents = store._embedding_retrieval([0.1, 0.2], top_k=3)

        assert [doc.id for doc in documents] == ["doc-1"]
        assert store._collection.query.near_vector.call_args.kwargs["near_vector"] == [0.1, 0.2]

    def test_hybrid_retrieval_passes_both_the_query_and_the_vector(self, store):
        store._collection.config.get.return_value = _config()
        store._collection.query.hybrid.return_value = SimpleNamespace(objects=[_object()])

        documents = store._hybrid_retrieval("a query", [0.1, 0.2], top_k=4)

        assert [doc.id for doc in documents] == ["doc-1"]
        kwargs = store._collection.query.hybrid.call_args.kwargs
        assert kwargs["query"] == "a query"
        assert kwargs["vector"] == [0.1, 0.2]


class TestAsyncDocumentPaths:
    """
    The async paths whose mechanics differ from their synchronous twin.

    The rest of the async surface is a line-by-line mirror of the sync code above
    — same validation, same error wrapping — so it is covered there instead.
    """

    @pytest.fixture
    def collection(self, async_store):
        collection = async_store._async_collection
        collection.config.get = AsyncMock(return_value=_config())
        collection.query.fetch_objects = AsyncMock(return_value=SimpleNamespace(objects=[_object()]))
        collection.query.bm25 = AsyncMock(return_value=SimpleNamespace(objects=[_object()]))
        collection.query.near_vector = AsyncMock(return_value=SimpleNamespace(objects=[_object()]))
        collection.query.hybrid = AsyncMock(return_value=SimpleNamespace(objects=[_object()]))
        collection.data.delete_many = AsyncMock(return_value=SimpleNamespace(successful=1))
        collection.data.replace = AsyncMock()
        collection.data.insert = AsyncMock()
        collection.data.exists = AsyncMock(return_value=False)
        collection.name = "Default"
        return collection

    async def test_filter_documents_async_iterates_the_collection_asynchronously(self, async_store, collection):
        async def _iterate(**_):
            yield _object()

        collection.iterator = _iterate

        documents = await async_store.filter_documents_async()

        assert [doc.id for doc in documents] == ["doc-1"]

    async def test_filter_documents_async_with_filters_pages_through_fetch_objects(self, async_store, collection):
        documents = await async_store.filter_documents_async({"field": "meta.kind", "operator": "==", "value": "book"})

        assert [doc.id for doc in documents] == ["doc-1"]
        assert collection.query.fetch_objects.call_args.kwargs["filters"] is not None

    @pytest.mark.usefixtures("collection")
    async def test_writing_uses_the_streaming_batch_api(self, async_store):
        # The async client streams its batch, where the sync one uses a dynamic batch.
        client = MagicMock()
        client.batch.failed_objects = []
        batch = AsyncMock()
        client.batch.stream.return_value.__aenter__ = AsyncMock(return_value=batch)
        client.batch.stream.return_value.__aexit__ = AsyncMock(return_value=False)
        async_store._async_client = client

        assert await async_store.write_documents_async([Document(id="a", content="alpha")]) == 1
        batch.add_object.assert_awaited_once()

    async def test_delete_all_documents_async_iterates_and_deletes_the_collected_ids(self, async_store, collection):
        async_store._async_client = MagicMock()

        async def _iterate(**_):
            yield _object(uuid=UUIDS[0])

        collection.iterator = _iterate

        await async_store.delete_all_documents_async()

        collection.data.delete_many.assert_awaited_once()

    async def test_update_by_filter_async_merges_the_new_metadata(self, async_store, collection):
        collection.query.fetch_objects = AsyncMock(return_value=SimpleNamespace(objects=[_object(vector=[0.1])]))

        updated = await async_store.update_by_filter_async(
            {"field": "meta.kind", "operator": "==", "value": "book"}, {"seen": True}
        )

        assert updated == 1
        kwargs = collection.data.replace.call_args.kwargs
        assert kwargs["properties"]["seen"] is True
        assert kwargs["vector"] == [0.1]

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("_bm25_retrieval_async", ("a query",)),
            ("_embedding_retrieval_async", ([0.1, 0.2],)),
            ("_hybrid_retrieval_async", ("a query", [0.1, 0.2])),
        ],
    )
    @pytest.mark.usefixtures("collection")
    async def test_every_retrieval_method_converts_the_objects_it_gets_back(self, async_store, method, args):
        documents = await getattr(async_store, method)(*args, top_k=3)

        assert [doc.id for doc in documents] == ["doc-1"]
