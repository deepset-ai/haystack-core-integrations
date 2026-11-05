# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
from haystack.dataclasses import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.chroma import ChromaDocumentStore


def _get_result(ids=None, documents=None, metadatas=None, embeddings=None) -> dict:
    """A Chroma `get` result, whose fields are flat lists."""
    return {
        "ids": ids if ids is not None else [],
        "documents": documents if documents is not None else [],
        "metadatas": metadatas if metadatas is not None else [],
        "embeddings": embeddings,
    }


@pytest.fixture
def store() -> ChromaDocumentStore:
    """
    A store with its collection injected.

    Both `_ensure_initialized` and `_ensure_initialized_async` short-circuit once a
    collection is set, so no Chroma client is ever created.
    """
    store = ChromaDocumentStore(collection_name="test")
    store._collection = mock.MagicMock()
    store._client = mock.MagicMock()
    store._collection.get.return_value = _get_result()
    return store


@pytest.fixture
def async_store() -> ChromaDocumentStore:
    store = ChromaDocumentStore(collection_name="test", host="localhost", port=8000)
    store._async_collection = mock.MagicMock()
    store._async_client = mock.MagicMock()
    store._async_collection.get = mock.AsyncMock(return_value=_get_result())
    store._async_collection.count = mock.AsyncMock(return_value=0)
    store._async_collection.delete = mock.AsyncMock()
    store._async_collection.update = mock.AsyncMock()
    store._async_collection.add = mock.AsyncMock()
    store._async_collection.upsert = mock.AsyncMock()
    store._async_collection.query = mock.AsyncMock()
    return store


class TestPrepareGetKwargs:
    def test_without_filters_only_asks_for_the_standard_fields(self):
        assert ChromaDocumentStore._prepare_get_kwargs() == {"include": ["embeddings", "documents", "metadatas"]}

    def test_a_metadata_filter_becomes_a_where_clause(self):
        kwargs = ChromaDocumentStore._prepare_get_kwargs({"field": "meta.year", "operator": "==", "value": 2024})

        assert kwargs["where"] == {"year": {"$eq": 2024}}
        assert "ids" not in kwargs
        assert "where_document" not in kwargs

    def test_an_id_filter_becomes_an_ids_clause(self):
        kwargs = ChromaDocumentStore._prepare_get_kwargs({"field": "id", "operator": "==", "value": "doc-1"})

        assert kwargs["ids"] == ["doc-1"]

    def test_a_content_filter_becomes_a_where_document_clause(self):
        kwargs = ChromaDocumentStore._prepare_get_kwargs({"field": "content", "operator": "contains", "value": "text"})

        assert kwargs["where_document"] == {"$contains": "text"}


class TestPrepareQueryKwargs:
    def test_without_filters_asks_for_distances_too(self):
        assert ChromaDocumentStore._prepare_query_kwargs() == {
            "include": ["embeddings", "documents", "metadatas", "distances"]
        }

    def test_with_filters_carries_both_clauses(self):
        kwargs = ChromaDocumentStore._prepare_query_kwargs({"field": "meta.year", "operator": "==", "value": 2024})

        assert kwargs["where"] == {"year": {"$eq": 2024}}
        assert kwargs["where_document"] is None
        assert "distances" in kwargs["include"]


class TestInferTypeFromValue:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(True, "boolean"), (1, "long"), (1.5, "float"), ("text", "keyword")],
    )
    def test_maps_python_types_to_search_types(self, value, expected):
        assert ChromaDocumentStore._infer_type_from_value(value) == expected


class TestBuildFieldsInfo:
    def test_infers_a_type_for_every_field(self):
        metadatas = [{"year": 2024, "kind": "book"}, {"public": True, "rating": 4.5}]

        assert ChromaDocumentStore._build_fields_info(metadatas) == {
            "year": {"type": "long"},
            "kind": {"type": "keyword"},
            "public": {"type": "boolean"},
            "rating": {"type": "float"},
        }

    @pytest.mark.parametrize("metadatas", [None, [], [None]], ids=["none", "empty", "all-none"])
    def test_returns_nothing_without_usable_metadata(self, metadatas):
        assert ChromaDocumentStore._build_fields_info(metadatas) == {}


class TestPrepareMetadataUpdate:
    def test_merges_the_new_metadata_into_each_document(self):
        docs = [
            Document(id="a", content="alpha", meta={"year": 2020}),
            Document(id="b", content="beta", meta={"year": 2024}),
        ]

        ids, metadatas = ChromaDocumentStore._prepare_metadata_update(docs, {"reviewed": True})

        assert ids == ["a", "b"]
        assert metadatas == [{"year": 2020, "reviewed": True}, {"year": 2024, "reviewed": True}]

    def test_unsupported_metadata_types_are_dropped(self):
        _, metadatas = ChromaDocumentStore._prepare_metadata_update(
            [Document(id="a", meta={"year": 2020})], {"nested": {"a": 1}}
        )

        assert metadatas == [{"year": 2020}]


PAYLOADS = [{"ids": ["a"]}, {"ids": ["b"]}]


class TestApplyDuplicatePolicy:
    def test_fail_raises_when_an_id_already_exists(self):
        with pytest.raises(DuplicateDocumentError, match="already exist"):
            ChromaDocumentStore._apply_duplicate_policy(PAYLOADS, {"a"}, DuplicatePolicy.FAIL)

    def test_skip_drops_the_payloads_that_already_exist(self):
        assert ChromaDocumentStore._apply_duplicate_policy(PAYLOADS, {"a"}, DuplicatePolicy.SKIP) == [{"ids": ["b"]}]


class TestConvertDocumentToChroma:
    def test_maps_content_id_embedding_and_metadata(self):
        doc = Document(id="a", content="alpha", meta={"year": 2024}, embedding=[0.1, 0.2])

        data = ChromaDocumentStore._convert_document_to_chroma(doc)

        assert data["ids"] == ["a"]
        assert data["documents"] == ["alpha"]
        assert data["metadatas"] == [{"year": 2024}]
        assert data["embeddings"] == [[0.1, 0.2]]

    def test_skips_a_document_without_content(self, caplog):
        assert ChromaDocumentStore._convert_document_to_chroma(Document(id="a")) is None
        assert "content=None" in caplog.text

    def test_keeps_a_list_of_values_of_one_supported_type(self):
        data = ChromaDocumentStore._convert_document_to_chroma(
            Document(id="a", content="alpha", meta={"tags": ["x", "y"]})
        )

        assert data["metadatas"] == [{"tags": ["x", "y"]}]

    @pytest.mark.parametrize(
        "value",
        [[], ["x", 1], {"nested": 1}],
        ids=["empty-list", "mixed-list", "dict"],
    )
    def test_drops_metadata_chroma_cannot_store(self, value, caplog):
        data = ChromaDocumentStore._convert_document_to_chroma(
            Document(id="a", content="alpha", meta={"ok": 1, "bad": value})
        )

        assert data["metadatas"] == [{"ok": 1}]
        assert "bad" in caplog.text

    def test_a_blob_is_ignored_with_a_warning(self, caplog):
        doc = Document(id="a", content="alpha", blob=ByteStream(data=b"bytes"))

        data = ChromaDocumentStore._convert_document_to_chroma(doc)

        assert data["documents"] == ["alpha"]
        assert "blob" in caplog.text


class TestInit:
    def test_rejects_an_unknown_distance_function(self):
        with pytest.raises(ValueError, match="Invalid distance_function"):
            ChromaDocumentStore(distance_function="not-a-distance")


class TestCollectionBackedReads:
    def test_count_documents(self, store):
        store._collection.count.return_value = 3

        assert store.count_documents() == 3

    def test_filter_documents_converts_the_result(self, store):
        store._collection.get.return_value = _get_result(ids=["a"], documents=["alpha"], metadatas=[{"year": 2024}])

        docs = store.filter_documents()

        assert [doc.id for doc in docs] == ["a"]
        assert docs[0].meta == {"year": 2024}

    def test_filter_documents_passes_the_filters_to_the_collection(self, store):
        store.filter_documents({"field": "meta.year", "operator": "==", "value": 2024})

        assert store._collection.get.call_args.kwargs["where"] == {"year": {"$eq": 2024}}

    def test_count_unique_metadata_by_filter(self, store):
        store._collection.get.return_value = _get_result(
            ids=["a", "b"], documents=["alpha", "beta"], metadatas=[{"kind": "book"}, {"kind": "book"}]
        )

        assert store.count_unique_metadata_by_filter({}, ["kind"]) == {"kind": 1}

    def test_get_metadata_fields_info(self, store):
        store._collection.get.return_value = _get_result(ids=["a"], documents=["alpha"], metadatas=[{"year": 2024}])

        assert store.get_metadata_fields_info() == {"year": {"type": "long"}}

    def test_get_metadata_field_min_max(self, store):
        store._collection.get.return_value = _get_result(
            ids=["a", "b"], documents=["alpha", "beta"], metadatas=[{"year": 2020}, {"year": 2024}]
        )

        assert store.get_metadata_field_min_max("meta.year") == {"min": 2020, "max": 2024}

    def test_get_metadata_field_unique_values(self, store):
        store._collection.get.return_value = _get_result(
            ids=["a", "b"], documents=["alpha", "beta"], metadatas=[{"kind": "book"}, {"kind": "paper"}]
        )

        values, total = store.get_metadata_field_unique_values("meta.kind")

        assert values == ["book", "paper"]
        assert total == 2

    def test_search_forwards_the_query_and_top_k(self, store):
        store._collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}

        store.search(["a query"], top_k=5)

        kwargs = store._collection.query.call_args.kwargs
        assert kwargs["query_texts"] == ["a query"]
        assert kwargs["n_results"] == 5


class TestCollectionBackedWrites:
    def test_write_documents_adds_each_document(self, store):
        written = store.write_documents([Document(id="a", content="alpha")])

        assert written == 1
        store._collection.add.assert_called_once()

    def test_write_documents_upserts_when_overwriting(self, store):
        store.write_documents([Document(id="a", content="alpha")], policy=DuplicatePolicy.OVERWRITE)

        store._collection.upsert.assert_called_once()

    def test_write_documents_skips_the_documents_that_already_exist(self, store):
        store._collection.get.return_value = _get_result(ids=["a"])

        written = store.write_documents(
            [Document(id="a", content="alpha"), Document(id="b", content="beta")], policy=DuplicatePolicy.SKIP
        )

        assert written == 1

    def test_write_documents_fails_on_a_duplicate(self, store):
        store._collection.get.return_value = _get_result(ids=["a"])

        with pytest.raises(DuplicateDocumentError):
            store.write_documents([Document(id="a", content="alpha")], policy=DuplicatePolicy.FAIL)

    def test_delete_documents(self, store):
        store.delete_documents(["a", "b"])

        store._collection.delete.assert_called_once_with(ids=["a", "b"])

    def test_delete_by_filter_deletes_by_id_when_the_filter_selects_ids(self, store):
        store._collection.get.return_value = _get_result(ids=["a"], documents=["alpha"])

        deleted = store.delete_by_filter({"field": "id", "operator": "==", "value": "a"})

        assert deleted == 1
        assert store._collection.delete.call_args.kwargs == {"ids": ["a"]}

    def test_delete_by_filter_deletes_by_where_clause_otherwise(self, store):
        store._collection.get.return_value = _get_result(ids=["a"], documents=["alpha"])

        deleted = store.delete_by_filter({"field": "meta.year", "operator": "==", "value": 2024})

        assert deleted == 1
        assert store._collection.delete.call_args.kwargs == {"where": {"year": {"$eq": 2024}}}

    def test_delete_by_filter_wraps_backend_errors(self, store):
        store._collection.get.side_effect = RuntimeError("chroma is down")

        with pytest.raises(DocumentStoreError, match="Failed to delete documents by filter"):
            store.delete_by_filter({"field": "meta.year", "operator": "==", "value": 2024})

    def test_update_by_filter_updates_the_matching_documents(self, store):
        store._collection.get.return_value = _get_result(ids=["a"], documents=["alpha"], metadatas=[{"year": 2020}])

        updated = store.update_by_filter({"field": "meta.year", "operator": "==", "value": 2020}, {"seen": True})

        assert updated == 1
        kwargs = store._collection.update.call_args.kwargs
        assert kwargs["ids"] == ["a"]
        assert kwargs["metadatas"] == [{"year": 2020, "seen": True}]

    def test_update_by_filter_wraps_backend_errors(self, store):
        store._collection.get.side_effect = RuntimeError("chroma is down")

        with pytest.raises(DocumentStoreError, match="Failed to update documents by filter"):
            store.update_by_filter({"field": "meta.year", "operator": "==", "value": 1}, {})

    def test_delete_all_documents_deletes_every_id(self, store):
        store._collection.get.return_value = _get_result(ids=["a", "b"])

        store.delete_all_documents()

        store._collection.delete.assert_called_once_with(ids=["a", "b"])

    def test_delete_all_documents_can_recreate_the_collection(self, store):
        store.delete_all_documents(recreate_index=True)

        store._client.delete_collection.assert_called_once_with(name="test")
        store._client.create_collection.assert_called_once()

    def test_delete_all_documents_wraps_backend_errors(self, store):
        store._collection.get.side_effect = RuntimeError("chroma is down")

        with pytest.raises(DocumentStoreError, match="Failed to delete all documents"):
            store.delete_all_documents()


class TestAsyncPaths:
    @pytest.mark.asyncio
    async def test_write_documents_async(self, async_store):
        written = await async_store.write_documents_async([Document(id="a", content="alpha")])

        assert written == 1
        async_store._async_collection.add.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_by_filter_async(self, async_store):
        async_store._async_collection.get = mock.AsyncMock(
            return_value=_get_result(ids=["a"], documents=["alpha"], metadatas=[{"year": 2020}])
        )

        updated = await async_store.update_by_filter_async(
            {"field": "meta.year", "operator": "==", "value": 2020}, {"seen": True}
        )

        assert updated == 1
        assert async_store._async_collection.update.call_args.kwargs["metadatas"] == [{"year": 2020, "seen": True}]

    @pytest.mark.asyncio
    async def test_delete_by_filter_async_wraps_backend_errors(self, async_store):
        async_store._async_collection.get = mock.AsyncMock(side_effect=RuntimeError("chroma is down"))

        with pytest.raises(DocumentStoreError, match="Failed to delete documents by filter"):
            await async_store.delete_by_filter_async({"field": "meta.year", "operator": "==", "value": 1})

    @pytest.mark.asyncio
    async def test_update_by_filter_async_wraps_backend_errors(self, async_store):
        async_store._async_collection.get = mock.AsyncMock(side_effect=RuntimeError("chroma is down"))

        with pytest.raises(DocumentStoreError, match="Failed to update documents by filter"):
            await async_store.update_by_filter_async({"field": "meta.year", "operator": "==", "value": 1}, {})

    @pytest.mark.asyncio
    async def test_delete_all_documents_async_wraps_backend_errors(self, async_store):
        async_store._async_collection.get = mock.AsyncMock(side_effect=RuntimeError("chroma is down"))

        with pytest.raises(DocumentStoreError, match="Failed to delete all documents"):
            await async_store.delete_all_documents_async()

    @pytest.mark.asyncio
    async def test_search_async(self, async_store):
        async_store._async_collection.query = mock.AsyncMock(
            return_value={"ids": [[]], "documents": [[]], "metadatas": [[]]}
        )

        await async_store.search_async(["a query"], top_k=2)

        assert async_store._async_collection.query.call_args.kwargs["n_results"] == 2


class TestAsyncSurface:
    """
    Every async method has its own body rather than delegating to the sync one, so
    each needs to be executed. The behaviour is asserted by the sync tests above;
    these check the await plumbing and that the collection is driven the same way.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "args", "result", "expected"),
        [
            ("count_documents_by_filter_async", ({},), _get_result(ids=["a"], documents=["alpha"]), 1),
            (
                "count_unique_metadata_by_filter_async",
                ({}, ["kind"]),
                _get_result(ids=["a"], documents=["alpha"], metadatas=[{"kind": "book"}]),
                {"kind": 1},
            ),
            (
                "get_metadata_fields_info_async",
                (),
                _get_result(ids=["a"], documents=["alpha"], metadatas=[{"year": 2024}]),
                {"year": {"type": "long"}},
            ),
            (
                "get_metadata_field_min_max_async",
                ("meta.year",),
                _get_result(ids=["a"], documents=["alpha"], metadatas=[{"year": 2024}]),
                {"min": 2024, "max": 2024},
            ),
            (
                "get_metadata_field_unique_values_async",
                ("meta.kind",),
                _get_result(ids=["a"], documents=["alpha"], metadatas=[{"kind": "book"}]),
                (["book"], 1),
            ),
        ],
    )
    async def test_the_async_reads_return_what_the_collection_gave_them(
        self, async_store, method, args, result, expected
    ):
        async_store._async_collection.get = mock.AsyncMock(return_value=result)

        assert await getattr(async_store, method)(*args) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "args", "collection_method"),
        [
            ("delete_documents_async", (["a"],), "delete"),
            ("write_documents_async", ([Document(id="a", content="alpha")],), "add"),
        ],
    )
    async def test_the_async_writes_reach_the_collection(self, async_store, method, args, collection_method):
        await getattr(async_store, method)(*args)

        getattr(async_store._async_collection, collection_method).assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_by_filter_async_counts_what_it_deleted(self, async_store):
        async_store._async_collection.get = mock.AsyncMock(return_value=_get_result(ids=["a"], documents=["alpha"]))

        assert await async_store.delete_by_filter_async({"field": "id", "operator": "==", "value": "a"}) == 1

    @pytest.mark.asyncio
    async def test_delete_all_documents_async_deletes_every_id(self, async_store):
        async_store._async_collection.get = mock.AsyncMock(return_value=_get_result(ids=["a"]))

        await async_store.delete_all_documents_async()

        async_store._async_collection.delete.assert_awaited_once_with(ids=["a"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "args"), [("search_async", (["a query"],)), ("search_embeddings_async", ([[0.1]],))]
    )
    async def test_the_async_searches_query_the_collection(self, async_store, method, args):
        async_store._async_collection.query = mock.AsyncMock(
            return_value={"ids": [[]], "documents": [[]], "metadatas": [[]]}
        )

        await getattr(async_store, method)(*args, top_k=2)

        assert async_store._async_collection.query.call_args.kwargs["n_results"] == 2
