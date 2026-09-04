# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError

from haystack_integrations.document_stores.faiss import FAISSDocumentStore


@pytest.fixture
def store() -> FAISSDocumentStore:
    """An in-memory store: FAISS runs in-process, so no external service is involved."""
    return FAISSDocumentStore(embedding_dim=3)


def _equals(field: str, value: object) -> dict:
    return {"field": field, "operator": "==", "value": value}


class TestGetDocValue:
    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("content", "the content"),
            ("id", "doc-1"),
            ("meta.year", 2024),
            ("meta.missing", None),
            # Not a "meta." path and not a Document attribute, so it falls back to a meta lookup.
            ("year", 2024),
            ("unknown", None),
        ],
    )
    def test_reads_the_field_from_the_right_place(self, field, expected):
        doc = Document(id="doc-1", content="the content", meta={"year": 2024})

        assert FAISSDocumentStore._get_doc_value(doc, field) == expected


class TestCheckCondition:
    @pytest.fixture
    def store_with_docs(self, store) -> FAISSDocumentStore:
        store.write_documents(
            [
                Document(id="a", content="alpha", meta={"year": 2020, "kind": "book"}),
                Document(id="b", content="beta", meta={"year": 2024, "kind": "paper"}),
                Document(id="c", content="gamma", meta={"year": 2022, "kind": "book"}),
            ]
        )
        return store

    @pytest.mark.parametrize(
        ("operator", "value", "expected_ids"),
        [
            ("==", 2020, ["a"]),
            ("!=", 2020, ["b", "c"]),
            (">", 2020, ["b", "c"]),
            (">=", 2022, ["b", "c"]),
            ("<", 2022, ["a"]),
            ("<=", 2022, ["a", "c"]),
            ("in", [2020, 2024], ["a", "b"]),
            ("not in", [2020, 2024], ["c"]),
        ],
    )
    def test_comparison_operators(self, store_with_docs, operator, value, expected_ids):
        docs = store_with_docs.filter_documents({"field": "meta.year", "operator": operator, "value": value})

        assert sorted(doc.id for doc in docs) == expected_ids

    def test_and_requires_every_condition_to_match(self, store_with_docs):
        docs = store_with_docs.filter_documents(
            {
                "operator": "AND",
                "conditions": [_equals("meta.kind", "book"), {"field": "meta.year", "operator": ">", "value": 2021}],
            }
        )

        assert [doc.id for doc in docs] == ["c"]

    def test_or_requires_any_condition_to_match(self, store_with_docs):
        docs = store_with_docs.filter_documents(
            {"operator": "OR", "conditions": [_equals("meta.year", 2020), _equals("meta.year", 2024)]}
        )

        assert sorted(doc.id for doc in docs) == ["a", "b"]

    def test_not_inverts_the_conjunction_of_its_conditions(self, store_with_docs):
        docs = store_with_docs.filter_documents({"operator": "NOT", "conditions": [_equals("meta.kind", "book")]})

        assert [doc.id for doc in docs] == ["b"]

    def test_comparisons_never_match_a_document_missing_the_field(self, store_with_docs):
        docs = store_with_docs.filter_documents({"field": "meta.absent", "operator": ">", "value": 1})

        assert docs == []

    @pytest.mark.parametrize(
        ("condition", "error_match"),
        [
            ({"field": "meta.year", "value": 2020}, "missing 'operator'"),
            ({"operator": "AND"}, "Missing 'conditions' for AND"),
            ({"operator": "OR"}, "Missing 'conditions' for OR"),
            ({"operator": "NOT"}, "Missing 'conditions' for NOT"),
            ({"operator": "NOT", "conditions": []}, "at least one condition"),
            ({"operator": "==", "value": 1}, "Missing 'field'"),
            ({"operator": "==", "field": 1, "value": 1}, "must be a string"),
            ({"operator": "==", "field": "meta.year"}, "Missing 'value'"),
            ({"field": "meta.year", "operator": ">", "value": "a string"}, "Type mismatch"),
            ({"field": "meta.year", "operator": "in", "value": 2020}, "must be a list"),
            ({"field": "meta.year", "operator": "not in", "value": 2020}, "must be a list"),
        ],
    )
    def test_malformed_conditions_raise_a_filter_error(self, store_with_docs, condition, error_match):
        with pytest.raises(FilterError, match=error_match):
            store_with_docs.filter_documents(condition)


class TestWriteDocuments:
    def test_overwrite_replaces_the_document_and_its_embedding(self, store):
        store.write_documents([Document(id="a", content="first", embedding=[0.1, 0.2, 0.3])])

        written = store.write_documents(
            [Document(id="a", content="second", embedding=[0.9, 0.9, 0.9])],
            policy=DuplicatePolicy.OVERWRITE,
        )

        assert written == 1
        assert store.count_documents() == 1
        assert store.filter_documents()[0].content == "second"
        assert store.index.ntotal == 1

    def test_skip_keeps_the_existing_document(self, store):
        store.write_documents([Document(id="a", content="first")])

        written = store.write_documents(
            [Document(id="a", content="second"), Document(id="b", content="new")],
            policy=DuplicatePolicy.SKIP,
        )

        assert written == 1
        assert store.filter_documents(_equals("id", "a"))[0].content == "first"

    def test_documents_without_embeddings_are_stored_but_not_indexed(self, store):
        store.write_documents([Document(id="a", content="no embedding")])

        assert store.count_documents() == 1
        assert store.index.ntotal == 0


class TestDeleteDocuments:
    def test_removes_the_document_and_its_embedding(self, store):
        store.write_documents(
            [
                Document(id="a", content="alpha", embedding=[0.1, 0.2, 0.3]),
                Document(id="b", content="beta", embedding=[0.4, 0.5, 0.6]),
            ]
        )

        store.delete_documents(["a"])

        assert store.count_documents() == 1
        assert store.index.ntotal == 1
        assert "a" not in store.inverse_id_map

    def test_delete_all_documents_also_resets_the_index_and_the_id_counter(self, store):
        store.write_documents([Document(id="a", content="alpha", embedding=[0.1, 0.2, 0.3])])

        store.delete_all_documents()

        assert store.count_documents() == 0
        assert store.index.ntotal == 0
        assert store._next_id == 0


class TestSearch:
    @pytest.fixture
    def store_with_embeddings(self, store) -> FAISSDocumentStore:
        store.write_documents(
            [
                Document(id="a", content="alpha", embedding=[1.0, 0.0, 0.0], meta={"kind": "book"}),
                Document(id="b", content="beta", embedding=[0.0, 1.0, 0.0], meta={"kind": "paper"}),
                Document(id="c", content="gamma", embedding=[0.0, 0.0, 1.0], meta={"kind": "book"}),
            ]
        )
        return store

    def test_returns_the_nearest_documents_first(self, store_with_embeddings):
        docs = store_with_embeddings.search(query_embedding=[1.0, 0.0, 0.0], top_k=3)

        assert [doc.id for doc in docs] == ["a", "b", "c"] or docs[0].id == "a"
        assert docs[0].id == "a"

    def test_scores_a_flat_index_as_the_inverse_of_the_distance(self, store_with_embeddings):
        docs = store_with_embeddings.search(query_embedding=[1.0, 0.0, 0.0], top_k=1)

        # An exact match has distance 0, so the score is 1 / (1 + 0).
        assert docs[0].score == pytest.approx(1.0)

    def test_applies_filters_after_retrieval(self, store_with_embeddings):
        docs = store_with_embeddings.search(
            query_embedding=[1.0, 0.0, 0.0], top_k=3, filters=_equals("meta.kind", "book")
        )

        assert sorted(doc.id for doc in docs) == ["a", "c"]

    def test_returns_nothing_when_the_index_is_empty(self, store):
        assert store.search(query_embedding=[1.0, 0.0, 0.0]) == []


class TestMetadataOperations:
    @pytest.fixture
    def store_with_docs(self, store) -> FAISSDocumentStore:
        store.write_documents(
            [
                Document(id="a", content="alpha", meta={"year": 2020, "kind": "book", "public": True}),
                Document(id="b", content="beta", meta={"year": 2024, "kind": "paper", "public": False}),
                Document(id="c", content="gamma", meta={"year": 2022, "kind": "book", "rating": 4.5}),
            ]
        )
        return store

    def test_get_metadata_fields_info_maps_python_types_to_search_types(self, store_with_docs):
        assert store_with_docs.get_metadata_fields_info() == {
            "year": {"type": "long"},
            "kind": {"type": "keyword"},
            "public": {"type": "boolean"},
            "rating": {"type": "float"},
        }

    def test_get_metadata_field_min_max(self, store_with_docs):
        assert store_with_docs.get_metadata_field_min_max("meta.year") == {"min": 2020, "max": 2024}

    def test_get_metadata_field_unique_values(self, store_with_docs):
        values, total = store_with_docs.get_metadata_field_unique_values("meta.kind")

        assert values == ["book", "paper"]
        assert total == 2

    def test_count_documents_by_filter(self, store_with_docs):
        assert store_with_docs.count_documents_by_filter(_equals("meta.kind", "book")) == 2

    def test_count_unique_metadata_by_filter(self, store_with_docs):
        counts = store_with_docs.count_unique_metadata_by_filter(
            filters=_equals("meta.kind", "book"), metadata_fields=["meta.year", "meta.kind"]
        )

        assert counts == {"meta.year": 2, "meta.kind": 1}

    def test_delete_by_filter_returns_the_number_deleted(self, store_with_docs):
        deleted = store_with_docs.delete_by_filter(_equals("meta.kind", "book"))

        assert deleted == 2
        assert store_with_docs.count_documents() == 1

    def test_update_by_filter_merges_the_new_metadata(self, store_with_docs):
        updated = store_with_docs.update_by_filter(filters=_equals("meta.kind", "book"), meta={"reviewed": True})

        assert updated == 2
        books = store_with_docs.filter_documents(_equals("meta.kind", "book"))
        assert all(doc.meta["reviewed"] for doc in books)
        # Existing metadata survives the merge.
        assert {doc.meta["year"] for doc in books} == {2020, 2022}


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        path = tmp_path / "index"
        store = FAISSDocumentStore(embedding_dim=3)
        store.write_documents([Document(id="a", content="alpha", embedding=[0.1, 0.2, 0.3])])

        store.save(path)

        loaded = FAISSDocumentStore(embedding_dim=3)
        loaded.load(path)

        assert loaded.count_documents() == 1
        assert loaded.filter_documents()[0].content == "alpha"
        assert loaded.index.ntotal == 1
        assert loaded._next_id == store._next_id

    def test_loading_a_missing_index_raises(self, tmp_path):
        with pytest.raises(ValueError, match="File not found"):
            FAISSDocumentStore(embedding_dim=3).load(tmp_path / "missing")


class TestSerialization:
    def test_to_dict_and_from_dict_round_trip(self):
        store = FAISSDocumentStore(index_path=None, index_string="Flat", embedding_dim=3)

        data = store.to_dict()
        assert data["init_parameters"] == {
            "index_path": None,
            "index_string": "Flat",
            "embedding_dim": 3,
        }

        deserialized = FAISSDocumentStore.from_dict(data)
        assert deserialized.index_string == "Flat"
        assert deserialized.embedding_dim == 3
