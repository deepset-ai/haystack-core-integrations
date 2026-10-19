# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from haystack.dataclasses import ByteStream, Document

from haystack_integrations.document_stores.solr.schema import (
    ALL_TYPE_CODES,
    JSON_TYPE_CODE,
    document_to_solr,
    meta_field_name,
    parse_meta_field_name,
    schema_payload,
    solr_to_document,
    type_code_for_value,
    validate_meta_keys,
    vector_field_type_name,
)


class TestTypeCodes:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("text", "s"),
            (True, "b"),
            (False, "b"),
            (7, "l"),
            (-7, "l"),
            (1.5, "d"),
            (["a", "b"], "ss"),
            ([1, 2], "ls"),
            ([1.5], "ds"),
            ([True], "bs"),
            # Anything Solr has no type for falls back to a JSON-encoded string.
            ({"nested": 1}, JSON_TYPE_CODE),
            ([1, "mixed"], JSON_TYPE_CODE),
            ([], JSON_TYPE_CODE),
            ([[1]], JSON_TYPE_CODE),
        ],
    )
    def test_type_code_for_value(self, value, expected):
        assert type_code_for_value(value) == expected

    def test_bool_is_not_treated_as_int(self):
        """`bool` subclasses `int`, so the order of the isinstance checks matters."""
        assert type_code_for_value(True) == "b"
        assert type_code_for_value(1) == "l"


class TestFieldNames:
    @pytest.mark.parametrize("type_code", ALL_TYPE_CODES)
    def test_round_trip(self, type_code):
        field = meta_field_name("page", type_code)
        assert parse_meta_field_name(field) == (type_code, "page")

    def test_key_may_contain_underscores(self):
        """Type codes contain no underscore, so a single split recovers the key intact."""
        field = meta_field_name("my_long_key", "ss")
        assert field == "meta_ss_my_long_key"
        assert parse_meta_field_name(field) == ("ss", "my_long_key")

    @pytest.mark.parametrize(
        "field",
        [
            "content",
            "embedding",
            "id",
            "_version_",
            "meta_",
            "meta_s_",
            # `zz` is not a known type code, so this is not one of our fields.
            "meta_zz_page",
        ],
    )
    def test_non_metadata_fields_are_rejected(self, field):
        assert parse_meta_field_name(field) is None


class TestValidateMetaKeys:
    def test_accepts_alphanumeric_and_underscore(self):
        validate_meta_keys({"page": 1, "chapter_2": "a", "x9": True})

    @pytest.mark.parametrize("key", ["with space", "with.dot", "with-dash", "with:colon", ""])
    def test_rejects_keys_solr_cannot_express(self, key):
        with pytest.raises(ValueError, match="Metadata keys must contain only"):
            validate_meta_keys({key: 1})

    def test_error_names_every_offending_key(self):
        with pytest.raises(ValueError, match=r"\['a\.b', 'c d'\]"):
            validate_meta_keys({"a.b": 1, "c d": 2, "fine": 3})


class TestDocumentConversion:
    def test_round_trip_preserves_metadata_types(self):
        document = Document(
            id="1",
            content="hello",
            meta={
                "page": "100",
                "number": 2,
                "rating": 0.5,
                "flag": True,
                "tags": ["a", "b"],
                "nested": {"x": 1},
            },
            embedding=[0.1, 0.2],
        )
        solr_document = document_to_solr(document)
        restored = solr_to_document(solr_document)

        assert restored.id == "1"
        assert restored.content == "hello"
        assert restored.meta == document.meta
        assert [type(restored.meta[key]) for key in ("page", "number", "rating", "flag")] == [
            str,
            int,
            float,
            bool,
        ]
        assert restored.embedding == [0.1, 0.2]

    def test_values_sharing_a_string_form_land_in_different_fields(self):
        """The int 1, the str "1", the float 1.0 and True must not collide."""
        fields = {
            type_code_for_value(value): meta_field_name("priority", type_code_for_value(value))
            for value in (1, "1", 1.0, True)
        }
        assert len(set(fields.values())) == 4

    def test_none_metadata_values_are_not_written(self):
        """Solr has no null, and a missing field already means "no value"."""
        solr_document = document_to_solr(Document(id="1", content="x", meta={"a": None, "b": 1}))
        assert "meta_l_b" in solr_document
        assert not any(field.startswith("meta_") and field.endswith("_a") for field in solr_document)

    def test_empty_content_survives(self):
        solr_document = document_to_solr(Document(id="1", content=""))
        assert solr_document["content"] == ""
        assert solr_to_document(solr_document).content == ""

    def test_blob_round_trip(self):
        blob = ByteStream(data=b"payload", mime_type="text/plain")
        solr_document = document_to_solr(Document(id="1", blob=blob))
        restored = solr_to_document(solr_document)
        assert restored.blob is not None
        assert restored.blob.data == b"payload"
        assert restored.blob.mime_type == "text/plain"

    def test_reserved_fields_are_dropped(self):
        """Solr adds `_version_` and `_root_` to responses; neither is document content."""
        restored = solr_to_document({"id": "1", "content": "x", "_version_": 123, "_root_": "1", "_text_": ["x"]})
        assert restored.meta == {}
        assert restored.content == "x"

    def test_score_is_attached_when_requested(self):
        assert solr_to_document({"id": "1"}, score=1.5).score == 1.5
        assert solr_to_document({"id": "1"}).score is None

    def test_json_fallback_is_encoded_as_a_string(self):
        solr_document = document_to_solr(Document(id="1", meta={"nested": {"x": [1, 2]}}))
        assert solr_document["meta_j_nested"] == json.dumps({"x": [1, 2]})
        assert solr_to_document(solr_document).meta == {"nested": {"x": [1, 2]}}


class TestSchemaPayload:
    def test_creates_everything_on_an_empty_core(self):
        payload = schema_payload(
            embedding_dim=4,
            similarity_function="cosine",
            existing_field_types=set(),
            existing_fields=set(),
            existing_dynamic_fields=set(),
        )
        assert payload["add-field-type"]["name"] == vector_field_type_name(4)
        assert payload["add-field-type"]["class"] == "solr.DenseVectorField"
        assert payload["add-field-type"]["vectorDimension"] == 4
        assert {field["name"] for field in payload["add-field"]} == {"content", "embedding", "blob"}
        # One dynamic field per type code.
        assert len(payload["add-dynamic-field"]) == len(ALL_TYPE_CODES)
        assert all(
            field["name"].startswith("meta_") and field["name"].endswith("_*") for field in payload["add-dynamic-field"]
        )

    def test_is_idempotent(self):
        """Re-running the bootstrap against a fully provisioned core asks for nothing."""
        dynamic_fields = {f"meta_{code}_*" for code in ALL_TYPE_CODES}
        payload = schema_payload(
            embedding_dim=4,
            similarity_function="cosine",
            existing_field_types={vector_field_type_name(4)},
            existing_fields={"content", "embedding", "blob"},
            existing_dynamic_fields=dynamic_fields,
        )
        assert payload == {}

    def test_only_missing_pieces_are_requested(self):
        payload = schema_payload(
            embedding_dim=4,
            similarity_function="cosine",
            existing_field_types={vector_field_type_name(4)},
            existing_fields={"content"},
            existing_dynamic_fields={f"meta_{code}_*" for code in ALL_TYPE_CODES},
        )
        assert "add-field-type" not in payload
        assert "add-dynamic-field" not in payload
        assert {field["name"] for field in payload["add-field"]} == {"embedding", "blob"}

    def test_hnsw_attributes_are_only_set_when_given(self):
        """Solr 10 renamed these, so the default payload must stay silent about them."""
        default = schema_payload(
            embedding_dim=4,
            similarity_function="cosine",
            existing_field_types=set(),
            existing_fields=set(),
            existing_dynamic_fields=set(),
        )
        assert "hnswM" not in default["add-field-type"]
        assert "hnswMaxConnections" not in default["add-field-type"]

        tuned = schema_payload(
            embedding_dim=4,
            similarity_function="cosine",
            existing_field_types=set(),
            existing_fields=set(),
            existing_dynamic_fields=set(),
            vector_field_type_params={"hnswM": 32},
        )
        assert tuned["add-field-type"]["hnswM"] == 32

    def test_json_dynamic_field_is_not_indexed(self):
        payload = schema_payload(
            embedding_dim=4,
            similarity_function="cosine",
            existing_field_types=set(),
            existing_fields=set(),
            existing_dynamic_fields=set(),
        )
        json_field = next(
            field for field in payload["add-dynamic-field"] if field["name"] == f"meta_{JSON_TYPE_CODE}_*"
        )
        assert json_field["indexed"] is False
        assert json_field["stored"] is True
