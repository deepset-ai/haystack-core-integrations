# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import struct

import pytest
from glide_shared.commands.server_modules.ft_options.ft_create_options import DistanceMetricType
from glide_shared.commands.server_modules.ft_options.ft_search_options import FtSearchOptions
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.valkey import ValkeyDocumentStore


class TestValkeyDocumentStoreStaticMethods:
    """Test static methods that were refactored from instance methods."""

    def test_parse_metric_valid_metrics(self):
        """Test _parse_metric static method with valid metrics."""
        assert ValkeyDocumentStore._parse_metric("l2") == DistanceMetricType.L2
        assert ValkeyDocumentStore._parse_metric("cosine") == DistanceMetricType.COSINE
        assert ValkeyDocumentStore._parse_metric("ip") == DistanceMetricType.IP

    def test_parse_metric_invalid_metric(self):
        """Test _parse_metric static method with invalid metric."""
        with pytest.raises(ValueError, match="Unsupported metric: invalid"):
            ValkeyDocumentStore._parse_metric("invalid")

    def test_to_float32_bytes(self):
        """Test _to_float32_bytes static method."""
        vec = [1.0, 2.5, -3.7]
        result = ValkeyDocumentStore._to_float32_bytes(vec)

        # Verify it's bytes
        assert isinstance(result, bytes)

        # Verify correct length (4 bytes per float)
        assert len(result) == len(vec) * 4

        # Verify correct values by unpacking
        unpacked = [struct.unpack("<f", result[i : i + 4])[0] for i in range(0, len(result), 4)]
        assert unpacked == pytest.approx(vec, rel=1e-6)

    def test_verify_node_list_valid(self):
        """Test _verify_node_list static method with valid node list."""
        # Should not raise any exception
        ValkeyDocumentStore._verify_node_list([("localhost", 6379)])
        ValkeyDocumentStore._verify_node_list([("host1", 6379), ("host2", 6380)])

    def test_verify_node_list_empty(self):
        """Test _verify_node_list static method with empty node list."""
        with pytest.raises(Exception, match="Node list is empty"):
            ValkeyDocumentStore._verify_node_list([])

        with pytest.raises(Exception, match="Node list is empty"):
            ValkeyDocumentStore._verify_node_list(None)

    def test_build_credentials_with_username_and_password(self):
        """Test _build_credentials static method with both username and password."""
        creds = ValkeyDocumentStore._build_credentials("user", "pass")
        assert creds is not None
        assert creds.username == "user"
        assert creds.password == "pass"

    def test_build_credentials_with_username_only(self):
        """Test _build_credentials static method with username only."""
        # ServerCredentials requires password, so username-only should return None
        creds = ValkeyDocumentStore._build_credentials("user", None)
        assert creds is None

    def test_build_credentials_with_password_only(self):
        """Test _build_credentials static method with password only."""
        creds = ValkeyDocumentStore._build_credentials(None, "pass")
        assert creds is not None
        # Username should default to None (ServerCredentials will use "default")
        assert creds.password == "pass"

    def test_build_credentials_with_neither(self):
        """Test _build_credentials static method with neither username nor password."""
        creds = ValkeyDocumentStore._build_credentials(None, None)
        assert creds is None

    def test_validate_documents_valid(self):
        """Test _validate_documents static method with valid documents."""
        docs = [
            Document(id="1", content="test"),
            Document(id="2", content="test2"),
        ]
        # Should not raise any exception
        ValkeyDocumentStore._validate_documents(docs)

    def test_validate_documents_invalid_type(self):
        """Test _validate_documents static method with invalid document type."""
        with pytest.raises(ValueError, match="expects a list of Documents"):
            ValkeyDocumentStore._validate_documents([Document(id="1", content="test"), "not_a_document"])

    def test_validate_policy_valid(self):
        """Test _validate_policy static method with valid policies."""
        # Should not raise any exception, but may log warnings
        ValkeyDocumentStore._validate_policy(DuplicatePolicy.NONE)
        ValkeyDocumentStore._validate_policy(DuplicatePolicy.OVERWRITE)

    def test_build_search_query_and_options_basic(self):
        """Test _build_search_query_and_options static method with basic parameters."""
        embedding = [0.1, 0.2, 0.3]
        filters = None
        limit = 10
        with_embedding = True

        query, options = ValkeyDocumentStore._build_search_query_and_options(
            embedding, filters, limit, with_embedding=with_embedding
        )

        assert isinstance(query, str)
        assert isinstance(options, FtSearchOptions)
        assert "KNN 10" in query
        assert "query_vector" in query
        assert "query_vector" in options.params

    def test_build_search_query_and_options_with_filters(self):
        """Test _build_search_query_and_options static method with filters."""
        embedding = [0.1, 0.2, 0.3]
        filters = {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": "news"}]}
        limit = 5
        with_embedding = False

        query, options = ValkeyDocumentStore._build_search_query_and_options(
            embedding, filters, limit, with_embedding=with_embedding
        )

        assert "meta_category:{news}" in query
        assert "KNN 5" in query
        # Should not include vector field when with_embedding=False
        vector_fields = [
            field for field in options.return_fields if hasattr(field, "alias") and field.alias == "vector"
        ]
        assert len(vector_fields) == 0

    def test_build_search_query_and_options_with_embedding_return(self):
        """Test _build_search_query_and_options static method with embedding return."""
        embedding = [0.1, 0.2, 0.3]
        filters = None
        limit = 10

        _, options = ValkeyDocumentStore._build_search_query_and_options(embedding, filters, limit, with_embedding=True)

        # Should have vector field when with_embedding=True
        vector_fields = [
            field for field in options.return_fields if hasattr(field, "alias") and field.alias == "vector"
        ]
        assert len(vector_fields) == 1, "Should have exactly one vector field when with_embedding=True"

        # Should have more return fields when with_embedding=True vs False
        _, options_no_embed = ValkeyDocumentStore._build_search_query_and_options(
            embedding, filters, limit, with_embedding=False
        )

        vector_fields = [
            field for field in options_no_embed.return_fields if hasattr(field, "alias") and field.alias == "vector"
        ]
        assert len(vector_fields) == 0, "Should have no vector field with_embedding=False"

    def test_parse_documents_from_ft_empty_results(self):
        """Test _parse_documents_from_ft static method with empty results."""
        raw_results = [0, {}]  # Empty results format
        with_embedding = True

        docs = ValkeyDocumentStore._parse_documents_from_ft(raw_results, with_embedding=with_embedding)
        assert docs == []

    def test_parse_documents_from_ft_no_results(self):
        """Test _parse_documents_from_ft static method with no results."""
        raw_results = None
        with_embedding = True

        docs = ValkeyDocumentStore._parse_documents_from_ft(raw_results, with_embedding=with_embedding)
        assert docs == []

    def test_class_constants_accessible(self):
        """Test that class constants are accessible and have correct values."""
        assert ValkeyDocumentStore._DUMMY_VALUE == -10.0
        assert "l2" in ValkeyDocumentStore._METRIC_MAP
        assert "cosine" in ValkeyDocumentStore._METRIC_MAP
        assert "ip" in ValkeyDocumentStore._METRIC_MAP
        assert ValkeyDocumentStore._METRIC_MAP["cosine"] == DistanceMetricType.COSINE

    def test_dummy_vector_consistency(self):
        """Test that dummy vector uses the class constant consistently."""
        store = ValkeyDocumentStore(embedding_dim=5)
        expected_dummy = [ValkeyDocumentStore._DUMMY_VALUE] * 5
        assert store._dummy_vector == expected_dummy

    def test_static_methods_dont_need_instance(self):
        """Test that static methods can be called without creating an instance."""
        # These should all work without instantiating ValkeyDocumentStore
        ValkeyDocumentStore._parse_metric("cosine")
        ValkeyDocumentStore._to_float32_bytes([1.0, 2.0])
        ValkeyDocumentStore._verify_node_list([("localhost", 6379)])
        ValkeyDocumentStore._build_credentials("user", "pass")
        ValkeyDocumentStore._validate_documents([Document(id="1", content="test")])
        ValkeyDocumentStore._validate_policy(DuplicatePolicy.NONE)
