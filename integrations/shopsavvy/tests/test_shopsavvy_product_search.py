# SPDX-FileCopyrightText: 2026-present Monolith Technologies, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.converters.shopsavvy import ShopSavvyProductSearch


class TestShopSavvyProductSearch:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("SHOPSAVVY_API_KEY", "ss_test_key123")
        search = ShopSavvyProductSearch()
        assert search.top_k == 10
        assert search.api_key.resolve_value() == "ss_test_key123"

    def test_init_with_params(self):
        search = ShopSavvyProductSearch(
            api_key=Secret.from_token("ss_test_custom_key"),
            top_k=5,
        )
        assert search.top_k == 5
        assert search.api_key.resolve_value() == "ss_test_custom_key"

    def test_run_returns_documents(self):
        search = ShopSavvyProductSearch(
            api_key=Secret.from_token("ss_test_key123"),
            top_k=5,
        )

        mock_product = MagicMock()
        mock_product.title = "Sony WH-1000XM5"
        mock_product.shopsavvy = "sp_123"
        mock_product.brand = "Sony"
        mock_product.category = "Headphones"
        mock_product.barcode = "027242923782"
        mock_product.amazon = "B09XS7JWHH"
        mock_product.model = "WH1000XM5"
        mock_product.description = "Noise-canceling headphones"

        mock_result = MagicMock()
        mock_result.data = [mock_product]

        mock_client = MagicMock()
        mock_client.search_products.return_value = mock_result
        search._client = mock_client

        result = search.run(query="sony headphones")

        assert "documents" in result
        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert isinstance(doc, Document)

        data = json.loads(doc.content)
        assert data["title"] == "Sony WH-1000XM5"
        assert data["brand"] == "Sony"
        assert doc.meta["source"] == "shopsavvy"
        assert doc.meta["shopsavvy_id"] == "sp_123"

        mock_client.search_products.assert_called_once_with(query="sony headphones", limit=5)

    def test_run_returns_empty_on_error(self):
        search = ShopSavvyProductSearch(
            api_key=Secret.from_token("ss_test_key123"),
        )

        mock_client = MagicMock()
        mock_client.search_products.side_effect = Exception("API error")
        search._client = mock_client

        result = search.run(query="test")
        assert result["documents"] == []

    def test_warm_up_initializes_client(self):
        search = ShopSavvyProductSearch(
            api_key=Secret.from_token("ss_test_key123"),
        )
        assert search._client is None

        with patch(
            "haystack_integrations.components.converters.shopsavvy.product_search.ShopSavvyDataAPI"
        ) as mock_cls:
            search.warm_up()
            assert search._client is mock_cls.return_value

    @pytest.mark.skipif(
        not os.environ.get("SHOPSAVVY_API_KEY"),
        reason="Export SHOPSAVVY_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        search = ShopSavvyProductSearch(
            api_key=Secret.from_env_var("SHOPSAVVY_API_KEY"),
            top_k=3,
        )
        result = search.run(query="iphone 15 pro")

        assert "documents" in result
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) > 0
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content
