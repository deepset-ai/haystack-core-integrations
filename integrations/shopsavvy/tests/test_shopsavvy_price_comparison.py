# SPDX-FileCopyrightText: 2026-present Monolith Technologies, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.converters.shopsavvy import ShopSavvyPriceComparison


class TestShopSavvyPriceComparison:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("SHOPSAVVY_API_KEY", "ss_test_key123")
        compare = ShopSavvyPriceComparison()
        assert compare.retailer is None
        assert compare.api_key.resolve_value() == "ss_test_key123"

    def test_init_with_params(self):
        compare = ShopSavvyPriceComparison(
            api_key=Secret.from_token("ss_test_custom_key"),
            retailer="amazon.com",
        )
        assert compare.retailer == "amazon.com"
        assert compare.api_key.resolve_value() == "ss_test_custom_key"

    def test_run_returns_offer_documents(self):
        compare = ShopSavvyPriceComparison(
            api_key=Secret.from_token("ss_test_key123"),
        )

        mock_offer_1 = MagicMock()
        mock_offer_1.retailer = "Amazon"
        mock_offer_1.price = 299.99
        mock_offer_1.currency = "USD"
        mock_offer_1.availability = "In Stock"
        mock_offer_1.condition = "New"
        mock_offer_1.URL = "https://amazon.com/dp/B09XS7JWHH"
        mock_offer_1.seller = None
        mock_offer_1.timestamp = "2026-04-01T12:00:00Z"

        mock_offer_2 = MagicMock()
        mock_offer_2.retailer = "Best Buy"
        mock_offer_2.price = 319.99
        mock_offer_2.currency = "USD"
        mock_offer_2.availability = "In Stock"
        mock_offer_2.condition = "New"
        mock_offer_2.URL = "https://bestbuy.com/product/123"
        mock_offer_2.seller = None
        mock_offer_2.timestamp = "2026-04-01T11:30:00Z"

        mock_product = MagicMock()
        mock_product.title = "Sony WH-1000XM5"
        mock_product.offers = [mock_offer_1, mock_offer_2]

        mock_result = MagicMock()
        mock_result.data = [mock_product]

        mock_client = MagicMock()
        mock_client.get_current_offers.return_value = mock_result
        compare._client = mock_client

        result = compare.run(identifier="B09XS7JWHH")

        assert "documents" in result
        assert len(result["documents"]) == 2

        doc_0 = result["documents"][0]
        assert isinstance(doc_0, Document)
        data_0 = json.loads(doc_0.content)
        assert data_0["retailer"] == "Amazon"
        assert data_0["price"] == 299.99
        assert doc_0.meta["source"] == "shopsavvy"

        doc_1 = result["documents"][1]
        data_1 = json.loads(doc_1.content)
        assert data_1["retailer"] == "Best Buy"
        assert data_1["price"] == 319.99

        mock_client.get_current_offers.assert_called_once_with(
            identifier="B09XS7JWHH",
            retailer=None,
        )

    def test_run_with_retailer_filter(self):
        compare = ShopSavvyPriceComparison(
            api_key=Secret.from_token("ss_test_key123"),
            retailer="amazon.com",
        )

        mock_product = MagicMock()
        mock_product.title = "Test"
        mock_product.offers = []
        mock_result = MagicMock()
        mock_result.data = [mock_product]

        mock_client = MagicMock()
        mock_client.get_current_offers.return_value = mock_result
        compare._client = mock_client

        compare.run(identifier="B09XS7JWHH")
        mock_client.get_current_offers.assert_called_once_with(
            identifier="B09XS7JWHH",
            retailer="amazon.com",
        )

    def test_run_runtime_retailer_overrides_init(self):
        compare = ShopSavvyPriceComparison(
            api_key=Secret.from_token("ss_test_key123"),
            retailer="amazon.com",
        )

        mock_product = MagicMock()
        mock_product.title = "Test"
        mock_product.offers = []
        mock_result = MagicMock()
        mock_result.data = [mock_product]

        mock_client = MagicMock()
        mock_client.get_current_offers.return_value = mock_result
        compare._client = mock_client

        compare.run(identifier="B09XS7JWHH", retailer="walmart.com")
        mock_client.get_current_offers.assert_called_once_with(
            identifier="B09XS7JWHH",
            retailer="walmart.com",
        )

    def test_run_returns_empty_on_error(self):
        compare = ShopSavvyPriceComparison(
            api_key=Secret.from_token("ss_test_key123"),
        )

        mock_client = MagicMock()
        mock_client.get_current_offers.side_effect = Exception("API error")
        compare._client = mock_client

        result = compare.run(identifier="test")
        assert result["documents"] == []

    def test_warm_up_initializes_client(self):
        compare = ShopSavvyPriceComparison(
            api_key=Secret.from_token("ss_test_key123"),
        )
        assert compare._client is None

        with patch(
            "haystack_integrations.components.converters.shopsavvy.price_comparison.ShopSavvyDataAPI"
        ) as mock_cls:
            compare.warm_up()
            assert compare._client is mock_cls.return_value

    @pytest.mark.skipif(
        not os.environ.get("SHOPSAVVY_API_KEY"),
        reason="Export SHOPSAVVY_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        compare = ShopSavvyPriceComparison(
            api_key=Secret.from_env_var("SHOPSAVVY_API_KEY"),
        )
        result = compare.run(identifier="B09XS7JWHH")

        assert "documents" in result
        assert isinstance(result["documents"], list)
