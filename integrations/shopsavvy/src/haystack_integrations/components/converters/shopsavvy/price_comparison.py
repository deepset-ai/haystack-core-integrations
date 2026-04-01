# SPDX-FileCopyrightText: 2026-present Monolith Technologies, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Optional

from haystack import Document, component, logging
from haystack.utils import Secret
from shopsavvy import ShopSavvyDataAPI, ShopSavvyConfig

logger = logging.getLogger(__name__)


@component
class ShopSavvyPriceComparison:
    """
    A component that retrieves current prices for a product across retailers
    using the ShopSavvy Data API.

    Given a product identifier (barcode, ASIN, URL, model number, or name),
    returns one Document per retailer offer with pricing details.

    You need a ShopSavvy API key from [shopsavvy.com/data](https://shopsavvy.com/data).

    ### Usage example

    ```python
    from haystack_integrations.components.converters.shopsavvy import ShopSavvyPriceComparison
    from haystack.utils import Secret

    compare = ShopSavvyPriceComparison(
        api_key=Secret.from_env_var("SHOPSAVVY_API_KEY"),
    )
    result = compare.run(identifier="B09XS7JWHH")
    for doc in result["documents"]:
        print(doc.meta["retailer"], doc.meta["price"])
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("SHOPSAVVY_API_KEY"),
        retailer: Optional[str] = None,
    ) -> None:
        """
        Initialize the ShopSavvyPriceComparison component.

        :param api_key:
            API key for ShopSavvy Data API.
            Defaults to the `SHOPSAVVY_API_KEY` environment variable.
        :param retailer:
            Optional retailer domain to filter offers (e.g. "amazon.com").
        """
        self.api_key = api_key
        self.retailer = retailer
        self._client: Optional[ShopSavvyDataAPI] = None

    def warm_up(self) -> None:
        """Initialize the ShopSavvy API client."""
        if self._client is None:
            resolved_key = self.api_key.resolve_value()
            self._client = ShopSavvyDataAPI(ShopSavvyConfig(api_key=resolved_key))

    @component.output_types(documents=list[Document])
    def run(
        self,
        identifier: str,
        retailer: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get current prices for a product across retailers.

        :param identifier:
            Product identifier: barcode/UPC, Amazon ASIN, product URL,
            model number, or product name.
        :param retailer:
            Optional retailer domain to filter. Overrides the init-time value.
        :returns:
            A dictionary with key `documents` containing a list of Documents.
            Each document represents one retailer offer with price, availability,
            and a link to the product page.
        """
        if self._client is None:
            self.warm_up()

        effective_retailer = retailer if retailer is not None else self.retailer

        try:
            result = self._client.get_current_offers(  # type: ignore[union-attr]
                identifier=identifier,
                retailer=effective_retailer,
            )
        except Exception as error:
            logger.exception("Failed to get ShopSavvy offers for '%s': %s", identifier, error)
            return {"documents": []}

        documents: list[Document] = []
        for product in result.data:
            for offer in product.offers:
                doc_data = {
                    "product_title": product.title,
                    "retailer": offer.retailer,
                    "price": offer.price,
                    "currency": offer.currency,
                    "availability": offer.availability,
                    "condition": offer.condition,
                    "url": offer.URL,
                    "seller": offer.seller,
                    "last_updated": offer.timestamp,
                }
                doc = Document(
                    content=json.dumps(doc_data, default=str),
                    meta={
                        "product_title": product.title,
                        "retailer": offer.retailer,
                        "price": offer.price,
                        "currency": offer.currency,
                        "availability": offer.availability,
                        "url": offer.URL,
                        "source": "shopsavvy",
                    },
                )
                documents.append(doc)

        return {"documents": documents}
