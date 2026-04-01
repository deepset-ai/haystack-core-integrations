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
class ShopSavvyProductSearch:
    """
    A component that searches for products using the ShopSavvy Data API.

    Given a search query, returns matching products as Haystack Documents
    with product details in the content and structured metadata.

    You need a ShopSavvy API key from [shopsavvy.com/data](https://shopsavvy.com/data).

    ### Usage example

    ```python
    from haystack_integrations.components.converters.shopsavvy import ShopSavvyProductSearch
    from haystack.utils import Secret

    search = ShopSavvyProductSearch(
        api_key=Secret.from_env_var("SHOPSAVVY_API_KEY"),
        top_k=5,
    )
    result = search.run(query="sony headphones")
    for doc in result["documents"]:
        print(doc.meta["title"], doc.meta["brand"])
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("SHOPSAVVY_API_KEY"),
        top_k: int = 10,
    ) -> None:
        """
        Initialize the ShopSavvyProductSearch component.

        :param api_key:
            API key for ShopSavvy Data API.
            Defaults to the `SHOPSAVVY_API_KEY` environment variable.
        :param top_k:
            Maximum number of products to return. Defaults to 10.
        """
        self.api_key = api_key
        self.top_k = top_k
        self._client: Optional[ShopSavvyDataAPI] = None

    def warm_up(self) -> None:
        """Initialize the ShopSavvy API client."""
        if self._client is None:
            resolved_key = self.api_key.resolve_value()
            self._client = ShopSavvyDataAPI(ShopSavvyConfig(api_key=resolved_key))

    @component.output_types(documents=list[Document])
    def run(self, query: str) -> dict[str, Any]:
        """
        Search for products by keyword and return results as Documents.

        :param query:
            Search query string (product name, keyword, or description).
        :returns:
            A dictionary with key `documents` containing a list of Documents.
            Each document's content is a JSON string with product details,
            and metadata includes title, brand, category, and identifiers.
        """
        if self._client is None:
            self.warm_up()

        try:
            result = self._client.search_products(query=query, limit=self.top_k)  # type: ignore[union-attr]
        except Exception as error:
            logger.exception("Failed to search ShopSavvy for query '%s': %s", query, error)
            return {"documents": []}

        documents: list[Document] = []
        for product in result.data:
            doc_data = {
                "title": product.title,
                "shopsavvy_id": product.shopsavvy,
                "brand": product.brand,
                "category": product.category,
                "barcode": product.barcode,
                "asin": product.amazon,
                "model": product.model,
                "description": product.description,
            }
            doc = Document(
                content=json.dumps(doc_data, default=str),
                meta={
                    "title": product.title,
                    "brand": product.brand,
                    "category": product.category,
                    "shopsavvy_id": product.shopsavvy,
                    "barcode": product.barcode,
                    "asin": product.amazon,
                    "source": "shopsavvy",
                },
            )
            documents.append(doc)

        return {"documents": documents}
