# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import component

from haystack_integrations.components.mrscraper._base import MrScraperComponent
from haystack_integrations.utils.mrscraper.validation import validate_choice, validate_integer, validate_nonblank


@component
class MrScraperGetResults(MrScraperComponent):
    """Get one explicitly paginated and sorted page of results for an existing scraper."""

    @staticmethod
    def _params(
        scraper_id: str,
        page: int,
        page_size: int,
        sort_by: Literal["createdAt"],
        sort_order: Literal["ASC", "DESC"],
    ) -> dict[str, Any]:
        return {
            "filters[scraperId]": validate_nonblank(scraper_id, "scraper_id"),
            "page": validate_integer(page, "page"),
            "pageSize": validate_integer(page_size, "page_size"),
            "sort": validate_choice(sort_by, "sort_by", ("createdAt",)),
            "sortOrder": validate_choice(sort_order, "sort_order", ("ASC", "DESC")),
        }

    @component.output_types(result=Any)
    def run(
        self,
        scraper_id: str,
        page: int = 1,
        page_size: int = 10,
        sort_by: Literal["createdAt"] = "createdAt",
        sort_order: Literal["ASC", "DESC"] = "DESC",
    ) -> dict[str, Any]:
        """
        Get paginated scraper results.

        :param scraper_id: Nonblank ID of the scraper whose results should be fetched.
        :param page: Page number.
        :param page_size: Results per page.
        :param sort_by: Sort field; currently only `createdAt`.
        :param sort_order: Sort direction, `ASC` or `DESC`.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        params = self._params(scraper_id, page, page_size, sort_by, sort_order)
        return {"result": self._client().primary("GET", "/api/v1/results", params=params)}

    @component.output_types(result=Any)
    async def run_async(
        self,
        scraper_id: str,
        page: int = 1,
        page_size: int = 10,
        sort_by: Literal["createdAt"] = "createdAt",
        sort_order: Literal["ASC", "DESC"] = "DESC",
    ) -> dict[str, Any]:
        """
        Asynchronously get paginated scraper results.

        :param scraper_id: Nonblank ID of the scraper whose results should be fetched.
        :param page: Page number.
        :param page_size: Results per page.
        :param sort_by: Sort field; currently only `createdAt`.
        :param sort_order: Sort direction, `ASC` or `DESC`.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        params = self._params(scraper_id, page, page_size, sort_by, sort_order)
        return {"result": await self._client().primary_async("GET", "/api/v1/results", params=params)}


@component
class MrScraperGetLatestResults(MrScraperComponent):
    """Get only the newest N results for an existing scraper."""

    @staticmethod
    def _params(scraper_id: str, count: int) -> dict[str, Any]:
        return {
            "filters[scraperId]": validate_nonblank(scraper_id, "scraper_id"),
            "page": 1,
            "pageSize": validate_integer(count, "count"),
            "sort": "createdAt",
            "sortOrder": "DESC",
        }

    @component.output_types(result=Any)
    def run(self, scraper_id: str, count: int = 10) -> dict[str, Any]:
        """
        Get the latest scraper results.

        :param scraper_id: Nonblank ID of the scraper whose latest results should be fetched.
        :param count: Number of newest results to return.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        return {"result": self._client().primary("GET", "/api/v1/results", params=self._params(scraper_id, count))}

    @component.output_types(result=Any)
    async def run_async(self, scraper_id: str, count: int = 10) -> dict[str, Any]:
        """
        Asynchronously get the latest scraper results.

        :param scraper_id: Nonblank ID of the scraper whose latest results should be fetched.
        :param count: Number of newest results to return.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        result = await self._client().primary_async("GET", "/api/v1/results", params=self._params(scraper_id, count))
        return {"result": result}


@component
class MrScraperGetResultDetail(MrScraperComponent):
    """Get one complete scraper result by its result ID."""

    @component.output_types(result=Any)
    def run(self, result_id: str) -> dict[str, Any]:
        """
        Get one result in detail.

        :param result_id: Nonblank result ID, URL-encoded as one path segment.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        value = validate_nonblank(result_id, "result_id")
        client = self._client()
        path = f"/api/v1/results/{client.encoded_path_segment(value)}"
        return {"result": client.primary("GET", path)}

    @component.output_types(result=Any)
    async def run_async(self, result_id: str) -> dict[str, Any]:
        """
        Asynchronously get one result in detail.

        :param result_id: Nonblank result ID, URL-encoded as one path segment.
        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        value = validate_nonblank(result_id, "result_id")
        client = self._client()
        path = f"/api/v1/results/{client.encoded_path_segment(value)}"
        return {"result": await client.primary_async("GET", path)}
