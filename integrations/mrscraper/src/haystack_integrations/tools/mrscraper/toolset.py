# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Literal

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import ComponentTool, Tool, Toolset
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.mrscraper import (
    MrScraperCrawlWebsiteUrls,
    MrScraperCreateListingScraper,
    MrScraperCreatePromptScraper,
    MrScraperCreateWebsiteCrawlScraper,
    MrScraperExtractListings,
    MrScraperExtractPageByPrompt,
    MrScraperExtractStructuredData,
    MrScraperFetchRenderedHtml,
    MrScraperGetAccountInfo,
    MrScraperGetLatestResults,
    MrScraperGetResultDetail,
    MrScraperGetResults,
    MrScraperRunExistingScraper,
    MrScraperRunExistingScraperBatch,
    MrScraperSearchGoogleSerp,
)
from haystack_integrations.components.mrscraper.scraper_runs import conditional_run_tool_schema
from haystack_integrations.utils.mrscraper.validation import validate_choice, validate_number

MrScraperToolGroup = Literal["account", "discovery", "extraction", "results", "scraper_creation", "scraper_runs"]

_GROUPS: tuple[MrScraperToolGroup, ...] = (
    "account",
    "discovery",
    "extraction",
    "results",
    "scraper_creation",
    "scraper_runs",
)


def _result_to_string(result: Any) -> str:
    """Convert a native component result for an Agent without altering text responses."""
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class MrScraperToolset(Toolset):
    """A serializable set of 15 independently named MrScraper ComponentTools."""

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MRSCRAPER_API_TOKEN"),
        groups: list[MrScraperToolGroup] | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 300.0,
    ) -> None:
        """
        Create the MrScraper tools.

        :param api_key: MrScraper API token. Defaults to the `MRSCRAPER_API_TOKEN` environment variable.
        :param groups: Optional subset of `account`, `discovery`, `extraction`, `results`, `scraper_creation`,
            and `scraper_runs`. All groups are included by default.
        :param connect_timeout: Maximum seconds to establish an HTTP connection.
        :param read_timeout: Maximum seconds to wait while reading an HTTP response.
        """
        self.api_key = api_key
        self.connect_timeout = validate_number(connect_timeout, "connect_timeout", minimum=0)
        self.read_timeout = validate_number(read_timeout, "read_timeout", minimum=0)
        if groups is None:
            selected_groups = _GROUPS
            self.groups: list[MrScraperToolGroup] | None = None
        else:
            if len(groups) != len(set(groups)):
                msg = "'groups' must not contain duplicate values."
                raise ValueError(msg)
            selected_groups = tuple(validate_choice(group, "group", _GROUPS) for group in groups)
            self.groups = list(selected_groups)

        component_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
        }
        definitions: list[tuple[MrScraperToolGroup, str, str, Any]] = [
            (
                "account",
                "mrscraper_get_account_info",
                "Get MrScraper account details, token usage, and token limits.",
                MrScraperGetAccountInfo,
            ),
            (
                "discovery",
                "mrscraper_crawl_website_urls",
                "Immediately discover URLs by crawling links from a starting website.",
                MrScraperCrawlWebsiteUrls,
            ),
            (
                "discovery",
                "mrscraper_search_google_serp",
                "Search Google now and return either native JSON search data or exact HTML.",
                MrScraperSearchGoogleSerp,
            ),
            (
                "extraction",
                "mrscraper_extract_page_by_prompt",
                "Immediately extract one page using instructions and an optional expected JSON shape.",
                MrScraperExtractPageByPrompt,
            ),
            (
                "extraction",
                "mrscraper_extract_listings",
                "Immediately extract repeated listing items across one or more pages.",
                MrScraperExtractListings,
            ),
            (
                "extraction",
                "mrscraper_extract_structured_data",
                "Immediately extract a page with an exact built-in category preset such as article or product.",
                MrScraperExtractStructuredData,
            ),
            (
                "extraction",
                "mrscraper_fetch_rendered_html",
                "Fetch a JavaScript-rendered page with browser, output, screenshot, and waiting controls.",
                MrScraperFetchRenderedHtml,
            ),
            (
                "results",
                "mrscraper_get_results",
                "Get an explicitly paginated and sorted page of results for a scraper.",
                MrScraperGetResults,
            ),
            (
                "results",
                "mrscraper_get_latest_results",
                "Get only the newest N results for a scraper.",
                MrScraperGetLatestResults,
            ),
            (
                "results",
                "mrscraper_get_result_detail",
                "Get the complete detail for one known result ID.",
                MrScraperGetResultDetail,
            ),
            (
                "scraper_creation",
                "mrscraper_create_prompt_scraper",
                "Create a reusable General AI scraper from extraction instructions.",
                MrScraperCreatePromptScraper,
            ),
            (
                "scraper_creation",
                "mrscraper_create_listing_scraper",
                "Create a reusable Listing AI scraper for repeated or paginated items.",
                MrScraperCreateListingScraper,
            ),
            (
                "scraper_creation",
                "mrscraper_create_website_crawl_scraper",
                "Create a reusable Map AI scraper for later URL-discovery runs.",
                MrScraperCreateWebsiteCrawlScraper,
            ),
            (
                "scraper_runs",
                "mrscraper_run_existing_scraper",
                "Run one URL with an existing AI or manual scraper and type-specific settings.",
                MrScraperRunExistingScraper,
            ),
            (
                "scraper_runs",
                "mrscraper_run_existing_scraper_batch",
                "Run multiple URLs in one batch with an existing AI or manual scraper.",
                MrScraperRunExistingScraperBatch,
            ),
        ]
        tools: list[Tool] = []
        for group, name, description, component_class in definitions:
            if group not in selected_groups:
                continue
            configured_component = component_class(**component_kwargs)
            parameters = None
            if component_class is MrScraperRunExistingScraper:
                generated_tool = ComponentTool(
                    component=configured_component,
                    name=name,
                    description=description,
                )
                parameters = conditional_run_tool_schema(generated_tool.parameters)
            tools.append(
                ComponentTool(
                    component=configured_component,
                    name=name,
                    description=description,
                    parameters=parameters,
                    outputs_to_string={"source": "result", "handler": _result_to_string},
                )
            )
        super().__init__(tools=tools)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration without resolving the API token."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "api_key": self.api_key.to_dict(),
                "groups": self.groups,
                "connect_timeout": self.connect_timeout,
                "read_timeout": self.read_timeout,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MrScraperToolset":
        """Deserialize the toolset and rebuild its independently configured components."""
        inner = dict(data["data"])
        deserialize_secrets_inplace(inner, keys=["api_key"])
        return cls(**inner)
