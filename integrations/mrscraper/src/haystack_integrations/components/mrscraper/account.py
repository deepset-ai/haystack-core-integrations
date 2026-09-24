# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component

from haystack_integrations.components.mrscraper._base import MrScraperComponent


@component
class MrScraperGetAccountInfo(MrScraperComponent):
    """Retrieve MrScraper account details, token usage, and token limits."""

    @component.output_types(result=Any)
    def run(self) -> dict[str, Any]:
        """
        Get account information.

        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        return {"result": self._client().primary("GET", "/api/v1/subscription-accounts")}

    @component.output_types(result=Any)
    async def run_async(self) -> dict[str, Any]:
        """
        Asynchronously get account information.

        :returns: A dictionary containing the unmodified decoded upstream value under `result`.
        """
        return {"result": await self._client().primary_async("GET", "/api/v1/subscription-accounts")}
