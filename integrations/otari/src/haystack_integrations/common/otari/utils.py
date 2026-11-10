# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import logging
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)

_DEFAULT_API_BASE_URL = "https://api.otari.ai/api/v1"
_EU_API_BASE_URL = "https://eu.api.otari.ai/api/v1"
_EU_API_KEY_PREFIX = "otk_v1_eu_"


def _warn_if_eu_key_on_default_url(api_key: Secret, api_base_url: str | None) -> None:
    """
    Log a warning if an API key of otari.ai's EU region is sent to the default otari.ai host.

    The default host rejects such keys with a bare 401 that doesn't name the right host.
    """
    resolved_api_key = api_key.resolve_value()
    if api_base_url == _DEFAULT_API_BASE_URL and resolved_api_key and resolved_api_key.startswith(_EU_API_KEY_PREFIX):
        logger.warning(
            "The Otari API key belongs to otari.ai's EU region, which {default_url} rejects. "
            "Set api_base_url to {eu_url}.",
            default_url=_DEFAULT_API_BASE_URL,
            eu_url=_EU_API_BASE_URL,
        )
