# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import re

# Cohere clients build a request URL by appending the endpoint path to the configured base URL, so a base URL that
# already ends with one of these paths produces a doubled path and a generic API error at request time.
_COHERE_ENDPOINT_PATH = re.compile(r"/v\d+/(?:rerank|embed|chat)$")


def validate_api_base_url(api_base_url: str) -> None:
    """
    Check that `api_base_url` is a base URL rather than a full endpoint URL.

    :param api_base_url: The base URL to validate.
    :raises ValueError: If `api_base_url` ends with a Cohere endpoint path, which the Cohere client appends itself.
    """
    trimmed = api_base_url.rstrip("/")
    match = _COHERE_ENDPOINT_PATH.search(trimmed)
    if match is None:
        return

    base = trimmed[: match.start()]
    hint = f" and pass '{base}' instead" if base else ""
    msg = (
        f"'api_base_url' must be a base URL, not a full endpoint URL, but got {api_base_url!r}. "
        f"The Cohere client appends the endpoint path itself, so remove the trailing '{match.group()}'{hint}."
    )
    raise ValueError(msg)
