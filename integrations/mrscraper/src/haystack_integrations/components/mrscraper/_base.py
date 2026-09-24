# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, TypeVar

from haystack import default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.utils.mrscraper.client import MrScraperClient, MrScraperError
from haystack_integrations.utils.mrscraper.validation import validate_number

T = TypeVar("T", bound="MrScraperComponent")


class MrScraperComponent:
    """Shared immutable authentication and timeout configuration for MrScraper components."""

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MRSCRAPER_API_TOKEN"),
        connect_timeout: float = 10.0,
        read_timeout: float = 300.0,
    ) -> None:
        """
        Initialize shared MrScraper component configuration.

        :param api_key: MrScraper API token. Defaults to the `MRSCRAPER_API_TOKEN` environment variable.
        :param connect_timeout: Maximum seconds to establish an HTTP connection.
        :param read_timeout: Maximum seconds to wait while reading an HTTP response.
        """
        self.api_key = api_key
        self.connect_timeout = validate_number(connect_timeout, "connect_timeout", minimum=0)
        self.read_timeout = validate_number(read_timeout, "read_timeout", minimum=0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component without resolving its API token."""
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
        )

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Deserialize a component and restore its Secret descriptor."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _client(self) -> MrScraperClient:
        api_key = self.api_key.resolve_value()
        if api_key is None:
            msg = "The MrScraper API token did not resolve to a value."
            raise MrScraperError(msg)
        return MrScraperClient(
            api_key=api_key,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
        )
