# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol


class StorageBackend(Protocol):
    """
    Protocol defining the interface that all storage backends must implement.

    This protocol ensures consistency across different storage implementations
    and allows users to create custom backends that integrate seamlessly
    with the Downloader component.
    """

    def download(self, url: str) -> tuple[bytes, dict[str, Any]]:
        """
        Download file and return content + metadata.

        :param url: The URL to download from
        :return: Tuple of (file_content: bytes, metadata: Dict[str, Any])
        :raises: Various exceptions depending on the backend (e.g., FileNotFoundError, ConnectionError)
        """
        ...

    def can_handle(self, url: str) -> bool:
        """
        Check if this backend can handle the given URL.

        :param url: The URL to check
        :return: True if this backend can handle the URL
        """
        ...

    def warm_up(self) -> None:
        """
        Warm up the backend by executing authentication and connection tests.

        This method is optional but recommended for cloud storage backends
        to validate credentials and establish connections early. It should
        not raise exceptions but log warnings if warm-up fails.
        """
        ...
