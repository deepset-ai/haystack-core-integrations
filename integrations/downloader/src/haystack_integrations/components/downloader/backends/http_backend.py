# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from haystack import logging
from haystack.utils import Secret

from .base import StorageBackend

logger = logging.getLogger(__name__)


class HTTPBackend(StorageBackend):
    """
    HTTP/HTTPS storage backend for downloading files from web servers.

    Supports basic authentication, bearer token authentication, and custom headers.
    Automatically handles redirects and provides comprehensive metadata.
    """

    def __init__(
        self,
        auth_token: Optional[Secret] = None,
        username: Optional[Secret] = None,
        password: Optional[Secret] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 30.0,
        max_redirects: int = 5,
        verify_ssl: bool = True,
    ):
        """
        Initialize the HTTP backend.

        :param auth_token: Bearer token for authentication
        :param username: Username for basic authentication
        :param password: Password for basic authentication
        :param headers: Additional HTTP headers to include in requests
        :param timeout: Request timeout in seconds
        :param max_redirects: Maximum number of redirects to follow
        :param verify_ssl: Whether to verify SSL certificates
        """
        self.auth_token = auth_token
        self.username = username
        self.password = password
        self.headers = headers or {}
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.verify_ssl = verify_ssl

        # Initialize HTTP client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the HTTP client with authentication and configuration."""
        # Build headers
        headers = self.headers.copy()
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token.resolve_value()}"

        # Build auth tuple for basic authentication
        auth = None
        if self.username and self.password:
            username = self.username.resolve_value()
            password = self.password.resolve_value()
            if isinstance(username, (str, bytes)) and isinstance(password, (str, bytes)):
                auth = (username, password)

        # Create client
        self.client = httpx.Client(
            headers=headers,
            auth=auth,
            timeout=self.timeout,
            follow_redirects=True,
            max_redirects=self.max_redirects,
            verify=self.verify_ssl,
        )

    def can_handle(self, url: str) -> bool:
        """
        Check if this backend can handle the given URL.

        :param url: The URL to check
        :return: True if the URL starts with http:// or https://
        """
        return url.startswith(("http://", "https://"))

    def download(self, url: str) -> tuple[bytes, dict[str, Any]]:
        """
        Download file from HTTP/HTTPS URL.

        :param url: The URL to download from
        :return: Tuple of (file_content: bytes, metadata: Dict[str, Any])
        :raises: httpx.HTTPError, httpx.RequestError, httpx.TimeoutException
        """
        if not self.can_handle(url):
            error_msg = f"HTTP backend cannot handle URL: {url}"
            raise ValueError(error_msg)

        logger.debug(f"Downloading file from HTTP URL: {url}")

        try:
            response = self.client.get(url)
            response.raise_for_status()

            # Extract metadata
            metadata = self._extract_metadata(url, response)

            logger.debug(f"Successfully downloaded file from {url}, size: {len(response.content)} bytes")
            return response.content, metadata

        except httpx.HTTPError as e:
            logger.error(f"HTTP error downloading from {url}: {e}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error downloading from {url}: {e}")
            raise
        except httpx.TimeoutException as e:
            logger.error(f"Timeout downloading from {url}: {e}")
            raise

    def _extract_metadata(self, url: str, response: httpx.Response) -> dict[str, Any]:
        """
        Extract metadata from HTTP response.

        :param url: The original URL
        :param response: The HTTP response object
        :return: Dictionary containing metadata
        """
        # Parse URL to extract filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = "downloaded_file"

        # Extract content type
        content_type = response.headers.get("content-type", "application/octet-stream")
        if ";" in content_type:
            content_type = content_type.split(";")[0]

        # Extract last modified
        last_modified = response.headers.get("last-modified")

        # Extract ETag
        etag = response.headers.get("etag")

        # Extract content length
        content_length = response.headers.get("content-length")
        size = int(content_length) if content_length else len(response.content)

        # Build metadata
        metadata = {
            "filename": filename,
            "content_type": content_type,
            "size": size,
            "source_url": url,
            "last_modified": last_modified,
            "etag": etag,
            "backend": "http",
            "download_time": datetime.now(timezone.utc).isoformat(),
            "checksum": None,  # HTTP doesn't provide checksums by default
            "headers": dict(response.headers),
        }

        return metadata

    def warm_up(self) -> None:
        """
        Test HTTP client configuration and authentication.

        Makes a simple HEAD request to validate client configuration.
        Logs warnings if warm-up fails but doesn't raise exceptions.
        """
        try:
            # Test with a simple HEAD request to validate client configuration
            if self.auth_token or self.username:
                # Test authenticated client if credentials are provided
                test_response = self.client.head("https://www.google.com")
                test_response.raise_for_status()
            else:
                # Test basic client configuration
                test_response = self.client.head("https://www.google.com")
                test_response.raise_for_status()

            logger.info("HTTP backend warmed up successfully")
        except Exception as e:
            logger.warning(f"HTTP backend warm-up failed: {e}")

    def __del__(self) -> None:
        """Clean up HTTP client on deletion."""
        if hasattr(self, "client"):
            try:
                self.client.close()
            except Exception as e:
                logger.debug(f"Error closing HTTP client: {e}")
