# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the HTTP backend."""

from unittest.mock import Mock, patch

import pytest
from haystack.utils import Secret

from haystack_integrations.components.downloader.backends.http_backend import HTTPBackend


class TestHTTPBackend:
    """Test cases for HTTPBackend."""

    def test_init_default_values(self):
        """Test HTTPBackend initialization with default values."""
        backend = HTTPBackend()

        assert backend.auth_token is None
        assert backend.username is None
        assert backend.password is None
        assert backend.headers == {}
        assert backend.timeout == 30.0
        assert backend.max_redirects == 5
        assert backend.verify_ssl is True

    def test_init_with_credentials(self):
        """Test HTTPBackend initialization with credentials."""
        auth_token = Secret.from_token("test_token")
        username = Secret.from_token("test_user")
        password = Secret.from_token("test_pass")
        headers = {"User-Agent": "Test/1.0"}

        backend = HTTPBackend(
            auth_token=auth_token,
            username=username,
            password=password,
            headers=headers,
            timeout=60.0,
            max_redirects=10,
            verify_ssl=False,
        )

        assert backend.auth_token == auth_token
        assert backend.username == username
        assert backend.password == password
        assert backend.headers == headers
        assert backend.timeout == 60.0
        assert backend.max_redirects == 10
        assert backend.verify_ssl is False

    def test_can_handle_http_urls(self):
        """Test can_handle method with HTTP URLs."""
        backend = HTTPBackend()

        assert backend.can_handle("http://example.com/file.txt") is True
        assert backend.can_handle("https://example.com/file.pdf") is True
        assert backend.can_handle("ftp://example.com/file.zip") is False
        assert backend.can_handle("s3://bucket/file.txt") is False
        assert backend.can_handle("file:///path/to/file.txt") is False

    @patch("httpx.Client")
    def test_download_success(self, mock_client_class):
        """Test successful download."""
        # Mock response
        mock_response = Mock()
        mock_response.content = b"test content"
        mock_response.headers = {
            "content-type": "text/plain",
            "content-length": "12",
            "last-modified": "Wed, 21 Oct 2015 07:28:00 GMT",
            "etag": '"abc123"',
        }
        mock_response.raise_for_status.return_value = None

        # Mock client
        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        backend = HTTPBackend()
        content, metadata = backend.download("https://example.com/test.txt")

        assert content == b"test content"
        assert metadata["filename"] == "test.txt"
        assert metadata["content_type"] == "text/plain"
        assert metadata["size"] == 12
        assert metadata["backend"] == "http"
        assert metadata["etag"] == '"abc123"'

    def test_download_invalid_url(self):
        """Test download with invalid URL."""
        backend = HTTPBackend()

        with pytest.raises(ValueError, match="HTTP backend cannot handle URL"):
            backend.download("s3://bucket/file.txt")

    @patch("httpx.Client")
    def test_download_http_error(self, mock_client_class):
        """Test download with HTTP error."""
        # Mock client that raises HTTP error
        mock_client = Mock()
        mock_client.get.side_effect = Exception("HTTP error")
        mock_client_class.return_value = mock_client

        backend = HTTPBackend()

        with pytest.raises(Exception, match="HTTP error"):
            backend.download("https://example.com/test.txt")

    @patch("httpx.Client")
    def test_warm_up_success(self, mock_client_class):
        """Test successful warm-up."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.head.return_value = mock_response
        mock_client_class.return_value = mock_client

        backend = HTTPBackend()
        backend.warm_up()  # Should not raise any exception

    @patch("httpx.Client")
    def test_warm_up_failure(self, mock_client_class):
        """Test warm-up failure."""
        # Mock client that raises exception
        mock_client = Mock()
        mock_client.head.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client

        backend = HTTPBackend()
        backend.warm_up()  # Should not raise exception, just log warning

    def test_extract_metadata(self):
        """Test metadata extraction."""
        backend = HTTPBackend()

        # Mock response
        mock_response = Mock()
        mock_response.headers = {
            "content-type": "application/pdf",
            "content-length": "1024",
            "last-modified": "Wed, 21 Oct 2015 07:28:00 GMT",
            "etag": '"def456"',
        }
        mock_response.content = b"x" * 1024

        url = "https://example.com/document.pdf"
        metadata = backend._extract_metadata(url, mock_response)

        assert metadata["filename"] == "document.pdf"
        assert metadata["content_type"] == "application/pdf"
        assert metadata["size"] == 1024
        assert metadata["source_url"] == url
        assert metadata["backend"] == "http"
        assert metadata["etag"] == '"def456"'

    def test_extract_metadata_no_filename(self):
        """Test metadata extraction with URL that has no filename."""
        backend = HTTPBackend()

        # Mock response
        mock_response = Mock()
        mock_response.headers = {}
        mock_response.content = b"content"

        url = "https://example.com/"
        metadata = backend._extract_metadata(url, mock_response)

        assert metadata["filename"] == "downloaded_file"
        assert metadata["content_type"] == "application/octet-stream"
