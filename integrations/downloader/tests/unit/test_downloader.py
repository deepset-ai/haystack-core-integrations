# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Downloader component."""

import tempfile
from unittest.mock import patch

import pytest
from haystack.utils import Secret

from haystack_integrations.components.downloader import Downloader, StorageBackend


class MockStorageBackend(StorageBackend):
    """Mock storage backend for testing."""

    def __init__(self, can_handle_result=True):
        self.can_handle_result = can_handle_result
        self.download_called = False
        self.warm_up_called = False

    def download(self, url):
        self.download_called = True
        return b"mock content", {"filename": "mock.txt", "backend": "mock"}

    def can_handle(self, url):
        return self.can_handle_result

    def warm_up(self):
        self.warm_up_called = True


class TestDownloader:
    """Test cases for Downloader component."""

    def test_init_default_backends(self):
        """Test Downloader initialization with default backends."""
        downloader = Downloader()

        assert hasattr(downloader, "http_backend")
        assert hasattr(downloader, "s3_backend")
        assert hasattr(downloader, "local_backend")
        assert not hasattr(downloader, "storage_backend")

    def test_init_custom_backend(self):
        """Test Downloader initialization with custom backend."""
        custom_backend = MockStorageBackend()
        downloader = Downloader(storage_backend=custom_backend)

        assert downloader.storage_backend == custom_backend
        assert not hasattr(downloader, "http_backend")
        assert not hasattr(downloader, "s3_backend")
        assert not hasattr(downloader, "local_backend")

    def test_init_with_credentials(self):
        """Test Downloader initialization with credentials."""
        aws_key = Secret.from_token("test_key")
        aws_secret = Secret.from_token("test_secret")
        http_token = Secret.from_token("test_token")

        downloader = Downloader(
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            aws_region="us-west-2",
            http_auth_token=http_token,
        )

        assert downloader.s3_backend.access_key_id == aws_key
        assert downloader.s3_backend.secret_access_key == aws_secret
        assert downloader.s3_backend.region == "us-west-2"
        assert downloader.http_backend.auth_token == http_token

    def test_run_with_http_url(self):
        """Test run method with HTTP URL."""
        downloader = Downloader()

        with patch.object(downloader.http_backend, "download") as mock_download:
            mock_download.return_value = (b"content", {"filename": "test.txt"})

            result = downloader.run("https://example.com/test.txt")

            assert result["content"] == b"content"
            assert result["metadata"]["filename"] == "test.txt"
            mock_download.assert_called_once_with("https://example.com/test.txt")

    def test_run_with_s3_url(self):
        """Test run method with S3 URL."""
        downloader = Downloader()

        with patch.object(downloader.s3_backend, "download") as mock_download:
            mock_download.return_value = (b"content", {"filename": "test.txt"})

            result = downloader.run("s3://bucket/test.txt")

            assert result["content"] == b"content"
            assert result["metadata"]["filename"] == "test.txt"
            mock_download.assert_called_once_with("s3://bucket/test.txt")

    def test_run_with_file_url(self):
        """Test run method with file URL."""
        downloader = Downloader()

        with patch.object(downloader.local_backend, "download") as mock_download:
            mock_download.return_value = (b"content", {"filename": "test.txt"})

            result = downloader.run("file:///path/to/test.txt")

            assert result["content"] == b"content"
            assert result["metadata"]["filename"] == "test.txt"
            mock_download.assert_called_once_with("file:///path/to/test.txt")

    def test_run_with_custom_backend(self):
        """Test run method with custom backend."""
        custom_backend = MockStorageBackend()
        downloader = Downloader(storage_backend=custom_backend)

        result = downloader.run("custom://test.txt")

        assert result["content"] == b"mock content"
        assert result["metadata"]["filename"] == "mock.txt"
        assert custom_backend.download_called

    def test_run_invalid_url(self):
        """Test run method with invalid URL."""
        downloader = Downloader()

        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            downloader.run("invalid://test.txt")

    def test_run_empty_url(self):
        """Test run method with empty URL."""
        downloader = Downloader()

        with pytest.raises(ValueError, match="URL cannot be empty"):
            downloader.run("")

    def test_run_custom_backend_cannot_handle(self):
        """Test run method when custom backend cannot handle URL."""
        custom_backend = MockStorageBackend(can_handle_result=False)
        downloader = Downloader(storage_backend=custom_backend)

        with pytest.raises(ValueError, match="Custom backend cannot handle URL"):
            downloader.run("custom://test.txt")

    def test_warm_up_default_backends(self):
        """Test warm_up method with default backends."""
        downloader = Downloader()

        with (
            patch.object(downloader.http_backend, "warm_up") as mock_http_warmup,
            patch.object(downloader.s3_backend, "warm_up") as mock_s3_warmup,
            patch.object(downloader.local_backend, "warm_up") as mock_local_warmup,
        ):
            downloader.warm_up()

            mock_http_warmup.assert_called_once()
            mock_s3_warmup.assert_called_once()
            mock_local_warmup.assert_called_once()

    def test_warm_up_custom_backend(self):
        """Test warm_up method with custom backend."""
        custom_backend = MockStorageBackend()
        downloader = Downloader(storage_backend=custom_backend)

        downloader.warm_up()

        assert custom_backend.warm_up_called

    def test_warm_up_backend_failure(self):
        """Test warm_up method when backend fails."""
        downloader = Downloader()

        # Mock a backend that raises an exception during warm-up
        with patch.object(downloader.http_backend, "warm_up", side_effect=Exception("Warm-up failed")):
            # Should not raise exception, just log warning
            downloader.warm_up()

    def test_get_backend_for_url(self):
        """Test _get_backend_for_url method."""
        downloader = Downloader()

        # Test with default backends
        assert downloader._get_backend_for_url("https://example.com") == downloader.http_backend
        assert downloader._get_backend_for_url("s3://bucket/file") == downloader.s3_backend
        assert downloader._get_backend_for_url("file:///path/file") == downloader.local_backend

    def test_get_backend_for_url_custom(self):
        """Test _get_backend_for_url method with custom backend."""
        custom_backend = MockStorageBackend()
        downloader = Downloader(storage_backend=custom_backend)

        assert downloader._get_backend_for_url("custom://test") == custom_backend

    def test_to_dict_default_backends(self):
        """Test to_dict method with default backends."""
        downloader = Downloader(aws_region="us-west-2", local_base_path=tempfile.gettempdir())

        result = downloader.to_dict()

        assert "init_parameters" in result
        assert result["init_parameters"]["aws_region"] == "us-west-2"
        # Verify the base path is a valid temp directory
        # On macOS, temp paths can resolve differently, so check if it's a valid temp dir
        temp_dir = tempfile.gettempdir()
        actual_path = result["init_parameters"]["local_base_path"]
        assert actual_path in [
            temp_dir,
            temp_dir.replace("/private", ""),
            temp_dir.replace("/var/folders", "/private/var/folders"),
        ]

    def test_to_dict_custom_backend(self):
        """Test to_dict method with custom backend."""
        custom_backend = MockStorageBackend()
        downloader = Downloader(storage_backend=custom_backend)

        result = downloader.to_dict()

        assert "init_parameters" in result
        assert "storage_backend" in result["init_parameters"]

    def test_from_dict_default_backends(self):
        """Test from_dict method with default backends."""
        data = {
            "type": "haystack_integrations.components.downloader.downloader.Downloader",
            "init_parameters": {"aws_region": "us-west-2", "local_base_path": tempfile.gettempdir()},
        }

        downloader = Downloader.from_dict(data)

        assert downloader.s3_backend.region == "us-west-2"
        # Verify the base path is a valid temp directory
        # On macOS, temp paths can resolve differently, so check if it's a valid temp dir
        temp_dir = tempfile.gettempdir()
        actual_path = str(downloader.local_backend.base_path)
        assert actual_path in [
            temp_dir,
            temp_dir.replace("/private", ""),
            temp_dir.replace("/var/folders", "/private/var/folders"),
        ]

    def test_from_dict_custom_backend(self):
        """Test from_dict method with custom backend."""
        data = {
            "type": "haystack_integrations.components.downloader.downloader.Downloader",
            "init_parameters": {"storage_backend": MockStorageBackend()},
        }

        downloader = Downloader.from_dict(data)

        assert hasattr(downloader, "storage_backend")
