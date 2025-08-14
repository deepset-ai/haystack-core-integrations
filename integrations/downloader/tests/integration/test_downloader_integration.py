# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the Downloader component."""

import os
import tempfile

import pytest

from haystack_integrations.components.downloader import Downloader


@pytest.mark.integration
class TestDownloaderIntegration:
    """Integration tests for Downloader component."""

    def test_local_file_download(self):
        """Test downloading a local file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("This is a test file for integration testing.")
            temp_file_path = f.name

        try:
            # Initialize downloader with restricted base path
            downloader = Downloader(local_base_path=os.path.dirname(temp_file_path))

            # Download the file
            result = downloader.run(f"file://{temp_file_path}")

            # Verify results
            assert result["content"] == b"This is a test file for integration testing."
            assert result["metadata"]["filename"] == os.path.basename(temp_file_path)
            assert result["metadata"]["backend"] == "local"
            assert result["metadata"]["size"] > 0
            assert result["metadata"]["content_type"] == "text/plain"

        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_local_file_security(self):
        """Test that local file access is properly restricted."""
        downloader = Downloader(local_base_path=tempfile.gettempdir())

        # Try to access a file outside the base path
        with pytest.raises(ValueError, match="outside allowed base path"):
            downloader.run("file:///etc/passwd")

    def test_http_download_public_file(self):
        """Test downloading a public HTTP file."""
        downloader = Downloader()

        # Download a simple public file from a reliable source
        result = downloader.run("https://raw.githubusercontent.com/deepset-ai/haystack/main/README.md")

        # Verify results
        assert result["content"] is not None
        assert len(result["content"]) > 0
        assert result["metadata"]["backend"] == "http"
        assert (
            result["metadata"]["source_url"] == "https://raw.githubusercontent.com/deepset-ai/haystack/main/README.md"
        )
        assert result["metadata"]["size"] > 0

    def test_warm_up_functionality(self):
        """Test the warm-up functionality."""
        downloader = Downloader()

        # Warm up should not raise exceptions
        downloader.warm_up()

        # Component should still be usable after warm-up
        assert hasattr(downloader, "http_backend")
        assert hasattr(downloader, "s3_backend")
        assert hasattr(downloader, "local_backend")

    def test_component_serialization(self):
        """Test component serialization and deserialization."""
        # Create downloader with some configuration
        original_downloader = Downloader(
            aws_region="us-west-2", local_base_path=tempfile.gettempdir(), http_timeout=60.0
        )

        # Serialize
        data = original_downloader.to_dict()

        # Deserialize
        restored_downloader = Downloader.from_dict(data)

        # Verify configuration was preserved
        assert restored_downloader.s3_backend.region == "us-west-2"
        # Verify the base path is a valid temp directory
        # On macOS, temp paths can resolve differently, so check if it's a valid temp dir
        temp_dir = tempfile.gettempdir()
        actual_path = str(restored_downloader.local_backend.base_path)
        assert actual_path in [
            temp_dir,
            temp_dir.replace("/private", ""),
            temp_dir.replace("/var/folders", "/private/var/folders"),
        ]
        assert restored_downloader.http_backend.timeout == 60.0
