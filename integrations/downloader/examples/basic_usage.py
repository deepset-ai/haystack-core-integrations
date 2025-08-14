# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Basic usage example for the Haystack Downloader component.

This example demonstrates how to use the Downloader component to download
files from different storage backends, including real files from the internet.
"""

import os
import tempfile

from haystack_integrations.components.downloader import Downloader


def main():
    """Demonstrate basic usage of the Downloader component."""

    # Initialize the downloader with default backends
    # This will use environment variables for credentials if available
    downloader = Downloader()

    print("Downloader initialized successfully!")

    # Example 1: Download from HTTP/HTTPS - well-known files
    print("\n--- Example 1: HTTP Downloads ---")

    # Download robots.txt from a well-known site
    try:
        result = downloader.run("https://www.google.com/robots.txt")
        print(f"✓ Downloaded Google robots.txt: {result['metadata']['filename']}")
        print(f"  Size: {result['metadata']['size']} bytes")
        print(f"  Content type: {result['metadata']['content_type']}")
        print(f"  Content preview: {result['content'][:100].decode('utf-8', errors='ignore')}...")
    except Exception as e:
        print(f"✗ Google robots.txt download failed: {e}")

    # Download a simple text file from a reliable source
    try:
        result = downloader.run("https://raw.githubusercontent.com/deepset-ai/haystack/main/README.md")
        print(f"✓ Downloaded Haystack README: {result['metadata']['filename']}")
        print(f"  Size: {result['metadata']['size']} bytes")
        print(f"  Content type: {result['metadata']['content_type']}")
        print(f"  Content preview: {result['content'][:100].decode('utf-8', errors='ignore')}...")
    except Exception as e:
        print(f"✗ Haystack README download failed: {e}")

    # Download a JSON file from a reliable source
    try:
        result = downloader.run("https://api.github.com/zen")
        print(f"✓ Downloaded GitHub Zen API: {result['metadata']['filename']}")
        print(f"  Size: {result['metadata']['size']} bytes")
        print(f"  Content type: {result['metadata']['content_type']}")
        print(f"  Content preview: {result['content'][:100].decode('utf-8', errors='ignore')}...")
    except Exception as e:
        print(f"✗ GitHub Zen API download failed: {e}")

    # Example 2: Download from local filesystem
    print("\n--- Example 2: Local File Download ---")
    try:
        # Create a temporary file for demonstration
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("This is a test file for the downloader component.\n")
            f.write("It contains multiple lines of text.\n")
            f.write("The downloader will read this file and provide metadata.")
            temp_file_path = f.name

        # Download the local file
        result = downloader.run(f"file://{temp_file_path}")
        print(f"✓ Downloaded local file: {result['metadata']['filename']}")
        print(f"  Size: {result['metadata']['size']} bytes")
        print(f"  Content: {result['content'].decode('utf-8')}")
        print(f"  Local path: {result['metadata']['local_path']}")
        print(f"  Permissions: {result['metadata']['local_permissions']}")

        # Clean up
        os.remove(temp_file_path)
    except Exception as e:
        print(f"✗ Local file download failed: {e}")

    # Example 3: Test S3 backend (will fail without credentials, but tests the backend)
    print("\n--- Example 3: S3 Backend Test ---")
    try:
        # This will fail without AWS credentials, but it tests the backend initialization
        result = downloader.run("s3://example-bucket/example.txt")
        print(f"✓ Downloaded from S3: {result['metadata']['filename']}")
        print(f"  Bucket: {result['metadata']['s3_bucket']}")
        print(f"  Key: {result['metadata']['s3_key']}")
    except Exception as e:
        print(f"✗ S3 download failed (expected): {e}")
        print("  Note: S3 download requires proper AWS credentials and bucket access")

    # Example 4: Test warm-up functionality
    print("\n--- Example 4: Warm-up Test ---")
    try:
        downloader.warm_up()
        print("✓ Warm-up completed successfully")
    except Exception as e:
        print(f"✗ Warm-up failed: {e}")

    print("\n--- Basic usage example completed ---")


if __name__ == "__main__":
    main()
