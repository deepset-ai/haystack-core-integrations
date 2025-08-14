# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline integration example for the Haystack Downloader component.

This example demonstrates how to integrate the Downloader component into
a Haystack pipeline for document processing workflows.
"""

import os
import tempfile

from haystack import Pipeline

from haystack_integrations.components.downloader import Downloader


def create_download_pipeline():
    """Create a pipeline that downloads documents."""

    # Create pipeline
    pipeline = Pipeline()

    # Add components
    pipeline.add_component("downloader", Downloader())

    return pipeline


def process_http_document():
    """Process a document downloaded from HTTP."""
    print("--- Processing HTTP Document ---")

    pipeline = create_download_pipeline()

    # Run pipeline with HTTP URL
    result = pipeline.run(
        {"downloader": {"url": "https://raw.githubusercontent.com/deepset-ai/haystack/main/README.md"}}
    )

    # Access results
    download_result = result["downloader"]
    print(f"Document downloaded: {download_result['metadata']['filename']}")
    print(f"Content length: {len(download_result['content'])} bytes")
    print(f"Content preview: {download_result['content'][:100].decode('utf-8', errors='ignore')}...")

    return result


def process_local_document():
    """Process a local document."""
    print("\n--- Processing Local Document ---")

    # Create a temporary file for demonstration
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("This is a test document for pipeline integration.\n")
        f.write("It contains multiple lines of text.\n")
        f.write("The downloader will read this file and provide metadata.")
        temp_file_path = f.name

    try:
        pipeline = create_download_pipeline()

        # Run pipeline with local file URL
        result = pipeline.run({"downloader": {"url": f"file://{temp_file_path}"}})

        # Access results
        download_result = result["downloader"]
        print(f"Document downloaded: {download_result['metadata']['filename']}")
        print(f"Content length: {len(download_result['content'])} bytes")
        print(f"Content: {download_result['content'].decode('utf-8')}")

        return result

    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def process_multiple_documents():
    """Process multiple documents in sequence."""
    print("\n--- Processing Multiple Documents ---")

    pipeline = create_download_pipeline()

    # List of URLs to process
    urls = [
        "https://raw.githubusercontent.com/deepset-ai/haystack/main/README.md",
        "https://api.github.com/zen",
    ]

    all_results = []

    for url in urls:
        try:
            print(f"\nProcessing: {url}")
            result = pipeline.run({"downloader": {"url": url}})

            download_result = result["downloader"]
            all_results.append(download_result)
            print(f"✓ Successfully processed: {download_result['metadata']['filename']}")
            print(f"  Size: {download_result['metadata']['size']} bytes")
            print(f"  Content type: {download_result['metadata']['content_type']}")

        except Exception as e:
            print(f"✗ Failed to process {url}: {e}")

    print(f"\nTotal documents processed: {len(all_results)}")
    return all_results


def test_pipeline_warm_up():
    """Test pipeline warm-up functionality."""
    print("\n--- Testing Pipeline Warm-up ---")

    pipeline = create_download_pipeline()

    # Get the downloader component
    downloader = pipeline.get_component("downloader")

    try:
        # Warm up the component
        downloader.warm_up()
        print("✓ Pipeline warm-up completed successfully")
    except Exception as e:
        print(f"✗ Pipeline warm-up failed: {e}")


def main():
    """Run all pipeline integration examples."""
    print("Haystack Downloader Pipeline Integration Examples")
    print("=" * 50)

    try:
        # Example 1: Process HTTP document
        process_http_document()

        # Example 2: Process local document
        process_local_document()

        # Example 3: Process multiple documents
        process_multiple_documents()

        # Example 4: Test pipeline warm-up
        test_pipeline_warm_up()

        print("\n--- All pipeline examples completed successfully ---")

    except Exception as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    main()
