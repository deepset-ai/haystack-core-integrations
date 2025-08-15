# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Basic usage example for the S3 Downloader component in Amazon Bedrock integration.

This example demonstrates how to use the S3Downloader component to download
files from Amazon S3 and S3-compatible storage services, sharing the same AWS
authentication patterns as other Amazon Bedrock components.
"""

import os

from haystack import Secret
from haystack_integrations.components.downloader import S3Downloader


def main():
    """Demonstrate basic usage of the S3Downloader component."""

    print("Amazon Bedrock S3 Downloader - Basic Usage Example")
    print("=" * 52)

    # Example 1: Default credentials (environment variables or IAM role)
    print("\n--- Example 1: Default Credentials ---")
    
    # Initialize with default credentials (same as other Bedrock components)
    downloader = S3Downloader()
    print("✓ S3Downloader initialized with default credentials")

    # Test warm-up functionality
    try:
        downloader.warm_up()
        print("✓ Warm-up completed successfully")
    except Exception as e:
        print(f"⚠ Warm-up failed (expected if no AWS credentials): {e}")

    # Example 2: Explicit credentials (matching Bedrock pattern)
    print("\n--- Example 2: Explicit Credentials ---")
    
    # Check if AWS credentials are available in environment
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    if aws_key and aws_secret:
        print("✓ AWS credentials found in environment variables")
        
        downloader_with_creds = S3Downloader(
            aws_access_key_id=Secret.from_token(aws_key),
            aws_secret_access_key=Secret.from_token(aws_secret),
            aws_region=aws_region
        )
        
        try:
            downloader_with_creds.warm_up()
            print("✓ Explicit credentials validated successfully")
        except Exception as e:
            print(f"✗ Credential validation failed: {e}")
    else:
        print("⚠ No AWS credentials in environment variables")
        print("  Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to test explicit credentials")

    # Example 3: AWS Profile support (same as Bedrock components)
    print("\n--- Example 3: AWS Profile Support ---")
    
    # Using AWS profile (useful when working with multiple AWS accounts)
    profile_downloader = S3Downloader(
        aws_profile_name="default",  # or your specific profile name
        aws_region="us-west-2"
    )
    print("✓ S3Downloader configured with AWS profile")

    # Example 4: Test S3 URL validation
    print("\n--- Example 4: URL Validation ---")
    
    test_urls = [
        "s3://my-bucket/document.txt",  # Valid
        "s3://bucket/path/to/file.pdf",  # Valid
        "s3://bucket/",  # Invalid (no key)
        "s3://",  # Invalid (no bucket)
        "http://example.com/file.txt",  # Invalid (not S3)
    ]

    for url in test_urls:
        try:
            # This will validate the URL format but won't actually download
            # since we don't have access to these test buckets
            result = downloader.run(url)
            print(f"✓ {url} - Valid URL, download would succeed with proper access")
        except ValueError as e:
            print(f"✗ {url} - Invalid URL: {e}")
        except Exception as e:
            print(f"⚠ {url} - Valid URL format, but access failed: {type(e).__name__}")

    # Example 5: S3-compatible service configuration
    print("\n--- Example 5: S3-Compatible Services ---")
    
    # Example configuration for MinIO or other S3-compatible services
    minio_downloader = S3Downloader(
        aws_access_key_id=Secret.from_token("minioadmin"),
        aws_secret_access_key=Secret.from_token("minioadmin"),
        endpoint_url="http://localhost:9000",
        aws_region="us-east-1"
    )
    print("✓ S3-compatible service downloader configured (MinIO example)")
    print("  Note: This requires a running MinIO instance on localhost:9000")

    # Example 6: Integration with Bedrock ecosystem
    print("\n--- Example 6: Bedrock Integration ---")
    
    print("The S3Downloader can be used alongside other Amazon Bedrock components:")
    print("• Same AWS credential management patterns")
    print("• Shared boto3 and botocore dependencies")
    print("• Consistent authentication flow")
    print("• Can download files to feed into Bedrock generators/embedders")
    
    try:
        from haystack import Pipeline
        from haystack.components.converters import TextFileToDocument

        # Create pipeline combining S3 download with document processing
        pipeline = Pipeline()
        pipeline.add_component("s3_downloader", S3Downloader())
        pipeline.add_component("converter", TextFileToDocument())

        # Connect components
        pipeline.connect("s3_downloader.content", "converter.sources")

        print("✓ Pipeline created successfully")
        print("  Components: S3Downloader -> TextFileToDocument")
        print("  Ready to process S3 files for Bedrock models")
        
    except ImportError as e:
        print(f"⚠ Pipeline example requires additional dependencies: {e}")

    # Example 7: Metadata demonstration
    print("\n--- Example 7: Rich Metadata ---")
    
    print("S3Downloader provides comprehensive metadata:")
    print("- filename: Extracted from S3 key")
    print("- content_type: MIME type from S3")
    print("- size: File size in bytes")
    print("- s3_bucket: Source bucket name")
    print("- s3_key: Source object key")
    print("- s3_metadata: Custom S3 metadata")
    print("- etag: S3 ETag for versioning")
    print("- last_modified: File modification timestamp")
    print("- checksum: MD5 hash of content")
    print("- AWS region and other metadata")

    print("\n--- Basic usage example completed ---")
    print("\nTo test with real S3 files:")
    print("1. Set up AWS credentials (same as for Bedrock components)")
    print("2. Create an S3 bucket and upload a test file")
    print("3. Use downloader.run('s3://your-bucket/your-file.txt')")
    print("4. Use the downloaded content with Bedrock generators or embedders")


if __name__ == "__main__":
    main() 