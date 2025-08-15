# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import boto3
from moto import mock_s3

from haystack import Secret
from haystack_integrations.components.downloader import S3Downloader


@pytest.mark.integration
class TestS3DownloaderIntegration:
    """Integration tests for S3Downloader component in Amazon Bedrock ecosystem."""

    @mock_s3
    def test_bedrock_models_download(self):
        """Test downloading Bedrock model files from mocked S3."""
        # Create mock S3 bucket and object (simulating Bedrock model storage)
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "bedrock-models"
        key = "claude/v3/model-config.json"
        content = b'{"model": "claude-3", "version": "v1", "max_tokens": 100000}'
        
        # Create bucket and upload object
        s3_client.create_bucket(Bucket=bucket_name)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=content,
            ContentType="application/json",
            Metadata={"model_family": "claude", "provider": "anthropic"}
        )
        
        # Create S3Downloader (using Bedrock-style configuration)
        downloader = S3Downloader(
            aws_access_key_id=Secret.from_token("bedrock_key"),
            aws_secret_access_key=Secret.from_token("bedrock_secret"),
            aws_region="us-east-1"
        )
        
        # Download file
        result = downloader.run(f"s3://{bucket_name}/{key}")
        
        # Verify result
        assert result["content"] == content
        assert result["metadata"]["filename"] == "model-config.json"
        assert result["metadata"]["content_type"] == "application/json"
        assert result["metadata"]["size"] == len(content)
        assert result["metadata"]["s3_bucket"] == bucket_name
        assert result["metadata"]["s3_key"] == key
        assert result["metadata"]["s3_metadata"]["model_family"] == "claude"
        assert result["metadata"]["s3_metadata"]["provider"] == "anthropic"

    @mock_s3
    def test_training_data_download(self):
        """Test downloading training data from nested S3 path."""
        s3_client = boto3.client("s3", region_name="us-west-2")
        bucket_name = "bedrock-training-data"
        key = "datasets/2024/text/training_corpus.jsonl"
        content = b'{"text": "Sample training data for Bedrock fine-tuning"}\n'
        
        # Create bucket and upload object
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"}
        )
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=content,
            ContentType="application/x-jsonlines"
        )
        
        # Create S3Downloader
        downloader = S3Downloader(aws_region="us-west-2")
        
        # Download file
        result = downloader.run(f"s3://{bucket_name}/{key}")
        
        # Verify result
        assert result["content"] == content
        assert result["metadata"]["filename"] == "training_corpus.jsonl"
        assert result["metadata"]["content_type"] == "application/x-jsonlines"
        assert result["metadata"]["s3_key"] == key

    @mock_s3
    def test_warm_up_integration(self):
        """Test warm-up functionality with real S3 client (Bedrock pattern)."""
        # Create a bucket for warm-up to succeed
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="bedrock-bucket")
        
        downloader = S3Downloader()
        
        # Should not raise exception (similar to Bedrock component warm-up)
        downloader.warm_up()

    @mock_s3
    def test_multiple_model_files_download(self):
        """Test downloading multiple model files (typical Bedrock use case)."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "bedrock-models"
        
        # Create bucket
        s3_client.create_bucket(Bucket=bucket_name)
        
        # Upload multiple model files
        model_files = [
            ("claude/v3/config.json", b'{"model": "claude-3"}'),
            ("titan/v1/config.json", b'{"model": "titan-embed"}'),
            ("llama/v2/parameters.json", b'{"model": "llama-2"}')
        ]
        
        for key, content in model_files:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=content,
                ContentType="application/json"
            )
        
        downloader = S3Downloader()
        
        # Download all model files
        results = []
        for key, expected_content in model_files:
            result = downloader.run(f"s3://{bucket_name}/{key}")
            results.append(result)
            assert result["content"] == expected_content
        
        # Verify all downloads were successful
        assert len(results) == 3
        assert results[0]["metadata"]["filename"] == "config.json"
        assert results[1]["metadata"]["filename"] == "config.json"
        assert results[2]["metadata"]["filename"] == "parameters.json"

    @mock_s3
    def test_large_model_file_download(self):
        """Test downloading a large model file (common in Bedrock)."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "bedrock-models"
        key = "large-model/weights.bin"
        
        # Create large content (simulating model weights - 10MB)
        content = b"MODEL_DATA_" * (1024 * 1024)  # ~10MB
        
        # Create bucket and upload large object
        s3_client.create_bucket(Bucket=bucket_name)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=content,
            ContentType="application/octet-stream",
            Metadata={"model_type": "large_language_model"}
        )
        
        downloader = S3Downloader()
        result = downloader.run(f"s3://{bucket_name}/{key}")
        
        assert result["content"] == content
        assert result["metadata"]["size"] == len(content)
        assert result["metadata"]["s3_metadata"]["model_type"] == "large_language_model"
        assert len(result["content"]) > 10 * 1024 * 1024  # > 10MB

    @mock_s3
    def test_serialization_with_bedrock_config(self):
        """Test component serialization with Bedrock-style configuration."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "bedrock-test"
        key = "test-file.json"
        content = b'{"test": "serialization"}'
        
        # Create bucket and upload object
        s3_client.create_bucket(Bucket=bucket_name)
        s3_client.put_object(Bucket=bucket_name, Key=key, Body=content)
        
        # Create downloader with Bedrock-style config
        original_downloader = S3Downloader(
            aws_access_key_id=Secret.from_token("bedrock_key"),
            aws_secret_access_key=Secret.from_token("bedrock_secret"),
            aws_region="us-east-1",
            aws_profile_name="bedrock-profile"
        )
        
        # Serialize and deserialize
        data = original_downloader.to_dict()
        restored_downloader = S3Downloader.from_dict(data)
        
        # Test that restored downloader works
        result = restored_downloader.run(f"s3://{bucket_name}/{key}")
        
        assert result["content"] == content
        assert result["metadata"]["s3_bucket"] == bucket_name
        assert result["metadata"]["s3_key"] == key

    @mock_s3
    def test_bedrock_metadata_extraction(self):
        """Test metadata extraction for Bedrock use cases."""
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket_name = "bedrock-artifacts"
        key = "fine-tuning/jobs/2024/job-123/model-artifacts.tar.gz"
        content = b"compressed model artifacts data"
        
        # Create bucket and upload object with rich Bedrock metadata
        s3_client.create_bucket(Bucket=bucket_name)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=content,
            ContentType="application/gzip",
            Metadata={
                "job_id": "job-123",
                "base_model": "claude-3-sonnet",
                "training_status": "completed",
                "created_by": "bedrock-user",
                "fine_tuning_type": "instruction"
            },
            StorageClass="STANDARD"
        )
        
        downloader = S3Downloader()
        result = downloader.run(f"s3://{bucket_name}/{key}")
        
        metadata = result["metadata"]
        
        # Verify all metadata fields
        assert metadata["filename"] == "model-artifacts.tar.gz"
        assert metadata["content_type"] == "application/gzip"
        assert metadata["size"] == len(content)
        assert metadata["s3_bucket"] == bucket_name
        assert metadata["s3_key"] == key
        assert metadata["s3_metadata"]["job_id"] == "job-123"
        assert metadata["s3_metadata"]["base_model"] == "claude-3-sonnet"
        assert metadata["s3_metadata"]["training_status"] == "completed"
        assert metadata["s3_metadata"]["created_by"] == "bedrock-user"
        assert metadata["s3_metadata"]["fine_tuning_type"] == "instruction"
        assert metadata["backend"] == "s3"
        assert "checksum" in metadata
        assert "download_time" in metadata
        assert "etag" in metadata

    @mock_s3
    def test_custom_endpoint_bedrock_compatible(self):
        """Test S3Downloader with custom endpoint (for Bedrock-compatible services)."""
        # Create a downloader configured for custom endpoint
        downloader = S3Downloader(
            endpoint_url="http://localhost:9000",  # LocalStack or MinIO
            aws_access_key_id=Secret.from_token("bedrock_test_key"),
            aws_secret_access_key=Secret.from_token("bedrock_test_secret"),
            aws_region="us-east-1"
        )
        
        # Verify configuration matches Bedrock patterns
        assert downloader.endpoint_url == "http://localhost:9000"
        assert downloader.aws_access_key_id.resolve_value() == "bedrock_test_key"
        assert downloader.aws_secret_access_key.resolve_value() == "bedrock_test_secret"
        assert downloader.aws_region == "us-east-1" 