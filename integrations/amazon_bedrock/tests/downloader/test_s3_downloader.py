# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from botocore.exceptions import ClientError, NoCredentialsError

from haystack import Secret
from haystack_integrations.components.downloader import S3Downloader


class TestS3Downloader:
    """Test cases for S3Downloader component in Amazon Bedrock integration."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        downloader = S3Downloader()
        assert downloader.aws_access_key_id is None
        assert downloader.aws_secret_access_key is None
        assert downloader.aws_region is None
        assert downloader.aws_session_token is None
        assert downloader.aws_profile_name is None
        assert downloader.endpoint_url is None
        assert downloader.verify_ssl is True

    def test_init_with_credentials(self):
        """Test initialization with explicit credentials (Bedrock-style)."""
        access_key = Secret.from_token("test_key")
        secret_key = Secret.from_token("test_secret")
        
        downloader = S3Downloader(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_region="us-west-2",
            aws_profile_name="bedrock-profile",
            endpoint_url="https://custom-endpoint.com",
            verify_ssl=False
        )
        
        assert downloader.aws_access_key_id == access_key
        assert downloader.aws_secret_access_key == secret_key
        assert downloader.aws_region == "us-west-2"
        assert downloader.aws_profile_name == "bedrock-profile"
        assert downloader.endpoint_url == "https://custom-endpoint.com"
        assert downloader.verify_ssl is False

    @patch('boto3.Session')
    def test_init_client_with_profile(self, mock_session):
        """Test S3 client initialization with AWS profile (Bedrock pattern)."""
        mock_session_instance = Mock()
        mock_s3_client = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.return_value = mock_s3_client
        
        downloader = S3Downloader(
            aws_profile_name="bedrock-profile",
            aws_region="us-east-1"
        )
        
        # Verify session was created with profile
        mock_session.assert_called_once_with(
            profile_name="bedrock-profile",
            region_name="us-east-1"
        )
        
        # Verify S3 client was created
        mock_session_instance.client.assert_called_once_with("s3")

    @patch('boto3.Session')
    def test_init_client_full_credentials(self, mock_session):
        """Test S3 client initialization with full credentials."""
        mock_session_instance = Mock()
        mock_s3_client = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.return_value = mock_s3_client
        
        access_key = Secret.from_token("test_key")
        secret_key = Secret.from_token("test_secret")
        session_token = Secret.from_token("test_token")
        
        downloader = S3Downloader(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_region="us-east-1",
            aws_session_token=session_token,
            endpoint_url="https://s3.amazonaws.com"
        )
        
        # Verify session was created with correct parameters
        mock_session.assert_called_once_with(
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            region_name="us-east-1",
            aws_session_token="test_token"
        )
        
        # Verify S3 client was created with endpoint
        mock_session_instance.client.assert_called_once_with(
            "s3", 
            endpoint_url="https://s3.amazonaws.com"
        )

    def test_can_handle_valid_urls(self):
        """Test URL validation for valid S3 URLs."""
        downloader = S3Downloader()
        
        valid_urls = [
            "s3://bucket/file.txt",
            "s3://my-bucket/path/to/file.pdf",
            "s3://bedrock-docs/models/claude/config.json"
        ]
        
        for url in valid_urls:
            assert downloader._can_handle(url) is True

    def test_can_handle_invalid_urls(self):
        """Test URL validation for invalid URLs."""
        downloader = S3Downloader()
        
        invalid_urls = [
            "http://example.com/file.txt",
            "https://example.com/file.txt",
            "file:///path/to/file.txt",
            "bedrock://model/config",
            ""
        ]
        
        for url in invalid_urls:
            assert downloader._can_handle(url) is False

    def test_parse_s3_url_valid(self):
        """Test parsing valid S3 URLs."""
        downloader = S3Downloader()
        
        test_cases = [
            ("s3://bucket/file.txt", ("bucket", "file.txt")),
            ("s3://bedrock-models/claude/v1/config.json", ("bedrock-models", "claude/v1/config.json")),
            ("s3://training-data/documents/manual.pdf", ("training-data", "documents/manual.pdf"))
        ]
        
        for url, expected in test_cases:
            bucket, key = downloader._parse_s3_url(url)
            assert bucket == expected[0]
            assert key == expected[1]

    def test_run_empty_url(self):
        """Test run method with empty URL."""
        downloader = S3Downloader()
        
        with pytest.raises(ValueError, match="URL cannot be empty"):
            downloader.run("")

    def test_run_invalid_url(self):
        """Test run method with invalid URL."""
        downloader = S3Downloader()
        
        with pytest.raises(ValueError, match="S3Downloader cannot handle URL"):
            downloader.run("bedrock://model/config")

    @patch('boto3.Session')
    def test_run_successful_download(self, mock_session):
        """Test successful file download."""
        # Mock S3 client and response
        mock_session_instance = Mock()
        mock_s3_client = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.return_value = mock_s3_client
        
        # Mock S3 response
        mock_body = Mock()
        mock_body.read.return_value = b"bedrock model config content"
        
        mock_response = {
            "Body": mock_body,
            "ContentType": "application/json",
            "ContentLength": 27,
            "LastModified": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "ETag": '"bedrock123"',
            "Metadata": {"model": "claude", "version": "v1"}
        }
        
        mock_s3_client.get_object.return_value = mock_response
        
        downloader = S3Downloader()
        result = downloader.run("s3://bedrock-models/claude/config.json")
        
        # Verify S3 client was called correctly
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="bedrock-models",
            Key="claude/config.json"
        )
        
        # Verify result
        assert result["content"] == b"bedrock model config content"
        assert result["metadata"]["filename"] == "config.json"
        assert result["metadata"]["content_type"] == "application/json"
        assert result["metadata"]["size"] == 27
        assert result["metadata"]["s3_bucket"] == "bedrock-models"
        assert result["metadata"]["s3_key"] == "claude/config.json"
        assert result["metadata"]["etag"] == "bedrock123"
        assert result["metadata"]["backend"] == "s3"
        assert result["metadata"]["s3_metadata"]["model"] == "claude"

    @patch('boto3.Session')
    def test_warm_up_success(self, mock_session):
        """Test successful warm-up (matching Bedrock component patterns)."""
        mock_session_instance = Mock()
        mock_s3_client = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.return_value = mock_s3_client
        
        # Mock successful list_buckets call
        mock_s3_client.list_buckets.return_value = {"Buckets": []}
        
        downloader = S3Downloader()
        
        # Should not raise any exception
        downloader.warm_up()
        
        mock_s3_client.list_buckets.assert_called_once_with(MaxBuckets=1)

    @patch('boto3.Session')
    def test_warm_up_failure(self, mock_session):
        """Test warm-up failure handling."""
        mock_session_instance = Mock()
        mock_s3_client = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.return_value = mock_s3_client
        
        # Mock failed list_buckets call
        mock_s3_client.list_buckets.side_effect = NoCredentialsError()
        
        downloader = S3Downloader()
        
        # Should not raise exception but log warning
        downloader.warm_up()

    def test_extract_metadata_comprehensive(self):
        """Test comprehensive metadata extraction."""
        downloader = S3Downloader()
        
        # Mock S3 response with Bedrock-style metadata
        response = {
            "ContentType": "application/json; charset=utf-8",
            "ContentLength": 1024,
            "LastModified": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "ETag": '"bedrock-etag-123"',
            "Metadata": {
                "model": "claude-3",
                "version": "v1",
                "created-by": "bedrock-team"
            },
            "StorageClass": "STANDARD",
            "VersionId": "bedrock-version-123"
        }
        
        content = b"bedrock model configuration data"
        url = "s3://bedrock-models/claude/v3/config.json"
        bucket = "bedrock-models"
        key = "claude/v3/config.json"
        
        metadata = downloader._extract_metadata(url, bucket, key, response, content)
        
        assert metadata["filename"] == "config.json"
        assert metadata["content_type"] == "application/json"
        assert metadata["size"] == 1024
        assert metadata["source_url"] == url
        assert metadata["etag"] == "bedrock-etag-123"
        assert metadata["backend"] == "s3"
        assert metadata["s3_bucket"] == bucket
        assert metadata["s3_key"] == key
        assert metadata["s3_metadata"]["model"] == "claude-3"
        assert metadata["s3_metadata"]["version"] == "v1"
        assert metadata["s3_metadata"]["created-by"] == "bedrock-team"
        assert metadata["s3_storage_class"] == "STANDARD"
        assert metadata["s3_version_id"] == "bedrock-version-123"
        assert "checksum" in metadata
        assert "download_time" in metadata

    def test_to_dict_with_profile(self):
        """Test serialization with AWS profile (Bedrock pattern)."""
        access_key = Secret.from_token("bedrock_key")
        secret_key = Secret.from_token("bedrock_secret")
        
        downloader = S3Downloader(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_region="us-west-2",
            aws_profile_name="bedrock-profile",
            endpoint_url="https://s3.amazonaws.com"
        )
        
        data = downloader.to_dict()
        
        assert data["init_parameters"]["aws_access_key_id"] == access_key
        assert data["init_parameters"]["aws_secret_access_key"] == secret_key
        assert data["init_parameters"]["aws_region"] == "us-west-2"
        assert data["init_parameters"]["aws_profile_name"] == "bedrock-profile"
        assert data["init_parameters"]["endpoint_url"] == "https://s3.amazonaws.com"

    def test_from_dict_bedrock_style(self):
        """Test deserialization with Bedrock-style configuration."""
        access_key = Secret.from_token("bedrock_key")
        secret_key = Secret.from_token("bedrock_secret")
        
        data = {
            "type": "haystack_integrations.components.downloader.s3_downloader.S3Downloader",
            "init_parameters": {
                "aws_access_key_id": access_key,
                "aws_secret_access_key": secret_key,
                "aws_region": "us-west-2",
                "aws_profile_name": "bedrock-profile",
                "endpoint_url": "https://s3.amazonaws.com"
            }
        }
        
        downloader = S3Downloader.from_dict(data)
        
        assert downloader.aws_access_key_id == access_key
        assert downloader.aws_secret_access_key == secret_key
        assert downloader.aws_region == "us-west-2"
        assert downloader.aws_profile_name == "bedrock-profile"
        assert downloader.endpoint_url == "https://s3.amazonaws.com" 