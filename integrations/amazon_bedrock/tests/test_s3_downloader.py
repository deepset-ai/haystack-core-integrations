import io
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from haystack import Secret

from haystack_integrations.components.downloader.s3_downloader import S3Downloader


class TestS3Downloader:
    def test_init_default(self, mock_boto3_session, set_env_variables):
        """Test default initialization uses environment variables."""
        downloader = S3Downloader()

        assert downloader.aws_access_key_id is None
        assert downloader.aws_secret_access_key is None
        assert downloader.aws_region is None
        assert downloader.aws_session_token is None
        assert downloader.aws_profile_name is None
        assert downloader.endpoint_url is None
        assert downloader.verify_ssl is True

        # Assert mocked boto3 session called exactly once
        mock_boto3_session.assert_called_once()

        # Assert mocked boto3 session was called with empty parameters (uses env vars)
        mock_boto3_session.assert_called_with()

    def test_init_with_credentials(self, mock_boto3_session):
        """Test initialization with explicit credentials."""
        downloader = S3Downloader(
            aws_access_key_id=Secret.from_token("test_key_id"),
            aws_secret_access_key=Secret.from_token("test_secret_key"),
            aws_region="us-west-2",
            aws_session_token=Secret.from_token("test_session_token"),
            aws_profile_name="test_profile",
            endpoint_url="https://custom-s3-endpoint.com",
            verify_ssl=False,
        )

        assert downloader.aws_access_key_id.resolve_value() == "test_key_id"
        assert downloader.aws_secret_access_key.resolve_value() == "test_secret_key"
        assert downloader.aws_region == "us-west-2"
        assert downloader.aws_session_token.resolve_value() == "test_session_token"
        assert downloader.aws_profile_name == "test_profile"
        assert downloader.endpoint_url == "https://custom-s3-endpoint.com"
        assert downloader.verify_ssl is False

        # Assert mocked boto3 session called exactly once
        mock_boto3_session.assert_called_once()

    def test_to_dict(self, mock_boto3_session):
        """Test serialization to dictionary."""
        downloader = S3Downloader(
            aws_access_key_id=Secret.from_token("test_key"),
            aws_secret_access_key=Secret.from_token("test_secret"),
            aws_region="us-east-1",
            endpoint_url="https://custom-endpoint.com",
            verify_ssl=False,
        )

        expected_dict = {
            "type": "haystack_integrations.components.downloader.s3_downloader.S3Downloader",
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_region": "us-east-1",
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "endpoint_url": "https://custom-endpoint.com",
                "verify_ssl": False,
            },
        }

        assert downloader.to_dict() == expected_dict

    def test_from_dict(self, mock_boto3_session):
        """Test deserialization from dictionary."""
        data = {
            "type": "haystack_integrations.components.downloader.s3_downloader.S3Downloader",
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_region": "us-west-2",
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "endpoint_url": "https://test-endpoint.com",
                "verify_ssl": True,
            },
        }

        downloader = S3Downloader.from_dict(data)

        assert downloader.aws_region == "us-west-2"
        assert downloader.endpoint_url == "https://test-endpoint.com"
        assert downloader.verify_ssl is True

    def test_warm_up_success(self, mock_boto3_session):
        """Test successful warm up."""
        downloader = S3Downloader()
        
        # Mock the s3_client.list_buckets method
        mock_s3_client = MagicMock()
        mock_s3_client.list_buckets.return_value = {"Buckets": []}
        downloader.s3_client = mock_s3_client

        # Should not raise any exception
        downloader.warm_up()
        
        # Verify list_buckets was called
        mock_s3_client.list_buckets.assert_called_once_with(MaxBuckets=1)

    def test_warm_up_failure(self, mock_boto3_session):
        """Test warm up with connection failure."""
        downloader = S3Downloader()
        
        # Mock the s3_client.list_buckets method to raise an exception
        mock_s3_client = MagicMock()
        mock_s3_client.list_buckets.side_effect = NoCredentialsError()
        downloader.s3_client = mock_s3_client

        # Should not raise exception but should log warning
        downloader.warm_up()
        
        # Verify list_buckets was called
        mock_s3_client.list_buckets.assert_called_once_with(MaxBuckets=1)

    def test_can_handle_valid_s3_url(self, mock_boto3_session):
        """Test _can_handle method with valid S3 URLs."""
        downloader = S3Downloader()
        
        assert downloader._can_handle("s3://bucket/key")
        assert downloader._can_handle("s3://my-bucket/path/to/file.txt")

    def test_can_handle_invalid_urls(self, mock_boto3_session):
        """Test _can_handle method with invalid URLs."""
        downloader = S3Downloader()
        
        assert not downloader._can_handle("http://example.com/file.txt")
        assert not downloader._can_handle("https://s3.amazonaws.com/bucket/key")
        assert not downloader._can_handle("ftp://example.com/file.txt")
        assert not downloader._can_handle("")

    def test_parse_s3_url_valid(self, mock_boto3_session):
        """Test _parse_s3_url method with valid URLs."""
        downloader = S3Downloader()
        
        bucket, key = downloader._parse_s3_url("s3://my-bucket/path/to/file.txt")
        assert bucket == "my-bucket"
        assert key == "path/to/file.txt"
        
        bucket, key = downloader._parse_s3_url("s3://test-bucket/simple-file.pdf")
        assert bucket == "test-bucket"
        assert key == "simple-file.pdf"

    def test_parse_s3_url_invalid(self, mock_boto3_session):
        """Test _parse_s3_url method with invalid URLs."""
        downloader = S3Downloader()
        
        bucket, key = downloader._parse_s3_url("http://example.com/file.txt")
        assert bucket is None
        assert key is None

    def test_run_empty_url(self, mock_boto3_session):
        """Test run method with empty URL."""
        downloader = S3Downloader()
        
        with pytest.raises(ValueError, match="URL cannot be empty"):
            downloader.run("")

    def test_run_invalid_url(self, mock_boto3_session):
        """Test run method with invalid URL."""
        downloader = S3Downloader()
        
        with pytest.raises(ValueError, match="S3Downloader cannot handle URL"):
            downloader.run("http://example.com/file.txt")

    def test_run_invalid_s3_url_format(self, mock_boto3_session):
        """Test run method with invalid S3 URL format."""
        downloader = S3Downloader()
        
        with pytest.raises(ValueError, match="Invalid S3 URL format"):
            downloader.run("s3://")

    def test_run_successful_download(self, mock_boto3_session):
        """Test successful file download."""
        downloader = S3Downloader()
        
        # Mock S3 response
        mock_response = {
            "Body": io.BytesIO(b"test file content"),
            "ContentType": "text/plain",
            "ContentLength": 17,
            "LastModified": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "ETag": '"9a0364b9e99bb480dd25e1f0284c8555"',
            "Metadata": {"custom-meta": "value"},
            "StorageClass": "STANDARD",
            "VersionId": "version123",
        }
        
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.return_value = mock_response
        downloader.s3_client = mock_s3_client
        
        result = downloader.run("s3://test-bucket/test-file.txt")
        
        # Verify the call
        mock_s3_client.get_object.assert_called_once_with(Bucket="test-bucket", Key="test-file.txt")
        
        # Verify the result
        assert result["content"] == b"test file content"
        assert result["metadata"]["filename"] == "test-file.txt"
        assert result["metadata"]["content_type"] == "text/plain"
        assert result["metadata"]["size"] == 17
        assert result["metadata"]["source_url"] == "s3://test-bucket/test-file.txt"
        assert result["metadata"]["backend"] == "s3"
        assert result["metadata"]["s3_bucket"] == "test-bucket"
        assert result["metadata"]["s3_key"] == "test-file.txt"
        assert result["metadata"]["etag"] == "9a0364b9e99bb480dd25e1f0284c8555"
        assert result["metadata"]["s3_storage_class"] == "STANDARD"
        assert result["metadata"]["s3_version_id"] == "version123"
        assert result["metadata"]["s3_metadata"] == {"custom-meta": "value"}
        assert "checksum" in result["metadata"]
        assert "download_time" in result["metadata"]
        assert "last_modified" in result["metadata"]

    def test_run_file_not_found(self, mock_boto3_session):
        """Test download with file not found error."""
        downloader = S3Downloader()
        
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
            operation_name="GetObject",
        )
        downloader.s3_client = mock_s3_client
        
        with pytest.raises(FileNotFoundError, match="File not found in S3"):
            downloader.run("s3://test-bucket/nonexistent-file.txt")

    def test_run_bucket_not_found(self, mock_boto3_session):
        """Test download with bucket not found error."""
        downloader = S3Downloader()
        
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "NoSuchBucket", "Message": "The specified bucket does not exist"}},
            operation_name="GetObject",
        )
        downloader.s3_client = mock_s3_client
        
        with pytest.raises(FileNotFoundError, match="Bucket not found"):
            downloader.run("s3://nonexistent-bucket/file.txt")

    def test_run_credentials_error(self, mock_boto3_session):
        """Test download with no credentials error."""
        downloader = S3Downloader()
        
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.side_effect = NoCredentialsError()
        downloader.s3_client = mock_s3_client
        
        with pytest.raises(NoCredentialsError):
            downloader.run("s3://test-bucket/file.txt")

    def test_run_access_denied(self, mock_boto3_session):
        """Test download with access denied error."""
        downloader = S3Downloader()
        
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            operation_name="GetObject",
        )
        downloader.s3_client = mock_s3_client
        
        with pytest.raises(ClientError):
            downloader.run("s3://test-bucket/file.txt")

    def test_run_unexpected_error(self, mock_boto3_session):
        """Test download with unexpected error."""
        downloader = S3Downloader()
        
        mock_s3_client = MagicMock()
        mock_s3_client.get_object.side_effect = Exception("Unexpected error")
        downloader.s3_client = mock_s3_client
        
        with pytest.raises(Exception, match="Unexpected error"):
            downloader.run("s3://test-bucket/file.txt")

    def test_extract_metadata_with_minimal_response(self, mock_boto3_session):
        """Test metadata extraction with minimal S3 response."""
        downloader = S3Downloader()
        
        # Minimal response without optional fields
        response = {
            "Body": io.BytesIO(b"content"),
        }
        content = b"test content"
        
        metadata = downloader._extract_metadata(
            "s3://bucket/key", "bucket", "key", response, content
        )
        
        assert metadata["filename"] == "key"
        assert metadata["content_type"] == "application/octet-stream"
        assert metadata["size"] == len(content)
        assert metadata["source_url"] == "s3://bucket/key"
        assert metadata["backend"] == "s3"
        assert metadata["s3_bucket"] == "bucket"
        assert metadata["s3_key"] == "key"
        assert "checksum" in metadata
        assert "download_time" in metadata

    def test_extract_metadata_no_filename(self, mock_boto3_session):
        """Test metadata extraction when key has no filename."""
        downloader = S3Downloader()
        
        response = {"Body": io.BytesIO(b"content")}
        content = b"test content"
        
        metadata = downloader._extract_metadata(
            "s3://bucket/path/", "bucket", "path/", response, content
        )
        
        assert metadata["filename"] == "s3_file"

    def test_extract_metadata_content_type_with_charset(self, mock_boto3_session):
        """Test metadata extraction with content type including charset."""
        downloader = S3Downloader()
        
        response = {
            "Body": io.BytesIO(b"content"),
            "ContentType": "text/html; charset=utf-8",
        }
        content = b"test content"
        
        metadata = downloader._extract_metadata(
            "s3://bucket/file.html", "bucket", "file.html", response, content
        )
        
        assert metadata["content_type"] == "text/html" 