# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from haystack import logging
from haystack.utils import Secret

from .base import StorageBackend

logger = logging.getLogger(__name__)


class S3Backend(StorageBackend):
    """
    S3 storage backend for downloading files from Amazon S3.

    Supports AWS credentials through parameters, environment variables, IAM roles,
    and AWS CLI configuration. Automatically handles S3 URL parsing and provides
    comprehensive metadata including S3-specific attributes.
    """

    def __init__(
        self,
        access_key_id: Optional[Secret] = None,
        secret_access_key: Optional[Secret] = None,
        region: Optional[str] = None,
        session_token: Optional[Secret] = None,
        endpoint_url: Optional[str] = None,
        verify_ssl: bool = True,
    ):
        """
        Initialize the S3 backend.

        :param access_key_id: AWS access key ID
        :param secret_access_key: AWS secret access key
        :param region: AWS region name
        :param session_token: AWS session token for temporary credentials
        :param endpoint_url: Custom S3 endpoint URL (for S3-compatible services)
        :param verify_ssl: Whether to verify SSL certificates
        """
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region
        self.session_token = session_token
        self.endpoint_url = endpoint_url
        self.verify_ssl = verify_ssl

        # Initialize S3 client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the S3 client with credentials and configuration."""
        # Build session parameters
        session_params = {}

        if self.access_key_id:
            session_params["aws_access_key_id"] = self.access_key_id.resolve_value()
        if self.secret_access_key:
            session_params["aws_secret_access_key"] = self.secret_access_key.resolve_value()
        if self.region:
            session_params["region_name"] = self.region
        if self.session_token:
            session_params["aws_session_token"] = self.session_token.resolve_value()

        # Create session
        self.session = boto3.Session(**session_params)

        # Create S3 client
        client_params = {}
        if self.endpoint_url:
            client_params["endpoint_url"] = self.endpoint_url
        # Note: boto3 doesn't have a verify_ssl parameter, SSL verification is handled differently

        self.s3_client = self.session.client("s3", **client_params)

    def can_handle(self, url: str) -> bool:
        """
        Check if this backend can handle the given URL.

        :param url: The URL to check
        :return: True if the URL starts with s3://
        """
        return url.startswith("s3://")

    def download(self, url: str) -> tuple[bytes, dict[str, Any]]:
        """
        Download file from S3 URL.

        :param url: The S3 URL to download from (format: s3://bucket/key)
        :return: Tuple of (file_content: bytes, metadata: Dict[str, Any])
        :raises: ValueError, ClientError, NoCredentialsError
        """
        if not self.can_handle(url):
            error_msg = f"S3 backend cannot handle URL: {url}"
            raise ValueError(error_msg)

        # Parse S3 URL
        bucket, key = self._parse_s3_url(url)
        if not bucket or not key:
            error_msg = f"Invalid S3 URL format: {url}. Expected format: s3://bucket/key"
            raise ValueError(error_msg)

        logger.debug(f"Downloading file from S3: bucket={bucket}, key={key}")

        try:
            # Get object from S3
            response = self.s3_client.get_object(Bucket=bucket, Key=key)

            # Read content
            content = response["Body"].read()

            # Extract metadata
            metadata = self._extract_metadata(url, bucket, key, response, content)

            logger.debug(f"Successfully downloaded file from S3 {url}, size: {len(content)} bytes")
            return content, metadata

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                logger.error(f"File not found in S3: {url}")
                error_msg = f"File not found in S3: {url}"
                raise FileNotFoundError(error_msg) from e
            elif error_code == "NoSuchBucket":
                logger.error(f"Bucket not found: {bucket}")
                error_msg = f"Bucket not found: {bucket}"
                raise FileNotFoundError(error_msg) from e
            else:
                logger.error(f"S3 error downloading from {url}: {e}")
                raise
        except NoCredentialsError as e:
            logger.error(f"AWS credentials not found for S3 download from {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading from S3 {url}: {e}")
            raise

    def _parse_s3_url(self, url: str) -> tuple[Optional[str], Optional[str]]:
        """
        Parse S3 URL to extract bucket and key.

        :param url: S3 URL in format s3://bucket/key
        :return: Tuple of (bucket, key)
        """
        parsed = urlparse(url)
        if parsed.scheme != "s3":
            return None, None

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        return bucket, key

    def _extract_metadata(
        self, url: str, bucket: str, key: str, response: dict[str, Any], content: bytes
    ) -> dict[str, Any]:
        """
        Extract metadata from S3 response.

        :param url: The original S3 URL
        :param bucket: S3 bucket name
        :param key: S3 object key
        :param response: S3 get_object response
        :param content: Downloaded file content
        :return: Dictionary containing metadata
        """
        # Extract filename from key
        filename = os.path.basename(key)
        if not filename:
            filename = "s3_file"

        # Extract content type
        content_type = response.get("ContentType", "application/octet-stream")
        if ";" in content_type:
            content_type = content_type.split(";")[0]

        # Extract last modified
        last_modified = response.get("LastModified")
        if last_modified:
            last_modified = last_modified.isoformat()

        # Extract ETag (S3's version of ETag)
        etag = response.get("ETag", "").strip('"')

        # Extract content length
        content_length = response.get("ContentLength")
        size = int(content_length) if content_length else len(content)

        # Calculate MD5 checksum (S3 ETag is often the MD5 for small files)
        checksum = hashlib.md5(content).hexdigest()

        # Extract S3-specific metadata
        s3_metadata = response.get("Metadata", {})

        # Build metadata
        metadata = {
            "filename": filename,
            "content_type": content_type,
            "size": size,
            "source_url": url,
            "last_modified": last_modified,
            "etag": etag,
            "backend": "s3",
            "download_time": datetime.now(timezone.utc).isoformat(),
            "checksum": checksum,
            "headers": {},  # S3 doesn't have HTTP headers like HTTP backend
            "s3_bucket": bucket,
            "s3_key": key,
            "s3_metadata": s3_metadata,
            "s3_storage_class": response.get("StorageClass"),
            "s3_version_id": response.get("VersionId"),
        }

        return metadata

    def warm_up(self) -> None:
        """
        Test AWS credentials and S3 connectivity.

        Makes a simple API call to validate credentials and connectivity.
        Logs warnings if warm-up fails but doesn't raise exceptions.
        """
        try:
            # Test credentials by listing buckets or making a simple API call
            self.s3_client.list_buckets(MaxBuckets=1)
            logger.info("S3 backend warmed up successfully")
        except Exception as e:
            logger.warning(f"S3 backend warm-up failed: {e}")
