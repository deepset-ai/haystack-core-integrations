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
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)


@component
class S3Downloader:
    """
    Component for downloading files from Amazon S3 and S3-compatible storage services.

    This component integrates with the Amazon Bedrock ecosystem, sharing the same AWS
    authentication patterns and dependencies. It supports downloading files from S3
    with comprehensive metadata extraction and error handling.

    ### Usage Examples

    #### Basic Usage with Environment Variables
    ```python
    from haystack_integrations.components.downloader import S3Downloader

    # Uses environment variables for credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    downloader = S3Downloader()

    # Warm up before using (optional but recommended)
    downloader.warm_up()

    # Download from S3
    result = downloader.run("s3://my-bucket/document.txt")
    print(f"Downloaded: {result['metadata']['filename']}")
    print(f"Size: {result['metadata']['size']} bytes")
    ```

    #### Explicit Credential Configuration
    ```python
    from haystack import Secret
    from haystack_integrations.components.downloader import S3Downloader

    downloader = S3Downloader(
        aws_access_key_id=Secret.from_token("my_aws_key"),
        aws_secret_access_key=Secret.from_token("my_aws_secret"),
        aws_region="us-west-2"
    )

    # Warm up to validate credentials
    downloader.warm_up()

    # Download file
    result = downloader.run("s3://my-bucket/path/to/file.pdf")
    ```

    #### Pipeline Integration
    ```python
    from haystack import Pipeline
    from haystack_integrations.components.downloader import S3Downloader
    from haystack.components.converters import TextFileToDocument

    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_component("s3_downloader", S3Downloader())
    pipeline.add_component("converter", TextFileToDocument())

    # Connect components
    pipeline.connect("s3_downloader.content", "converter.sources")

    # Run pipeline
    result = pipeline.run({
        "s3_downloader": {"url": "s3://my-bucket/document.txt"}
    })

    # Access results
    document = result["converter"]["documents"][0]
    ```
    """

    def __init__(
        self,
        aws_access_key_id: Optional[Secret] = None,
        aws_secret_access_key: Optional[Secret] = None,
        aws_region: Optional[str] = None,
        aws_session_token: Optional[Secret] = None,
        aws_profile_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        verify_ssl: bool = True,
    ):
        """
        Initialize the S3Downloader component.

        :param aws_access_key_id: AWS access key ID
        :param aws_secret_access_key: AWS secret access key
        :param aws_region: AWS region name
        :param aws_session_token: AWS session token for temporary credentials
        :param aws_profile_name: AWS profile name for credentials
        :param endpoint_url: Custom S3 endpoint URL (for S3-compatible services)
        :param verify_ssl: Whether to verify SSL certificates
        """
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region
        self.aws_session_token = aws_session_token
        self.aws_profile_name = aws_profile_name
        self.endpoint_url = endpoint_url
        self.verify_ssl = verify_ssl

        # Initialize S3 client
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the S3 client with credentials and configuration."""
        # Build session parameters (following Amazon Bedrock patterns)
        session_params = {}

        if self.aws_access_key_id:
            session_params["aws_access_key_id"] = self.aws_access_key_id.resolve_value()
        if self.aws_secret_access_key:
            session_params["aws_secret_access_key"] = self.aws_secret_access_key.resolve_value()
        if self.aws_region:
            session_params["region_name"] = self.aws_region
        if self.aws_session_token:
            session_params["aws_session_token"] = self.aws_session_token.resolve_value()
        if self.aws_profile_name:
            session_params["profile_name"] = self.aws_profile_name

        # Create session
        self.session = boto3.Session(**session_params)

        # Create S3 client
        client_params = {}
        if self.endpoint_url:
            client_params["endpoint_url"] = self.endpoint_url
        # Note: SSL verification is handled at the boto3 level

        self.s3_client = self.session.client("s3", **client_params)

    def warm_up(self) -> None:
        """
        Test AWS credentials and S3 connectivity.

        Makes a simple API call to validate credentials and connectivity.
        Logs warnings if warm-up fails but doesn't raise exceptions.
        """
        try:
            # Test credentials by listing buckets or making a simple API call
            self.s3_client.list_buckets(MaxBuckets=1)
            logger.info("S3 downloader warmed up successfully")
        except Exception as e:
            logger.warning(f"S3 downloader warm-up failed: {e}")

    @component.output_types(content=bytes, metadata=dict[str, Any])
    def run(self, url: str) -> dict[str, Any]:
        """
        Download file from S3 URL and return content + metadata.

        :param url: The S3 URL to download from (format: s3://bucket/key)
        :return: Dictionary containing 'content' (bytes) and 'metadata' (dict)
        :raises: ValueError, FileNotFoundError, ClientError, NoCredentialsError
        """
        if not url:
            error_msg = "URL cannot be empty"
            raise ValueError(error_msg)

        if not self._can_handle(url):
            error_msg = f"S3Downloader cannot handle URL: {url}. Expected format: s3://bucket/key"
            raise ValueError(error_msg)

        logger.debug(f"Downloading file from S3: {url}")

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
            return {"content": content, "metadata": metadata}

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

    def _can_handle(self, url: str) -> bool:
        """
        Check if this component can handle the given URL.

        :param url: The URL to check
        :return: True if the URL starts with s3://
        """
        return url.startswith("s3://")

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
            "s3_bucket": bucket,
            "s3_key": key,
            "s3_metadata": s3_metadata,
            "s3_storage_class": response.get("StorageClass"),
            "s3_version_id": response.get("VersionId"),
        }

        return metadata

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_region=self.aws_region,
            aws_session_token=self.aws_session_token,
            aws_profile_name=self.aws_profile_name,
            endpoint_url=self.endpoint_url,
            verify_ssl=self.verify_ssl,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "S3Downloader":
        """Deserialize the component from a dictionary."""
        return default_from_dict(cls, data) 