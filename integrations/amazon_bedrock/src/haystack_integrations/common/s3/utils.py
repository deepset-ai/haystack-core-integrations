# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Optional

from boto3.session import Session
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

from haystack_integrations.common.s3.errors import S3ConfigurationError, S3StorageError


@dataclass
class S3Storage:
    """This class provides a storage class for downloading files from an AWS S3 bucket."""

    def __init__(
        self,
        s3_bucket: str,
        session: Session,
        s3_prefix: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        """
        Initializes the S3Storage object with the provided parameters.

        :param s3_bucket: The name of the S3 bucket to download files from.
        :param session: The session to use for the S3 client.
        :param s3_prefix: The optional prefix of the files in the S3 bucket.
        Can be used to specify folder or naming structure.
            For example, if the file is in the folder "folder/subfolder/file.txt",
            the s3_prefix should be "folder/subfolder/". If the file is in the root of the S3 bucket,
            the s3_prefix should be None.
        :param endpoint_url: The endpoint URL of the S3 bucket to download files from.
        :param config: The configuration to use for the S3 client.
        """

        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.endpoint_url = endpoint_url
        self.session = session
        self.config = config

        try:
            self._client = self.session.client("s3", endpoint_url=self.endpoint_url, config=self.config)
        except Exception as e:
            msg = f"Failed to create S3 session client: {e}"
            raise S3ConfigurationError(msg) from e

    def download(self, key: str, local_file_path: Path) -> None:
        """Download a file from S3.

        :param key: The key of the file to download.
        :param local_file_path: The folder path to download the file to.
        It will be created if it does not exist. The file will be downloaded to
        the folder with the same name as the key.
        :raises S3ConfigurationError: If the S3 session client cannot be created.
        :raises S3StorageError: If the file does not exist in the S3 bucket
        or the file cannot be downloaded.
        """

        if self.s3_prefix:
            s3_key = f"{self.s3_prefix}{key}"
        else:
            s3_key = key

        try:
            self._client.download_file(self.s3_bucket, s3_key, str(local_file_path))

        except (NoCredentialsError, PartialCredentialsError) as e:
            msg = (
                f"Missing AWS credentials. Please check your AWS credentials (access key, secret key, region)."
                f"Error: {e}"
            )
            raise S3ConfigurationError(msg) from e

        except ClientError as e:
            error_code = int(e.response["Error"]["Code"])

            if error_code == HTTPStatus.FORBIDDEN:
                msg = (
                    f"Failed to access S3 bucket {self.s3_bucket!r}. "
                    f"Please check your AWS credentials (access key, secret key, region) and ensure "
                    f"they have the necessary S3 permissions. "
                    f"Error: {e}"
                )
                raise S3ConfigurationError(msg) from e

            elif error_code == HTTPStatus.NOT_FOUND:
                msg = f"The object {s3_key!r} does not exist in the S3 bucket {self.s3_bucket!r}. \n Error: {e}"
                raise S3StorageError(msg) from e
            else:
                msg = f"Failed to download file {s3_key!r} from S3. Error: {e}"
                raise S3StorageError(msg) from e

    @classmethod
    def from_env(cls, *, session: Session, config: Config) -> "S3Storage":
        """Create a S3Storage object from environment variables."""
        s3_bucket = os.getenv("S3_DOWNLOADER_BUCKET")
        if not s3_bucket:
            msg = (
                "Missing environment variable S3_DOWNLOADER_BUCKET."
                "Please set it to the name of the S3 bucket to download files from."
            )
            raise ValueError(msg)
        s3_prefix = os.getenv("S3_DOWNLOADER_PREFIX") or None
        endpoint_url = os.getenv("AWS_ENDPOINT_URL") or None
        return cls(
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            endpoint_url=endpoint_url,
            session=session,
            config=config,
        )
