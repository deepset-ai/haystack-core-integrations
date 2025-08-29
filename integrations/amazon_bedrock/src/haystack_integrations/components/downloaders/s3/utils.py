from http import HTTPStatus
from pathlib import Path
from typing import Optional

from boto3.session import Session
from botocore.exceptions import ClientError
import os
from dataclasses import dataclass


class StorageError(RuntimeError):
    """This exception is raised when an error occurs while interacting with a storage backend."""


class S3Storage():
    """This class provides a storage class for downloading indexable files from S3."""

    def __init__(
        self,

        bucket_name: str,
        s3_folder: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> None:

        self.bucket_name = bucket_name
        self.s3_folder = s3_folder
        self.endpoint_url = endpoint_url
        self.session = session or Session()

    def download(self, key: str, local_file_path: Path) -> None:
        """Download an indexable file from S3."""

        s3 = self.session.client("s3", endpoint_url=self.endpoint_url)
        if self.s3_folder:
            s3_key = f"{self.s3_folder}/{key}"
        else:
            s3_key = key

        try:
            s3.download_file(self.bucket_name, s3_key, str(local_file_path))
        except ClientError as e:
            error_code = int(e.response["Error"]["Code"])

            if error_code == HTTPStatus.NOT_FOUND:
                raise StorageError(
                    f"The object {s3_key!r} does not exist in the S3 bucket {self.bucket_name!r}."
                ) from e

            raise StorageError(f"Failed to download file {s3_key!r} from S3: {e}") from e

@dataclass
class S3DownloaderSettings:
    """Settings for the S3Downloader component."""

    s3_bucket: str
    s3_folder: Optional[str] = None
    aws_endpoint_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "S3DownloaderSettings":
        """Create a S3DownloaderSettings object from environment variables.

        :raises ValueError: Raised if the S3_BUCKET environment variable is not provided.
        :return: An instance of S3DownloaderSettings
        :rtype: S3DownloaderSettings
        """

        if not (s3_bucket := os.getenv("S3_BUCKET")):
            raise ValueError("S3_BUCKET environment variable not provided")

        if not (s3_folder := os.getenv("S3_FOLDER")):
            raise ValueError("S3_FOLDER environment variable not provided")

        return cls(
            s3_bucket=s3_bucket,
            s3_folder=s3_folder,
            aws_endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        )