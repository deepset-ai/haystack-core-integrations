# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from botocore.config import Config
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from haystack_integrations.common.s3.errors import S3ConfigurationError, S3Error
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session
from haystack_integrations.common.s3.utils import S3Storage


logger = logging.getLogger(__name__)


@component
class S3Downloader:
    """
    A component for downloading files from S3 to local filesystem with caching and concurrent downloads.
    Supports filtering by file extensions and returns documents with file paths.

    """

    def __init__(
        self,
        *,
        file_extensions: Optional[List[str]] = None,
        aws_access_key_id: Optional[Secret] = Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),  # noqa: B008
        aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
            "AWS_SECRET_ACCESS_KEY", strict=False
        ),
        aws_session_token: Optional[Secret] = Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),  # noqa: B008
        aws_region_name: Optional[Secret] = Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var("AWS_PROFILE", strict=False),  # noqa: B008
        boto3_config: Optional[Dict[str, Any]] = None,
        file_root_path: Optional[str] = None,
        max_workers: int = 32,
        max_cache_size: int = 100,
        input_file_meta_key: str = "file_name",
    ) -> None:
        """
        Initializes the S3Downloader with the provided parameters. The parameters are passed to the
        Amazon S3 client.

        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param boto3_config: The configuration for the boto3 client.
        :param file_root_path: The path where the file will be downloaded.
        :param file_extensions: The file extensions that are permitted to be downloaded.
        :param max_workers: The maximum number of workers to use for concurrent downloads.
        :param max_cache_size: The maximum number of files to cache.
        :raises S3ConfigurationError: If the AWS configuration is not correct.
        """
        
        # Set up download directory
        file_root_path = file_root_path or os.getenv("FILE_ROOT_PATH")
        
        if file_root_path is None:
            raise ValueError("file_root_path is not set. Please set the file_root_path parameter or the FILE_ROOT_PATH environment variable.")
        
        self.file_root_path = Path(file_root_path)
        
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region_name = aws_region_name
        self.aws_session_token = aws_session_token
        self.aws_profile_name = aws_profile_name
        self.boto3_config = boto3_config
        self.file_extensions = [e.lower() for e in file_extensions] if file_extensions else None
        self.max_workers = max_workers
        self.max_cache_size = max_cache_size
        self.storage : S3Storage = None
        self.input_file_meta_key = input_file_meta_key
        
        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        try:
            self.session = get_aws_session(
                aws_access_key_id=resolve_secret(aws_access_key_id),
                aws_secret_access_key=resolve_secret(aws_secret_access_key),
                aws_session_token=resolve_secret(aws_session_token),
                aws_region_name=resolve_secret(aws_region_name),
                aws_profile_name=resolve_secret(aws_profile_name),
            )
            self.config = Config(
                user_agent_extra="x-client-framework:haystack", **(self.boto3_config if self.boto3_config else {})
            )

        except Exception as exception:
            msg = (
                "Could not connect to Amazon S3. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise S3ConfigurationError(msg) from exception

    def warm_up(self) -> None:
        """Warm up the component by initializing the settings and storage."""
        if self.storage is None:
            self.file_root_path.mkdir(parents=True, exist_ok=True)
            self.storage = S3Storage.from_env(session=self.session, config=self.config)

    @component.output_types(documents=List[Document])
    def run(
        self,
        documents: List[Document],
    ) -> Dict[str, List[Document]]:
        """Download S3-linked files and return enriched `Document`s.

        :param documents: Documents to download. 
        :returns: A dictionary with:
            - `documents`: The downloaded `Document`s; each has `meta['file_path']` and may include `mime_type`.
        :raises S3Error: If a download attempt fails.
        :raises ValueError: If an S3 URL in inputs is invalid.
        """
        if self.file_extensions:
            filtered_documents = self._filter_documents_by_extensions(documents)
        else:
            filtered_documents = documents

        try:
            max_workers = min(self.max_workers, len(filtered_documents))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                iterable = executor.map(self.download_file, filtered_documents)
        finally:
            self.cleanup_cache(filtered_documents)

        downloaded_documents = [d for d in iterable if d is not None]
        return {"documents": downloaded_documents}

    def _filter_documents_by_extensions(self, documents: List[Document]) -> List[Document]:
        return [doc for doc in documents if Path(doc.meta.get(self.input_file_meta_key, "")).suffix.lower() in self.file_extensions]
        

    def download_file(self, document: Document) -> Optional[Document]:
        """
        Download a single S3 object into the local file system and enrich the `Document`.

        :param document: `Document` with file_id in the meta field.
        :returns:
            The same `Document` with `meta` containing the path of the
            downloaded file and the mime type or `None` if skipped.
        :raises S3Error: If the download or head request fails.
        """
        if self.storage is None:
            raise RuntimeError(
                f"The component {self.__class__.__name__} was not warmed up. "
                """Call "warm_up()" before calling "download_file()"."""
            )

        try:
            file_id = UUID(document.meta["file_id"])
        except KeyError:
            logger.warning(f"Document with ID {document.id!r} does not have a file_id in the meta field", exc_info=True)
            return None

        file_name = Path(document.meta.get(self.input_file_meta_key, ""))
        extension = file_name.suffix
        file_path = self.file_root_path / f"{file_id!s}{extension}"

        if file_path.is_file():
            # set access and modification time to now without redownloading the file
            file_path.touch()

        else:
            self.storage.download(file_name, file_path)

        document.meta["file_path"] = str(file_path)
        return document

    def cleanup_cache(self, documents: List[Document]) -> None:
        """
        Remove least-recently-accessed cache files when cache exceeds max_cache_size.
        """
        requested_ids = {
            str(abs(hash(str(doc.meta.get("cache_id", ""))))) for doc in documents if doc.meta.get("cache_id")
        }

        all_files = [p for p in self.file_root_path.iterdir() if p.is_file()]
        misses = [p for p in all_files if p.stem not in requested_ids]

        # cache budget = requested + misses; trim oldest misses if overflow
        overflow = len(misses) + len(requested_ids) - self.max_cache_size
        if overflow > 0:
            misses.sort(key=lambda p: p.stat().st_atime)
            for p in misses[:overflow]:
                try:
                    p.unlink()
                except Exception:
                    logger.warning(f"Failed to remove cache file {p}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            aws_access_key_id=self.aws_access_key_id.to_dict() if self.aws_access_key_id else None,
            aws_secret_access_key=self.aws_secret_access_key.to_dict() if self.aws_secret_access_key else None,
            aws_session_token=self.aws_session_token.to_dict() if self.aws_session_token else None,
            aws_region_name=self.aws_region_name.to_dict() if self.aws_region_name else None,
            aws_profile_name=self.aws_profile_name.to_dict() if self.aws_profile_name else None,
            file_root_path=str(self.file_root_path),
            max_workers=self.max_workers,
            max_cache_size=self.max_cache_size,
            file_extensions=self.file_extensions,
            input_file_meta_key=self.input_file_meta_key,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "S3Downloader":
        """
        Deserializes the component from a dictionary.
        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "aws_region_name", "aws_profile_name"],
        )
        return default_from_dict(cls, data)

