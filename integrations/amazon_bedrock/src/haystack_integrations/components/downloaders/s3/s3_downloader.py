# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from botocore.config import Config
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

from haystack_integrations.common.amazon_bedrock.utils import get_aws_session
from haystack_integrations.common.s3.utils import S3Storage

logger = logging.getLogger(__name__)


@component
class S3Downloader:
    """
    A component for downloading files from AWS S3 Buckets to local filesystem.
    Supports filtering by file extensions.
    """

    def __init__(
        self,
        *,
        aws_access_key_id: Optional[Secret] = Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),  # noqa: B008
        aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
            "AWS_SECRET_ACCESS_KEY", strict=False
        ),
        aws_session_token: Optional[Secret] = Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),  # noqa: B008
        aws_region_name: Optional[Secret] = Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var("AWS_PROFILE", strict=False),  # noqa: B008
        boto3_config: Optional[Dict[str, Any]] = None,
        file_root_path: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        file_name_meta_key: str = "file_name",
        max_workers: int = 32,
        max_cache_size: int = 100,
        s3_key_generation_function: Optional[Callable[[Document], str]] = None,
    ) -> None:
        """
        Initializes the `S3Downloader` with the provided parameters.

        Note that the AWS credentials are not required if the AWS environment is configured correctly. These are loaded
        automatically from the environment or the AWS configuration file and do not need to be provided explicitly via
        the constructor. If the AWS environment is not configured users need to provide the AWS credentials via the
        constructor. Three required parameters are `aws_access_key_id`, `aws_secret_access_key`,
        and `aws_region_name`.

        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param boto3_config: The configuration for the boto3 client.
        :param file_root_path: The path where the file will be downloaded.
            Can be set through this parameter or the `FILE_ROOT_PATH` environment variable.
            If none of them is set, a `ValueError` is raised.
        :param file_extensions: The file extensions that are permitted to be downloaded.
            By default, all file extensions are allowed.
        :param max_workers: The maximum number of workers to use for concurrent downloads.
        :param max_cache_size: The maximum number of files to cache.
        :param file_name_meta_key: The name of the meta key that contains the file name to download. The file name
            will also be used to create local file path for download.
            By default, the `Document.meta["file_name"]` is used. If you want to use a
            different key in `Document.meta`, you can set it here.
        :param s3_key_generation_function: An optional function that generates the S3 key for the file to download.
            If not provided, the default behavior is to use `Document.meta[file_name_meta_key]`.
            The function must accept a `Document` object and return a string.
            If the environment variable `S3_DOWNLOADER_PREFIX` is set, its value will be automatically
            prefixed to the generated S3 key.
        :raises ValueError: If the `file_root_path` is not set through
            the constructor or the `FILE_ROOT_PATH` environment variable.

        """

        # Set up download directory
        file_root_path = file_root_path or os.getenv("FILE_ROOT_PATH")

        if file_root_path is None:
            msg = (
                "The path where files will be downloaded is not set. Please set the "
                "`file_root_path` init parameter or the `FILE_ROOT_PATH` environment variable."
            )
            raise ValueError(msg)

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
        self.file_name_meta_key = file_name_meta_key
        self.s3_key_generation_function = s3_key_generation_function

        self._storage: Optional[S3Storage] = None

        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        self._session = get_aws_session(
            aws_access_key_id=resolve_secret(aws_access_key_id),
            aws_secret_access_key=resolve_secret(aws_secret_access_key),
            aws_session_token=resolve_secret(aws_session_token),
            aws_region_name=resolve_secret(aws_region_name),
            aws_profile_name=resolve_secret(aws_profile_name),
        )
        self._config = Config(
            user_agent_extra="x-client-framework:haystack", **(self.boto3_config if self.boto3_config else {})
        )

    def warm_up(self) -> None:
        """Warm up the component by initializing the settings and storage."""
        if self._storage is None:
            self.file_root_path.mkdir(parents=True, exist_ok=True)
            self._storage = S3Storage.from_env(session=self._session, config=self._config)

    @component.output_types(documents=List[Document])
    def run(
        self,
        documents: List[Document],
    ) -> Dict[str, List[Document]]:
        """Download files from AWS S3 Buckets to local filesystem.

        Return enriched `Document`s with the path of the downloaded file.
        :param documents: Document containing the name of the file to download in the meta field.
        :returns: A dictionary with:
            - `documents`: The downloaded `Document`s; each has `meta['file_path']`.
        :raises S3Error: If a download attempt fails or the file does not exist in the S3 bucket.
        :raises ValueError: If the path where files will be downloaded is not set.
        """

        if self._storage is None:
            msg = f"The component {self.__class__.__name__} was not warmed up. Call 'warm_up()' before calling run()."
            raise RuntimeError(msg)

        filtered_documents = self._filter_documents_by_extensions(documents) if self.file_extensions else documents

        if not filtered_documents:
            return {"documents": []}

        try:
            max_workers = min(self.max_workers, len(filtered_documents) if filtered_documents else self.max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                iterable = executor.map(self._download_file, filtered_documents)
        finally:
            self._cleanup_cache(filtered_documents)

        downloaded_documents = [d for d in iterable if d is not None]
        return {"documents": downloaded_documents}

    def _filter_documents_by_extensions(self, documents: List[Document]) -> List[Document]:
        """Filter documents by file extensions."""
        if not self.file_extensions:
            return documents
        return [
            doc
            for doc in documents
            if Path(doc.meta.get(self.file_name_meta_key, "")).suffix.lower() in self.file_extensions
        ]

    def _download_file(self, document: Document) -> Optional[Document]:
        """
        Download a single file from AWS S3 Bucket to local filesystem.

        :param document: `Document` with the name of the file to download in the meta field.
        :returns:
            The same `Document` with `meta` containing the `file_path` of the
            downloaded file.
        :raises S3Error: If the download or head request fails or the file does not exist in the S3 bucket.
        """

        file_name = document.meta.get(self.file_name_meta_key)
        if not file_name:
            logger.warning(
                f"Document missing required file name metadata key '{self.file_name_meta_key}'. Skipping download."
            )
            return None

        file_path = self.file_root_path / Path(file_name)

        if file_path.is_file():
            # set access and modification time to now without redownloading the file
            file_path.touch()

        else:
            s3_key = self.s3_key_generation_function(document) if self.s3_key_generation_function else file_name
            # we know that _storage is not None after warm_up() is called, but mypy does not know that
            self._storage.download(key=s3_key, local_file_path=file_path)  # type: ignore[union-attr]

        document.meta["file_path"] = str(file_path)
        return document

    def _cleanup_cache(self, documents: List[Document]) -> None:
        """
        Remove least-recently-accessed cache files when cache exceeds `max_cache_size`.

        :param documents: List of Document objects being used containing `cache_id` metadata.
        """
        requested_ids = {
            str(abs(hash(str(doc.meta.get("cache_id", ""))))) for doc in documents if doc.meta.get("cache_id")
        }

        all_files = [p for p in self.file_root_path.iterdir() if p.is_file()]
        misses = [p for p in all_files if p.stem not in requested_ids]

        overflow = len(misses) + len(requested_ids) - self.max_cache_size
        if overflow > 0:
            misses.sort(key=lambda p: p.stat().st_atime)
            for p in misses[:overflow]:
                try:
                    p.unlink()
                except Exception as error:
                    logger.warning("Failed to remove cache file at {path} with error: {e}", path=p, e=error)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the component to a dictionary."""

        s3_key_generation_function_name = (
            serialize_callable(self.s3_key_generation_function) if self.s3_key_generation_function else None
        )

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
            file_name_meta_key=self.file_name_meta_key,
            s3_key_generation_function=s3_key_generation_function_name,
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
        s3_key_generation_function_name = data["init_parameters"].get("s3_key_generation_function")
        if s3_key_generation_function_name:
            data["init_parameters"]["s3_key_generation_function"] = deserialize_callable(
                s3_key_generation_function_name
            )
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "aws_region_name", "aws_profile_name"],
        )
        return default_from_dict(cls, data)
