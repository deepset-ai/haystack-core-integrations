# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse
from uuid import UUID

from botocore.config import Config
from botocore.exceptions import ClientError
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream, Document
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from haystack_integrations.common.amazon_bedrock.errors import AmazonS3ConfigurationError, AmazonS3Error
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

logger = logging.getLogger(__name__)


@component
class S3Downloader:
    """
    A component for downloading files from S3 to local filesystem with caching and concurrent downloads.
    Supports filtering by file extensions and returns documents with file paths.

    Usage example:
    ```python
    from haystack import Document
    from haystack.utils import Secret
    from haystack_integrations.components.downloader.s3 import S3Downloader

    import os
    from pathlib import Path

    downloader = S3Downloader(
        aws_profile_name=Secret.from_token("prof"),
        aws_region_name=Secret.from_token("eu-central-1"),
        download_dir=str(Path.cwd() / "s3_cache"),
        file_extensions=[".json"],
        sources_target_type="str",
    )

    docs = [
        Document(content="", meta={"s3_bucket": "bucket_name", "s3_key": "your_key", "file_name": "file_name"}),
    ]
    result = downloader.run(documents=docs)
    print("Downloaded docs:")
    for d in result["documents"]:
        print(f"- {d.meta.get('file_name')} -> {d.meta.get('file_path')}")

    ```
    """

    max_cache_size: int = 100

    def __init__(
        self,
        aws_access_key_id: Optional[Secret] = Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),  # noqa: B008
        aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
            "AWS_SECRET_ACCESS_KEY", strict=False
        ),
        aws_session_token: Optional[Secret] = Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),  # noqa: B008
        aws_region_name: Optional[Secret] = Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var("AWS_PROFILE", strict=False),  # noqa: B008
        verify_ssl: bool = True,
        boto3_config: Optional[Dict[str, Any]] = None,
        download_dir: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        sources_target_type: Literal["str", "pathlib.Path", "haystack.dataclasses.ByteStream"] = "str",
        max_workers: int = 32,
        max_cache_size: int = 100,
    ) -> None:
        """
        Initializes the S3Downloader with the provided parameters. The parameters are passed to the
        Amazon S3 client.

        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param verify_ssl: Whether to verify SSL.
        :param boto3_config: The configuration for the boto3 client.
        :param download_dir: The directory where the files will be downloaded.
        :param file_extensions: The file extensions that are permitted to be downloaded.
        :param sources_target_type: The target type for the sources.
            Can be "str", "pathlib.Path" or
            "haystack.dataclasses.ByteStream".
        :param max_workers: The maximum number of workers to use for concurrent downloads.
        :param max_cache_size: The maximum number of files to cache.
        :raises AmazonS3ConfigurationError: If the AWS configuration is not correct.
        """

        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region_name = aws_region_name
        self.aws_session_token = aws_session_token
        self.aws_profile_name = aws_profile_name
        self.verify_ssl = verify_ssl
        self.boto3_config = boto3_config
        self.file_extensions = [e.lower() for e in file_extensions] if file_extensions else None
        self.sources_target_type = sources_target_type
        self.max_workers = max_workers
        self.max_cache_size = max_cache_size

        # Set up download directory
        if download_dir:
            self.download_dir = Path(download_dir)
        else:
            self.download_dir = Path.cwd() / "s3_downloads"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.file_root_path: Path = self.download_dir

        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        try:
            session = get_aws_session(
                aws_access_key_id=resolve_secret(aws_access_key_id),
                aws_secret_access_key=resolve_secret(aws_secret_access_key),
                aws_session_token=resolve_secret(aws_session_token),
                aws_region_name=resolve_secret(aws_region_name),
                aws_profile_name=resolve_secret(aws_profile_name),
            )
            config = Config(
                user_agent_extra="x-client-framework:haystack", **(self.boto3_config if self.boto3_config else {})
            )

            self._client = session.client("s3", config=config)
        except Exception as exception:
            msg = (
                "Could not connect to Amazon S3. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonS3ConfigurationError(msg) from exception

    def warm_up(self) -> None:
        """Ensure local cache directory exists."""
        self.file_root_path.mkdir(parents=True, exist_ok=True)

    @component.output_types(
        documents=List[Document],
        sources=List[Union[str, Path, ByteStream]],
    )
    def run(
        self,
        documents: Optional[List[Document]] = None,
        sources: Optional[List[Union[ByteStream, UUID, str]]] = None,
    ) -> Dict[str, Union[List[Document], List[Union[str, Path, ByteStream]]]]:
        """Download S3-linked files and return enriched `Document`s and source paths.

        You can pass existing `Document`s (with S3 coordinates in `meta`) and/or `sources`
        (e.g., `s3://...` URLs or `ByteStream` with S3 metadata). Files are cached locally.

        :param documents: Documents to download. Each must include one of:
            - `meta['s3_bucket']` + `meta['s3_key']`, or
            - `meta['s3_url']` (e.g., `s3://bucket/key`), or
            - `meta['s3_key']` (with default bucket provided elsewhere).
            Non-conforming documents are skipped.
        :param sources: Optional heterogeneous inputs used to build `Document`s:
            - `str` with `s3://bucket/key`,
            - `ByteStream` whose `meta` contains `s3_url` or (`s3_bucket` + `s3_key`),
            - Other types are ignored with a warning.
        :returns: A dictionary with:
            - `documents`: The downloaded `Document`s; each has `meta['file_path']` and may include `mime_type`.
            - `sources`: The sources in the requested target type (`str`, `Path`, or `ByteStream`).
        :raises AmazonS3Error: If a download attempt fails.
        :raises ValueError: If an S3 URL in inputs is invalid.
        """
        # Ensure cache dir
        self.warm_up()

        requested: List[Document] = []

        if sources:
            requested.extend(self._build_source_documents(sources))

        if documents:
            requested.extend(documents)

        if not requested:
            return {"documents": [], "sources": []}

        requested = self._filter_documents_by_extensions(requested)

        if not requested:
            return {"documents": [], "sources": []}

        try:
            if len(requested) == 1:
                iterable = iter([self.download_file(requested[0])])
            else:
                max_workers = min(self.max_workers, len(requested))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    iterable = executor.map(self.download_file, requested)
        finally:
            self.cleanup_cache(requested)

        downloaded_documents = [d for d in iterable if d is not None]
        target_sources = self._to_sources(downloaded_documents, self.sources_target_type)
        return {"documents": downloaded_documents, "sources": target_sources}

    def _filter_documents_by_extensions(self, documents: List[Document]) -> List[Document]:
        if not self.file_extensions:
            return documents
        out: List[Document] = []
        for doc in documents:
            name = doc.meta.get("file_name") or doc.meta.get("s3_key") or ""
            if Path(name).suffix.lower() in self.file_extensions:
                out.append(doc)
        return out

    @staticmethod
    def _to_sources(
        documents: List[Document],
        sources_target_type: Literal["str", "pathlib.Path", "haystack.dataclasses.ByteStream"],
    ) -> List[Union[str, Path, ByteStream]]:
        file_paths = [doc.meta.get("file_path", "") for doc in documents if doc.meta.get("file_path")]
        if sources_target_type == "str":
            return file_paths
        elif sources_target_type == "pathlib.Path":
            return [Path(p) for p in file_paths]
        elif sources_target_type == "haystack.dataclasses.ByteStream":
            return [
                ByteStream.from_file_path(
                    Path(doc.meta.get("file_path", "")),
                    mime_type=doc.meta.get("mime_type"),
                    meta=doc.meta,
                )
                for doc in documents
            ]

    def _build_source_documents(self, sources: List[Union[ByteStream, UUID, str]]) -> List[Document]:
        """
        Builds the documents from the provided sources.

        :param sources: Contains the information about the files to download. Can include:
            - str: S3 URL in the form `s3://bucket/key`.
            - ByteStream: `meta` must contain `s3_url` or (`s3_bucket` + `s3_key`).
            - `UUID`/other types: ignored with a warning.
        :returns: A list of documents with metadata required for downloading.
        :raises ValueError: If the S3 URL cannot be parsed.
        """
        docs: List[Document] = []
        for src in sources:
            # S3 URL string
            if isinstance(src, str):
                if src.startswith("s3://"):
                    bkt, key = self._parse_s3_url(src)
                    file_name = os.path.basename(key) or "s3_file"
                    docs.append(Document(content="", meta={"s3_bucket": bkt, "s3_key": key, "file_name": file_name}))

                else:
                    logger.warning(f"Unsupported non-S3 source string: {src!r}. Skipping.")

            # ByteStream carrying S3 metadata
            elif isinstance(src, ByteStream):
                meta = dict(src.meta or {})
                bucket = meta.get("s3_bucket")
                key = meta.get("s3_key")
                url = meta.get("s3_url")
                if url and (not bucket or not key):
                    bucket, key = self._parse_s3_url(url)
                if bucket and key:
                    file_name = meta.get("file_name") or os.path.basename(key) or "s3_file"
                    m = {"s3_bucket": bucket, "s3_key": key, "file_name": file_name, **meta}
                    docs.append(Document(content="", meta=m))
                else:
                    logger.warning("ByteStream missing S3 metadata ('s3_bucket'/'s3_key' or 's3_url'). Skipping.")
            # UUID or other types: ignore (no generic S3 mapping)
            else:
                logger.warning(f"Unsupported source type {type(src)}; skipping.")
        return docs

    def download_file(self, document: Document) -> Optional[Document]:
        """

        Download a single S3 object into the local cache and enrich the `Document`.

        Expected S3 information in `document.meta`:
          - `s3_bucket` + `s3_key` **or**
          - `s3_url` (e.g., `s3://bucket/key`) **or**
          - `s3_key` (with default bucket provided elsewhere).

        :param document: `Document` describing the S3 object to download.
        :returns:
            The same `Document` with `meta` containing the path of the
            downloaded file and the mime type or `None` if skipped.
        :raises AmazonS3Error: If the download or head request fails.
        """
        try:
            bucket = document.meta.get("s3_bucket")
            key = document.meta.get("s3_key")
            url = document.meta.get("s3_url")

            if (not bucket or not key) and url:
                bucket, key = self._parse_s3_url(url)

            if not (bucket and key):
                logger.warning("Document missing S3 coordinates ('s3_bucket'/'s3_key' or 's3_url'). Skipping.")
                return None

            file_name = document.meta.get("file_name") or os.path.basename(key) or "s3_file"
            ext = Path(file_name).suffix
            cache_id = f"{bucket}/{key}"
            stem = str(abs(hash(cache_id)))  # abs() to avoid negative numbers
            file_path = self.file_root_path / f"{stem}{ext}"

            if file_path.is_file():
                file_path.touch()
            else:
                # download
                self._client.download_file(bucket, key, file_path)

            # enrich meta (best-effort head call)
            head = self._client.head_object(Bucket=bucket, Key=key)
            document.meta.setdefault("mime_type", head.get("ContentType"))

            document.meta["file_path"] = str(file_path)
            document.meta["cache_id"] = cache_id
            return document

        except ClientError as e:
            msg = f"Error downloading from S3: {e}"
            raise AmazonS3Error(msg) from e
        except Exception as e:
            msg = f"Unexpected error downloading from S3: {e}"
            raise AmazonS3Error(msg) from e

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
            verify_ssl=self.verify_ssl,
            download_dir=str(self.download_dir),
            max_workers=self.max_workers,
            max_cache_size=self.max_cache_size,
            file_extensions=self.file_extensions,
            sources_target_type=self.sources_target_type,
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

    @staticmethod
    def _parse_s3_url(url: str) -> tuple[str, str]:
        """
        Parse an S3 URL into (bucket, key).

        :param url: S3 URL, e.g., `s3://bucket/key`.
        :returns: Tuple of (`bucket`, `key`).
        :raises ValueError: If the URL does not use the `s3` scheme or is otherwise invalid.
        """
        parsed = urlparse(url)
        if parsed.scheme != "s3":
            msg = f"Invalid S3 URL scheme: {url!r}. Expected scheme 's3://'"
            raise ValueError(msg)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            msg = f"Invalid S3 URL: {url!r}. Expected s3://bucket/key"
            raise ValueError(msg)
        return bucket, key
