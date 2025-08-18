# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse
from uuid import UUID

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream, Document
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from haystack_integrations.common.amazon_bedrock.errors import AmazonBedrockConfigurationError
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

logger = logging.getLogger(__name__)


@component
class S3Downloader:
    """
    S3 downloader component for Haystack.

    Downloads files from S3 to local filesystem with caching and concurrent downloads.
    Supports filtering by file extensions and returns documents with file paths.
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
        endpoint_url: Optional[str] = None,
        verify_ssl: bool = True,
        boto3_config: Optional[Dict[str, Any]] = None,
        download_dir: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        sources_target_type: Literal["str", "pathlib.Path", "haystack.dataclasses.ByteStream"] = "str",
        default_bucket: Optional[str] = None,
        max_workers: int = 32,
    ):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region_name = aws_region_name
        self.aws_session_token = aws_session_token
        self.aws_profile_name = aws_profile_name
        self.endpoint_url = endpoint_url
        self.verify_ssl = verify_ssl
        self.boto3_config = boto3_config
        self.file_extensions = [e.lower() for e in file_extensions] if file_extensions else None
        self.sources_target_type = sources_target_type
        self.default_bucket = default_bucket
        self.max_workers = max_workers
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
            if self.endpoint_url:
                self._client = session.client("s3", config=config, endpoint_url=self.endpoint_url)
            else:
                self._client = session.client("s3", config=config)
        except Exception as exception:
            msg = (
                "Could not connect to Amazon S3. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

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
        """
        Download files linked to S3:
        - If `sources` provided: accept S3 URLs (str) or ByteStreams carrying S3 meta.
        - If `documents` provided: read S3 info from each doc's meta.
          Expected meta keys: 's3_bucket' + 's3_key' OR 's3_url' OR ('s3_key' with default bucket set).
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

        print (documents)

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
        match sources_target_type:
            case "str":
                return file_paths
            case "pathlib.Path":
                return [Path(p) for p in file_paths]
            case "haystack.dataclasses.ByteStream":
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
        Accepts:
        - str: "s3://bucket/key"
        - ByteStream: meta may contain 's3_url' or ('s3_bucket' + 's3_key')
        - UUID/other str IDs: ignored unless `default_bucket` is set and the ID looks like a key
        """
        docs: List[Document] = []
        for src in sources:
            # S3 URL string
            if isinstance(src, str):
                if src.startswith("s3://"):
                    bkt, key = self._parse_s3_url(src)
                    file_name = os.path.basename(key) or "s3_file"
                    docs.append(Document(content="", meta={"s3_bucket": bkt, "s3_key": key, "file_name": file_name}))
                elif self.default_bucket:
                    # Treat as key-only if a default bucket is set
                    file_name = os.path.basename(src) or "s3_file"
                    docs.append(
                        Document(content="", meta={"s3_bucket": self.default_bucket, "s3_key": src, "file_name": file_name})
                    )
                else:
                    logger.warning(f"Unsupported non-S3 source string: {src!r}. Skipping.")
            # ByteStream carrying S3 metadata
            elif isinstance(src, ByteStream):
                meta = dict(src.meta or {})
                bucket = meta.get("s3_bucket")
                key = meta.get("s3_key")
                url = meta.get("s3_url")
                if url and (not bucket or not key):
                    try:
                        bucket, key = self._parse_s3_url(url)
                    except Exception:  # noqa: BLE001
                        pass
                if not (bucket and key) and self.default_bucket and key:
                    bucket = self.default_bucket
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
        Download a single document from S3 into the local cache.
        Expects doc.meta to provide:
          - 's3_bucket' + 's3_key'  OR
          - 's3_url'  OR
          - 's3_key' with self.default_bucket set
        """
        try:
            bucket = document.meta.get("s3_bucket")
            key = document.meta.get("s3_key")
            url = document.meta.get("s3_url")

            if (not bucket or not key) and url:
                bucket, key = self._parse_s3_url(url)

            if (not bucket or not key) and self.default_bucket and key:
                bucket = self.default_bucket

            if not (bucket and key):
                logger.warning("Document missing S3 coordinates ('s3_bucket'/'s3_key' or 's3_url'). Skipping.")
                return None

            file_name = document.meta.get("file_name") or os.path.basename(key) or "s3_file"
            ext = Path(file_name).suffix
            cache_id = f"{bucket}/{key}"
            stem = hashlib.sha1(cache_id.encode("utf-8")).hexdigest()
            file_path = self.file_root_path / f"{stem}{ext}"

            if file_path.is_file():
                file_path.touch()
            else:
                # download
                try:
                    response = self._client.get_object(Bucket=bucket, Key=key)
                    with open(file_path, 'wb') as f:
                        f.write(response['Body'].read())
                except ClientError as e:
                    code = e.response.get("Error", {}).get("Code", "")
                    if code in {"NoSuchKey", "404"}:
                        logger.error(f"File not found in S3: s3://{document.meta.get('s3_bucket')}/{document.meta.get('s3_key')}")
                        return None
                    elif code == "NoSuchBucket":
                        logger.error(f"Bucket not found: {document.meta.get('s3_bucket')}")
                        return None
                    else:
                        logger.error(f"S3 error: {e}", exc_info=True)
                        raise
                except NoCredentialsError as e:
                    logger.error(f"AWS credentials not found: {e}", exc_info=True)
                    raise
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Unexpected error downloading from S3: {e}", exc_info=True)
                    raise


            # enrich meta (best-effort head call)
            try:
                head = self._client.head_object(Bucket=bucket, Key=key)
                document.meta.setdefault("mime_type", head.get("ContentType"))
                document.meta.update(
                    {
                        "s3_etag": head.get("ETag"),
                        "s3_last_modified": str(head.get("LastModified")),
                        "s3_content_length": head.get("ContentLength"),
                    }
                )
            except Exception:  # noqa: BLE001
                pass

            document.meta["file_path"] = str(file_path)
            document.meta["cache_id"] = cache_id
            return document

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "404"}:
                logger.error(f"File not found in S3: s3://{document.meta.get('s3_bucket')}/{document.meta.get('s3_key')}")
                return None
            elif code == "NoSuchBucket":
                logger.error(f"Bucket not found: {document.meta.get('s3_bucket')}")
                return None
            else:
                logger.error(f"S3 error: {e}", exc_info=True)
                raise
        except NoCredentialsError as e:
            logger.error(f"AWS credentials not found: {e}", exc_info=True)
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error downloading from S3: {e}", exc_info=True)
            raise

    def cleanup_cache(self, documents: List[Document]) -> None:
        """
        Remove least-recently-accessed cache files when cache exceeds max_cache_size.
        """
        requested_ids = {
            hashlib.sha1(str(doc.meta.get("cache_id", "")).encode("utf-8")).hexdigest()
            for doc in documents
            if doc.meta.get("cache_id")
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
                except Exception:  # noqa: BLE001
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
            endpoint_url=self.endpoint_url,
            verify_ssl=self.verify_ssl,
            download_dir=str(self.download_dir),
            max_workers=self.max_workers,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "S3Downloader":
        """
        Deserializes the component from a dictionary.
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "aws_region_name", "aws_profile_name"],
        )
        return default_from_dict(cls, data)


    @staticmethod
    def _parse_s3_url(url: str) -> tuple[str, str]:
        parsed = urlparse(url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 URL: {url!r}. Expected s3://bucket/key")
        return bucket, key
