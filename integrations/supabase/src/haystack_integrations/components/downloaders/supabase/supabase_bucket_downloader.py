# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
from pathlib import Path
from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from supabase import Client, create_client

logger = logging.getLogger(__name__)


@component
class SupabaseBucketDownloader:
    """
    Downloads files from a Supabase Storage bucket and returns them as ByteStream objects.

    Files are downloaded in-memory and returned as `ByteStream` objects ready for further
    processing in indexing pipelines (e.g. passing to a `DocumentConverter`).

    Example usage:

    ```python
    from haystack_integrations.components.downloaders.supabase import SupabaseBucketDownloader
    from haystack.utils import Secret

    downloader = SupabaseBucketDownloader(
        supabase_url="https://<project-ref>.supabase.co",
        supabase_key=Secret.from_env_var("SUPABASE_SERVICE_KEY"),
        bucket_name="my-documents",
    )
    result = downloader.run(sources=["reports/report.pdf", "data/notes.txt"])
    streams = result["streams"]
    ```
    """

    def __init__(
        self,
        *,
        supabase_url: str,
        supabase_key: Secret = Secret.from_env_var("SUPABASE_SERVICE_KEY"),
        bucket_name: str,
        file_extensions: list[str] | None = None,
    ) -> None:
        """
        Creates a new SupabaseBucketDownloader instance.

        :param supabase_url: The URL of your Supabase project, e.g. `https://<project-ref>.supabase.co`.
        :param supabase_key: The Supabase API key used to authenticate requests. Defaults to the
            `SUPABASE_SERVICE_KEY` environment variable. Use the service role key for private buckets.
        :param bucket_name: The name of the Supabase Storage bucket to download files from.
        :param file_extensions: Optional list of file extensions to filter downloads (e.g. `[".pdf", ".txt"]`).
            If `None`, all files are downloaded. Extensions are matched case-insensitively.
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.bucket_name = bucket_name
        self.file_extensions = [e.lower() for e in file_extensions] if file_extensions else None
        self._client: Client | None = None

    def warm_up(self) -> None:
        """Initialize the Supabase client. Called once before the first run."""
        if self._client is None:
            key = self.supabase_key.resolve_value()
            if not key:
                msg = "Supabase API key could not be resolved. Set the SUPABASE_SERVICE_KEY environment variable."
                raise ValueError(msg)
            self._client = create_client(self.supabase_url, key)

    @component.output_types(streams=list[ByteStream])
    def run(self, sources: list[str]) -> dict[str, list[ByteStream]]:
        """
        Downloads files from the Supabase Storage bucket.

        :param sources: List of file paths within the bucket to download,
            e.g. `["folder/file.pdf", "notes.txt"]`.
        :returns: A dictionary with:
            - `streams`: list of `ByteStream` objects, one per successfully downloaded file.
                Each `ByteStream` has `meta["file_path"]` and `meta["bucket_name"]` set.
        """
        if self._client is None:
            self.warm_up()
        assert self._client is not None
        streams = []

        for path in sources:
            if self.file_extensions is not None:
                ext = Path(path).suffix.lower()
                if ext not in self.file_extensions:
                    logger.debug("Skipping {path} — extension not in filter list.", path=path)
                    continue

            try:
                data = self._client.storage.from_(self.bucket_name).download(path)
            except Exception as e:
                logger.warning(
                    "Failed to download {path} from bucket {bucket}: {error}",
                    path=path,
                    bucket=self.bucket_name,
                    error=e,
                )
                continue

            mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            streams.append(
                ByteStream(
                    data=data,
                    meta={"file_path": path, "bucket_name": self.bucket_name},
                    mime_type=mime_type,
                )
            )

        return {"streams": streams}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            supabase_url=self.supabase_url,
            supabase_key=self.supabase_key.to_dict(),
            bucket_name=self.bucket_name,
            file_extensions=self.file_extensions,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SupabaseBucketDownloader":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], ["supabase_key"])
        return default_from_dict(cls, data)
