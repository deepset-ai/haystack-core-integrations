# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.components.downloaders.supabase import SupabaseBucketDownloader

_MODULE = "haystack_integrations.components.downloaders.supabase.supabase_bucket_downloader"
PATCH_PATH = f"{_MODULE}.create_client"
COMPONENT_TYPE = f"{_MODULE}.SupabaseBucketDownloader"


class TestSupabaseBucketDownloader:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key")
        downloader = SupabaseBucketDownloader(
            supabase_url="https://project.supabase.co",
            bucket_name="my-bucket",
        )
        assert downloader.supabase_url == "https://project.supabase.co"
        assert downloader.bucket_name == "my-bucket"
        assert downloader.file_extensions is None
        assert downloader.supabase_key.resolve_value() == "test-key"

    def test_init_with_file_extensions(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key")
        downloader = SupabaseBucketDownloader(
            supabase_url="https://project.supabase.co",
            bucket_name="my-bucket",
            file_extensions=[".PDF", ".TXT"],
        )
        assert downloader.file_extensions == [".pdf", ".txt"]

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key")
        downloader = SupabaseBucketDownloader(
            supabase_url="https://project.supabase.co",
            bucket_name="docs",
            file_extensions=[".pdf"],
        )
        data = component_to_dict(downloader, "SupabaseBucketDownloader")
        assert data["type"] == COMPONENT_TYPE
        assert data["init_parameters"]["supabase_url"] == "https://project.supabase.co"
        assert data["init_parameters"]["bucket_name"] == "docs"
        assert data["init_parameters"]["file_extensions"] == [".pdf"]

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key")
        data = {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "supabase_url": "https://project.supabase.co",
                "bucket_name": "docs",
                "file_extensions": [".pdf", ".txt"],
                "supabase_key": {"env_vars": ["SUPABASE_SERVICE_KEY"], "strict": True, "type": "env_var"},
            },
        }
        downloader = component_from_dict(SupabaseBucketDownloader, data, "SupabaseBucketDownloader")
        assert downloader.supabase_url == "https://project.supabase.co"
        assert downloader.bucket_name == "docs"
        assert downloader.file_extensions == [".pdf", ".txt"]

    def test_run_returns_bytestreams(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key")
        downloader = SupabaseBucketDownloader(
            supabase_url="https://project.supabase.co",
            supabase_key=Secret.from_token("test-key"),
            bucket_name="my-bucket",
        )
        mock_bucket = MagicMock()
        mock_bucket.download.return_value = b"file content"

        with patch(PATCH_PATH) as mock_client:
            mock_client.return_value.storage.from_.return_value = mock_bucket
            result = downloader.run(sources=["folder/file.pdf"])

        assert len(result["streams"]) == 1
        assert isinstance(result["streams"][0], ByteStream)
        assert result["streams"][0].data == b"file content"
        assert result["streams"][0].meta["file_path"] == "folder/file.pdf"
        assert result["streams"][0].meta["bucket_name"] == "my-bucket"
        assert result["streams"][0].mime_type == "application/pdf"

    def test_run_filters_by_extension(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key")
        downloader = SupabaseBucketDownloader(
            supabase_url="https://project.supabase.co",
            supabase_key=Secret.from_token("test-key"),
            bucket_name="my-bucket",
            file_extensions=[".pdf"],
        )
        mock_bucket = MagicMock()
        mock_bucket.download.return_value = b"pdf content"

        with patch(PATCH_PATH) as mock_client:
            mock_client.return_value.storage.from_.return_value = mock_bucket
            result = downloader.run(sources=["doc.pdf", "notes.txt", "image.png"])

        assert len(result["streams"]) == 1
        assert result["streams"][0].meta["file_path"] == "doc.pdf"
        assert mock_bucket.download.call_count == 1

    def test_run_empty_sources(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key")
        downloader = SupabaseBucketDownloader(
            supabase_url="https://project.supabase.co",
            supabase_key=Secret.from_token("test-key"),
            bucket_name="my-bucket",
        )
        with patch(PATCH_PATH):
            result = downloader.run(sources=[])

        assert result["streams"] == []

    def test_run_skips_failed_downloads(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key")
        downloader = SupabaseBucketDownloader(
            supabase_url="https://project.supabase.co",
            supabase_key=Secret.from_token("test-key"),
            bucket_name="my-bucket",
        )
        mock_bucket = MagicMock()
        mock_bucket.download.side_effect = [Exception("Not found"), b"good content"]

        with patch(PATCH_PATH) as mock_client:
            mock_client.return_value.storage.from_.return_value = mock_bucket
            result = downloader.run(sources=["missing.pdf", "exists.txt"])

        assert len(result["streams"]) == 1
        assert result["streams"][0].meta["file_path"] == "exists.txt"

    def test_run_sets_mime_type(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "test-key")
        downloader = SupabaseBucketDownloader(
            supabase_url="https://project.supabase.co",
            supabase_key=Secret.from_token("test-key"),
            bucket_name="my-bucket",
        )
        mock_bucket = MagicMock()
        mock_bucket.download.return_value = b"data"

        with patch(PATCH_PATH) as mock_client:
            mock_client.return_value.storage.from_.return_value = mock_bucket
            result = downloader.run(sources=["doc.pdf", "data.csv", "unknown.xyz"])

        assert result["streams"][0].mime_type == "application/pdf"
        assert result["streams"][1].mime_type is not None
        assert result["streams"][2].mime_type == "application/octet-stream"

    @pytest.mark.skipif(
        not os.environ.get("SUPABASE_SERVICE_KEY") or not os.environ.get("SUPABASE_URL"),
        reason="Export SUPABASE_URL and SUPABASE_SERVICE_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        downloader = SupabaseBucketDownloader(
            supabase_url=os.environ["SUPABASE_URL"],
            supabase_key=Secret.from_env_var("SUPABASE_SERVICE_KEY"),
            bucket_name=os.environ.get("SUPABASE_BUCKET_NAME", "test-bucket"),
        )
        result = downloader.run(sources=["test-file.txt"])
        assert len(result["streams"]) > 0
        assert isinstance(result["streams"][0], ByteStream)
