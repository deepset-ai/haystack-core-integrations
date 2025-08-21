from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.common.amazon_bedrock.errors import AmazonS3Error
from haystack_integrations.components.downloader.s3.s3_downloader import S3Downloader

TYPE = "haystack_integrations.components.downloader.s3.s3_downloader.S3Downloader"


class TestS3Downloader:
    def test_init(self, mock_boto3_session, set_env_variables, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        assert d.download_dir == Path(tmp_path)
        mock_boto3_session.assert_called_once()
        # assert mocked boto3 client was called with the correct parameters
        mock_boto3_session.assert_called_with(
            aws_access_key_id="some_fake_id",
            aws_secret_access_key="some_fake_key",
            aws_session_token="some_fake_token",
            profile_name="some_fake_profile",
            region_name="fake_region",
        )

    def test_init_custom_parameters(self, mock_boto3_session, tmp_path):
        d = S3Downloader(
            aws_access_key_id=Secret.from_token("k"),
            aws_secret_access_key=Secret.from_token("s"),
            aws_session_token=Secret.from_token("t"),
            aws_region_name=Secret.from_token("eu-central-1"),
            aws_profile_name=Secret.from_token("prof"),
            verify_ssl=False,
            download_dir=str(tmp_path),
            file_extensions=[".pdf", ".txt"],
            sources_target_type="haystack.dataclasses.ByteStream",
        )
        assert d.verify_ssl is False
        assert d.file_extensions == [".pdf", ".txt"]
        assert d.sources_target_type == "haystack.dataclasses.ByteStream"


    @pytest.mark.parametrize("boto3_config", [None, {"read_timeout": 10}])
    def test_to_dict(self, mock_boto3_session: Any, tmp_path, boto3_config: Optional[Dict[str, Any]]):
        d = S3Downloader(download_dir=str(tmp_path), boto3_config=boto3_config)
        expected = {
            "type": TYPE,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "verify_ssl": True,
                "download_dir": str(tmp_path),
                "file_extensions": None,
                "max_cache_size": 100,
                "max_workers": 32,
                "sources_target_type": "str",
            },
        }
        assert d.to_dict() == expected

    @pytest.mark.parametrize("boto3_config", [None, {"read_timeout": 10}])
    def test_from_dict(self, mock_boto3_session: Any, tmp_path, boto3_config: Optional[Dict[str, Any]]):
        data = {
            "type": TYPE,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "verify_ssl": False,
                "download_dir": str(tmp_path),

            },
        }
        d = S3Downloader.from_dict(data)
        assert d.verify_ssl is False
        assert Path(d.download_dir) == tmp_path

    def test_parse_s3_url_valid(self):
        d = S3Downloader()
        b, k = d._parse_s3_url("s3://bucket/path/to/file.txt")
        assert b == "bucket" and k == "path/to/file.txt"

    def test_parse_s3_url_invalid(self):
        d = S3Downloader()
        with pytest.raises(ValueError):
            d._parse_s3_url("http://example.com/file.txt")


    def test_run_no_inputs(self):
        d = S3Downloader()
        out = d.run()
        assert out["documents"] == [] and out["sources"] == []


    def test_run_invalid_s3_url(self):
        d = S3Downloader()
        with pytest.raises(ValueError):
            d.run(sources=["s3://"])  # invalid: missing bucket/key


    def test_run_success_single(self, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        mock_s3 = MagicMock()

        def _dl(bucket, key, dst):
            Path(dst).write_bytes(b"content")

        mock_s3.download_file.side_effect = _dl
        mock_s3.head_object.return_value = {
            "ContentType": "text/plain",
            "ContentLength": 7,
            "LastModified": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            "ETag": '"etag"',
        }
        d._client = mock_s3

        out = d.run(sources=["s3://my-bucket/a.txt"])
        assert len(out["documents"]) == 1
        doc = out["documents"][0]
        assert Path(doc.meta["file_path"]).is_file()
        assert doc.meta["mime_type"] == "text/plain"
        assert out["sources"][0] == doc.meta["file_path"]


    def test_run_multiple(self, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        mock_s3 = MagicMock()

        def _dl(_b, _k, dst):
            Path(dst).write_bytes(b"x")

        mock_s3.download_file.side_effect = _dl
        mock_s3.head_object.return_value = {"ContentType": "application/octet-stream"}
        d._client = mock_s3

        out = d.run(sources=["s3://b/x.pdf", "s3://b/y.pdf"])
        assert len(out["documents"]) == 2
        assert mock_s3.download_file.call_count == 2


    def test_filter_extensions(self, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path), file_extensions=[".pdf"])
        mock_s3 = MagicMock()

        def _dl(_b, _k, dst):
            Path(dst).write_bytes(b"x")

        mock_s3.download_file.side_effect = _dl
        mock_s3.head_object.return_value = {"ContentType": "application/pdf"}
        d._client = mock_s3

        out = d.run(sources=["s3://b/a.pdf", "s3://b/a.txt"])
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["file_path"].endswith(".pdf")


    def test_run_file_not_found_raises_amazons3error(self, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "nope"}}, "GetObject"
        )
        d._client = mock_s3
        with pytest.raises(AmazonS3Error):
            d.run(sources=["s3://b/missing.txt"])


    def test_run_bucket_not_found_raises_amazons3error(self, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "nope"}}, "GetObject"
        )
        d._client = mock_s3
        with pytest.raises(AmazonS3Error):
            d.run(sources=["s3://missing/file.txt"])


    def test_run_credentials_error_is_wrapped(self, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        mock_s3 = MagicMock()
        # Generic Exception path in download_file → wrapped as AmazonS3Error
        mock_s3.download_file.side_effect = NoCredentialsError()
        d._client = mock_s3
        with pytest.raises(AmazonS3Error):
            d.run(sources=["s3://b/a.txt"])


    def test_run_access_denied_is_wrapped(self, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject"
        )
        d._client = mock_s3
        with pytest.raises(AmazonS3Error):
            d.run(sources=["s3://b/a.txt"])


    def test_run_unexpected_error_is_wrapped(self, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = Exception("boom")
        d._client = mock_s3
        with pytest.raises(AmazonS3Error, match="boom"):
            d.run(sources=["s3://b/a.txt"])


    def test_sources_as_bytestream(self, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path), sources_target_type="haystack.dataclasses.ByteStream")
        mock_s3 = MagicMock()

        def _dl(_b, _k, dst):
            Path(dst).write_bytes(b"x")

        mock_s3.download_file.side_effect = _dl
        mock_s3.head_object.return_value = {"ContentType": "application/octet-stream"}
        d._client = mock_s3

        out = d.run(sources=["s3://b/a.bin"])
        assert isinstance(out["sources"][0], ByteStream)
