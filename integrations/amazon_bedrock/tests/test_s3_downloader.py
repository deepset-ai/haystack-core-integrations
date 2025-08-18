import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from haystack.dataclasses import Document, ByteStream
from haystack.utils import Secret

from haystack_integrations.common.amazon_bedrock.errors import AmazonBedrockConfigurationError
from haystack_integrations.components.downloader.s3_downloader import S3Downloader

TYPE = "haystack_integrations.components.downloader.s3_downloader.S3Downloader"


@pytest.fixture
def set_env_variables(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "some_fake_id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "some_fake_key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "some_fake_token")
    monkeypatch.setenv("AWS_PROFILE", "some_fake_profile")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "fake_region")


@pytest.fixture
def mock_boto3_session(monkeypatch):
    import haystack_integrations.common.amazon_bedrock.utils as utils

    class _FakeSession:
        def client(self, *_a, **_k):
            return MagicMock()

    mock = MagicMock(return_value=_FakeSession())
    monkeypatch.setattr(utils, "get_aws_session", mock)
    return mock


class TestS3Downloader:
    def test_init(self, mock_boto3_session, set_env_variables, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        assert d.download_dir == Path(tmp_path)
        mock_boto3_session.assert_called_once()
        mock_boto3_session.assert_called_with(
            aws_access_key_id="some_fake_id",
            aws_secret_access_key="some_fake_key",
            aws_session_token="some_fake_token",
            aws_region_name="fake_region",
            aws_profile_name="some_fake_profile",
        )

    def test_init_custom_parameters(self, mock_boto3_session, tmp_path):
        d = S3Downloader(
            aws_access_key_id=Secret.from_token("k"),
            aws_secret_access_key=Secret.from_token("s"),
            aws_session_token=Secret.from_token("t"),
            aws_region_name=Secret.from_token("eu-central-1"),
            aws_profile_name=Secret.from_token("prof"),
            endpoint_url="https://example",
            verify_ssl=False,
            download_dir=str(tmp_path),
            file_extensions=[".pdf", ".txt"],
            sources_target_type="haystack.dataclasses.ByteStream",
        )
        assert d.endpoint_url == "https://example"
        assert d.verify_ssl is False
        assert d.file_extensions == [".pdf", ".txt"]
        assert d.sources_target_type == "haystack.dataclasses.ByteStream"

    def test_connection_error(self, monkeypatch):
        import haystack_integrations.common.amazon_bedrock.utils as utils

        monkeypatch.setattr(utils, "get_aws_session", MagicMock(side_effect=Exception("boom")))
        with pytest.raises(AmazonBedrockConfigurationError):
            S3Downloader()

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
                "endpoint_url": None,
                "verify_ssl": True,
                "download_dir": str(tmp_path),
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
                "aws_region_name": "eu-central-1",
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "endpoint_url": "https://example",
                "verify_ssl": False,
                "download_dir": str(tmp_path),
            },
        }
        d = S3Downloader.from_dict(data)
        assert d.aws_region_name.resolve_value() == "eu-central-1"
        assert d.endpoint_url == "https://example"
        assert d.verify_ssl is False
        assert Path(d.download_dir) == tmp_path

    def test_parse_s3_url_valid(self, mock_boto3_session):
        d = S3Downloader()
        b, k = d._parse_s3_url("s3://bucket/path/to/file.txt")
        assert b == "bucket" and k == "path/to/file.txt"

    def test_parse_s3_url_invalid(self, mock_boto3_session):
        d = S3Downloader()
        with pytest.raises(ValueError):
            d._parse_s3_url("http://example.com/file.txt")
        with pytest.raises(ValueError):
            d._parse_s3_url("s3://")

    def test_run_no_inputs(self, mock_boto3_session):
        d = S3Downloader()
        out = d.run()
        assert out["documents"] == [] and out["sources"] == []

    def test_run_invalid_s3_url(self, mock_boto3_session):
        d = S3Downloader()
        with pytest.raises(ValueError):
            d.run(sources=["s3://"])

    def test_run_success_single(self, mock_boto3_session, tmp_path):
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

    def test_run_multiple(self, mock_boto3_session, tmp_path):
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

    def test_filter_extensions(self, mock_boto3_session, tmp_path):
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

    def test_run_file_not_found(self, mock_boto3_session, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "nope"}}, "GetObject"
        )
        d._client = mock_s3
        out = d.run(sources=["s3://b/missing.txt"])
        assert out["documents"] == [] and out["sources"] == []

    def test_run_bucket_not_found(self, mock_boto3_session, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "nope"}}, "GetObject"
        )
        d._client = mock_s3
        out = d.run(sources=["s3://missing/file.txt"])
        assert out["documents"] == [] and out["sources"] == []

    def test_run_credentials_error(self, mock_boto3_session, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = NoCredentialsError()
        d._client = mock_s3
        with pytest.raises(NoCredentialsError):
            d.run(sources=["s3://b/a.txt"])

    def test_run_access_denied(self, mock_boto3_session, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject"
        )
        d._client = mock_s3
        with pytest.raises(ClientError):
            d.run(sources=["s3://b/a.txt"])

    def test_run_unexpected_error(self, mock_boto3_session, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.download_file.side_effect = Exception("boom")
        d._client = mock_s3
        with pytest.raises(Exception, match="boom"):
            d.run(sources=["s3://b/a.txt"])

    def test_sources_as_bytestream(self, mock_boto3_session, tmp_path):
        d = S3Downloader(download_dir=str(tmp_path), sources_target_type="haystack.dataclasses.ByteStream")
        mock_s3 = MagicMock()

        def _dl(_b, _k, dst):
            Path(dst).write_bytes(b"x")

        mock_s3.download_file.side_effect = _dl
        mock_s3.head_object.return_value = {"ContentType": "application/octet-stream"}
        d._client = mock_s3
        out = d.run(sources=["s3://b/a.bin"])
        assert isinstance(out["sources"][0], ByteStream)
