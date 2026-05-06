from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from haystack_integrations.common.s3.errors import S3ConfigurationError, S3StorageError
from haystack_integrations.common.s3.utils import S3Storage


def _client_error(code: int) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": str(code), "Message": "err"}},
        operation_name="download_file",
    )


class TestS3Storage:
    def test_init_client_creation_error(self):
        session = MagicMock()
        session.client.side_effect = Exception("boom")

        with pytest.raises(S3ConfigurationError, match="Failed to create S3 session client"):
            S3Storage(s3_bucket="bucket", session=session)

    def test_download_applies_prefix(self, tmp_path):
        session = MagicMock()
        storage = S3Storage(s3_bucket="bucket", session=session, s3_prefix="folder/")

        storage.download(key="file.txt", local_file_path=tmp_path / "file.txt")

        storage._client.download_file.assert_called_once_with("bucket", "folder/file.txt", str(tmp_path / "file.txt"))

    def test_download_no_prefix(self, tmp_path):
        session = MagicMock()
        storage = S3Storage(s3_bucket="bucket", session=session)

        storage.download(key="file.txt", local_file_path=tmp_path / "file.txt")

        storage._client.download_file.assert_called_once_with("bucket", "file.txt", str(tmp_path / "file.txt"))

    def test_download_missing_credentials(self, tmp_path):
        session = MagicMock()
        storage = S3Storage(s3_bucket="bucket", session=session)
        storage._client.download_file.side_effect = NoCredentialsError()

        with pytest.raises(S3ConfigurationError, match="Missing AWS credentials"):
            storage.download(key="file.txt", local_file_path=tmp_path / "file.txt")

    def test_download_forbidden(self, tmp_path):
        session = MagicMock()
        storage = S3Storage(s3_bucket="bucket", session=session)
        storage._client.download_file.side_effect = _client_error(403)

        with pytest.raises(S3ConfigurationError, match="Failed to access S3 bucket"):
            storage.download(key="file.txt", local_file_path=tmp_path / "file.txt")

    def test_download_not_found(self, tmp_path):
        session = MagicMock()
        storage = S3Storage(s3_bucket="bucket", session=session)
        storage._client.download_file.side_effect = _client_error(404)

        with pytest.raises(S3StorageError, match="does not exist"):
            storage.download(key="file.txt", local_file_path=tmp_path / "file.txt")

    def test_download_other_client_error(self, tmp_path):
        session = MagicMock()
        storage = S3Storage(s3_bucket="bucket", session=session)
        storage._client.download_file.side_effect = _client_error(500)

        with pytest.raises(S3StorageError, match="Failed to download file"):
            storage.download(key="file.txt", local_file_path=tmp_path / "file.txt")

    def test_from_env_success(self, monkeypatch):
        monkeypatch.delenv("S3_DOWNLOADER_BUCKET", raising=False)
        monkeypatch.setenv("MY_BUCKET_VAR", "my-bucket")
        monkeypatch.setenv("S3_DOWNLOADER_PREFIX", "prefix/")
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:9000")
        session = MagicMock()

        storage = S3Storage.from_env(session=session, config=MagicMock(), s3_bucket_name_env="MY_BUCKET_VAR")

        assert storage.s3_bucket == "my-bucket"
        assert storage.s3_prefix == "prefix/"
        assert storage.endpoint_url == "http://localhost:9000"

    def test_from_env_missing_bucket(self, monkeypatch):
        monkeypatch.delenv("S3_DOWNLOADER_BUCKET", raising=False)
        session = MagicMock()

        with pytest.raises(ValueError, match="Missing environment variable S3_DOWNLOADER_BUCKET"):
            S3Storage.from_env(session=session, config=MagicMock())
