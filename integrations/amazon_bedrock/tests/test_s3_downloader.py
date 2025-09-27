import os
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from haystack.dataclasses import Document
from haystack.utils import Secret

from haystack_integrations.common.s3.utils import S3Storage
from haystack_integrations.components.downloaders.s3.s3_downloader import S3Downloader

TYPE = "haystack_integrations.components.downloaders.s3.s3_downloader.S3Downloader"


@pytest.fixture
def mock_s3_storage():
    mock = MagicMock(spec=S3Storage)

    def fake_download(key, local_file_path: Path):
        Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_file_path).write_bytes(b"content")

    mock.download.side_effect = fake_download
    with patch(
        "haystack_integrations.components.downloaders.s3.s3_downloader.S3Storage.from_env",
        return_value=mock,
    ):
        yield mock


def s3_key_generation_function(document: Document) -> str:
    return document.meta["file_name"] + "_suffix"


class TestS3Downloader:
    def test_init(self, mock_boto3_session, set_env_variables, tmp_path):
        S3Downloader(file_root_path=str(tmp_path))
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
            file_root_path=str(tmp_path),
            file_extensions=[".pdf", ".txt"],
            max_cache_size=100,
            max_workers=32,
            file_name_meta_key="file_id",
        )
        assert d.file_extensions == [".pdf", ".txt"]

    @pytest.mark.parametrize("boto3_config", [None, {"read_timeout": 10}])
    def test_to_dict(self, mock_boto3_session: Any, tmp_path, boto3_config: Optional[Dict[str, Any]]):
        d = S3Downloader(file_root_path=str(tmp_path), boto3_config=boto3_config)
        expected = {
            "type": TYPE,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "file_root_path": str(tmp_path),
                "file_extensions": None,
                "max_cache_size": 100,
                "max_workers": 32,
                "file_name_meta_key": "file_name",
                "s3_key_generation_function": None,
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
                "file_root_path": str(tmp_path),
                "s3_key_generation_function": None,
            },
        }
        d = S3Downloader.from_dict(data)
        assert Path(d.file_root_path) == tmp_path

    def test_to_dict_with_parameters(self, tmp_path):
        d = S3Downloader(
            file_root_path=str(tmp_path),
            file_extensions=[".txt"],
            max_cache_size=400,
            max_workers=40,
            file_name_meta_key="new_file_key",
            s3_key_generation_function=s3_key_generation_function,
        )
        expected = {
            "type": TYPE,
            "init_parameters": {
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "file_root_path": str(tmp_path),
                "file_extensions": [".txt"],
                "max_cache_size": 400,
                "max_workers": 40,
                "file_name_meta_key": "new_file_key",
                "s3_key_generation_function": "tests.test_s3_downloader.s3_key_generation_function",
            },
        }
        assert d.to_dict() == expected

    def test_run(self, tmp_path, mock_s3_storage, mock_boto3_session):
        d = S3Downloader(file_root_path=str(tmp_path))
        S3Downloader.warm_up(d)
        d._storage = mock_s3_storage

        docs = [
            Document(meta={"file_id": str(uuid4()), "file_name": "a.txt"}),
            Document(meta={"file_id": str(uuid4()), "file_name": "b.txt"}),
        ]
        out = d.run(documents=docs)
        assert len(out["documents"]) == 2

    def test_run_with_extensions(self, tmp_path, mock_s3_storage, mock_boto3_session):
        d = S3Downloader(file_root_path=str(tmp_path), file_extensions=[".txt"])
        S3Downloader.warm_up(d)
        d._storage = mock_s3_storage

        docs = [
            Document(meta={"file_id": str(uuid4()), "file_name": "a.txt"}),
            Document(meta={"file_id": str(uuid4()), "file_name": "b.pdf"}),
        ]

        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["file_name"] == "a.txt"

    def test_run_with_input_file_meta_key(self, tmp_path, mock_s3_storage, mock_boto3_session):
        d = S3Downloader(file_root_path=str(tmp_path), file_name_meta_key="custom_file_key")
        S3Downloader.warm_up(d)
        d._storage = mock_s3_storage

        docs = [
            Document(meta={"file_id": str(uuid4()), "custom_file_key": "a.txt"}),
        ]

        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["custom_file_key"] == "a.txt"

    def test_run_with_s3_key_generation_function(self, tmp_path, mock_s3_storage, mock_boto3_session):
        d = S3Downloader(file_root_path=str(tmp_path), s3_key_generation_function=s3_key_generation_function)
        S3Downloader.warm_up(d)
        d._storage = mock_s3_storage

        docs = [
            Document(meta={"file_id": str(uuid4()), "file_name": "a.txt"}),
        ]
        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["file_name"] == "a.txt"

        mock_s3_storage.download.assert_called_once()
        assert mock_s3_storage.download.call_args.kwargs["key"] == "a.txt_suffix"

    def test_run_with_s3_key_generation_function_and_file_extensions(
        self, tmp_path, mock_s3_storage, mock_boto3_session
    ):
        d = S3Downloader(
            file_root_path=str(tmp_path),
            s3_key_generation_function=s3_key_generation_function,
            file_extensions=[".txt"],
        )
        S3Downloader.warm_up(d)
        d._storage = mock_s3_storage

        docs = [
            Document(meta={"file_id": str(uuid4()), "file_name": "a.txt"}),
            Document(meta={"file_id": str(uuid4()), "file_name": "b.pdf"}),
        ]
        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["file_name"] == "a.txt"
        mock_s3_storage.download.assert_called_once()
        assert mock_s3_storage.download.call_args.kwargs["key"] == "a.txt_suffix"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("S3_DOWNLOADER_BUCKET", None),
        reason="Export an env var called `S3_DOWNLOADER_BUCKET` containing the S3 bucket to run this test.",
    )
    def test_live_run(self, tmp_path, monkeypatch):
        d = S3Downloader(file_root_path=str(tmp_path))
        monkeypatch.setenv("S3_DOWNLOADER_PREFIX", "")
        S3Downloader.warm_up(d)

        docs = [
            Document(meta={"file_id": str(uuid4()), "file_name": "text-sample.txt"}),
            Document(meta={"file_id": str(uuid4()), "file_name": "document-sample.pdf"}),
        ]

        out = d.run(documents=docs)
        assert len(out["documents"]) == 2
        assert out["documents"][0].meta["file_name"] == "text-sample.txt"
        assert out["documents"][1].meta["file_name"] == "document-sample.pdf"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("S3_DOWNLOADER_BUCKET", None),
        reason="Export an env var called `S3_DOWNLOADER_BUCKET` containing the S3 bucket to run this test.",
    )
    def test_live_run_with_no_documents(self, tmp_path):
        d = S3Downloader(file_root_path=str(tmp_path))
        S3Downloader.warm_up(d)
        docs = []
        out = d.run(documents=docs)
        assert len(out["documents"]) == 0

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("S3_DOWNLOADER_BUCKET", None),
        reason="Export an env var called `S3_DOWNLOADER_BUCKET` containing the S3 bucket to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("S3_DOWNLOADER_BUCKET", None),
        reason="Export an env var called `S3_DOWNLOADER_BUCKET` containing the S3 bucket to run this test.",
    )
    def test_live_run_with_custom_meta_key(self, tmp_path, monkeypatch):
        d = S3Downloader(file_root_path=str(tmp_path), file_name_meta_key="custom_name")
        monkeypatch.setenv("S3_DOWNLOADER_PREFIX", "")
        S3Downloader.warm_up(d)
        docs = [
            Document(meta={"custom_name": "text-sample.txt"}),
        ]
        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["custom_name"] == "text-sample.txt"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("S3_DOWNLOADER_BUCKET", None),
        reason="Export an env var called `S3_DOWNLOADER_BUCKET` containing the S3 bucket to run this test.",
    )
    def test_live_run_with_prefix(self, tmp_path, monkeypatch):
        d = S3Downloader(file_root_path=str(tmp_path))
        monkeypatch.setenv("S3_DOWNLOADER_PREFIX", "subfolder/")

        S3Downloader.warm_up(d)
        docs = [
            Document(meta={"file_name": "employees.json"}),
        ]
        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["file_name"] == "employees.json"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("S3_DOWNLOADER_BUCKET", None),
        reason="Export an env var called `S3_DOWNLOADER_BUCKET` containing the S3 bucket to run this test.",
    )
    def test_live_run_with_s3_key_generation_function_and_file_extensions(self, tmp_path):
        # the file in the s3 bucket has this key: "dog.jpg_suffix"

        d = S3Downloader(
            file_root_path=str(tmp_path),
            file_extensions=[".jpg"],
            file_name_meta_key="file_name",
            s3_key_generation_function=s3_key_generation_function,
        )
        S3Downloader.warm_up(d)
        docs = [
            Document(meta={"file_name": "dog.jpg"}),
        ]
        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["file_name"] == "dog.jpg"
