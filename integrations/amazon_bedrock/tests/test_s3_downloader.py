import os
from pathlib import Path
from typing import Any
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
    def test_init_custom_parameters(self, tmp_path):
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
            s3_bucket_name_env="b",
        )
        assert d.file_extensions == [".pdf", ".txt"]
        assert d._storage is None

    @pytest.mark.parametrize("boto3_config", [None, {"read_timeout": 10}])
    def test_to_dict(self, tmp_path, boto3_config: dict[str, Any] | None):
        d = S3Downloader(file_root_path=str(tmp_path), boto3_config=boto3_config)
        expected = {
            "type": TYPE,
            "init_parameters": {
                "aws_access_key_id": {
                    "type": "env_var",
                    "env_vars": ["AWS_ACCESS_KEY_ID"],
                    "strict": False,
                },
                "aws_secret_access_key": {
                    "type": "env_var",
                    "env_vars": ["AWS_SECRET_ACCESS_KEY"],
                    "strict": False,
                },
                "aws_region_name": {
                    "type": "env_var",
                    "env_vars": ["AWS_DEFAULT_REGION"],
                    "strict": False,
                },
                "aws_session_token": {
                    "type": "env_var",
                    "env_vars": ["AWS_SESSION_TOKEN"],
                    "strict": False,
                },
                "aws_profile_name": {
                    "type": "env_var",
                    "env_vars": ["AWS_PROFILE"],
                    "strict": False,
                },
                "boto3_config": boto3_config,
                "file_root_path": str(tmp_path),
                "file_extensions": None,
                "max_cache_size": 100,
                "max_workers": 32,
                "file_name_meta_key": "file_name",
                "s3_key_generation_function": None,
                "s3_bucket_name_env": "S3_DOWNLOADER_BUCKET",
            },
        }
        assert d.to_dict() == expected

    def test_boto3_config_survives_a_serialization_round_trip(self, mock_boto3_session: Any, tmp_path):
        """boto3_config builds the botocore Config: timeouts, retries and proxies all live there."""
        downloader = S3Downloader(file_root_path=str(tmp_path), boto3_config={"read_timeout": 10, "connect_timeout": 5})

        restored = S3Downloader.from_dict(downloader.to_dict())

        assert restored.boto3_config == {"read_timeout": 10, "connect_timeout": 5}

    @pytest.mark.parametrize("boto3_config", [None, {"read_timeout": 10}])
    def test_from_dict(self, tmp_path, boto3_config: dict[str, Any] | None):
        data = {
            "type": TYPE,
            "init_parameters": {
                "aws_access_key_id": {
                    "type": "env_var",
                    "env_vars": ["AWS_ACCESS_KEY_ID"],
                    "strict": False,
                },
                "aws_secret_access_key": {
                    "type": "env_var",
                    "env_vars": ["AWS_SECRET_ACCESS_KEY"],
                    "strict": False,
                },
                "aws_region_name": {
                    "type": "env_var",
                    "env_vars": ["AWS_DEFAULT_REGION"],
                    "strict": False,
                },
                "aws_session_token": {
                    "type": "env_var",
                    "env_vars": ["AWS_SESSION_TOKEN"],
                    "strict": False,
                },
                "aws_profile_name": {
                    "type": "env_var",
                    "env_vars": ["AWS_PROFILE"],
                    "strict": False,
                },
                "file_root_path": str(tmp_path),
                "s3_key_generation_function": None,
                "s3_bucket_name_env": "S3_DOWNLOADER_BUCKET",
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
            s3_bucket_name_env="b",
        )
        expected = {
            "type": TYPE,
            "init_parameters": {
                "aws_access_key_id": {
                    "type": "env_var",
                    "env_vars": ["AWS_ACCESS_KEY_ID"],
                    "strict": False,
                },
                "aws_secret_access_key": {
                    "type": "env_var",
                    "env_vars": ["AWS_SECRET_ACCESS_KEY"],
                    "strict": False,
                },
                "aws_region_name": {
                    "type": "env_var",
                    "env_vars": ["AWS_DEFAULT_REGION"],
                    "strict": False,
                },
                "aws_session_token": {
                    "type": "env_var",
                    "env_vars": ["AWS_SESSION_TOKEN"],
                    "strict": False,
                },
                "aws_profile_name": {
                    "type": "env_var",
                    "env_vars": ["AWS_PROFILE"],
                    "strict": False,
                },
                "boto3_config": None,
                "file_root_path": str(tmp_path),
                "file_extensions": [".txt"],
                "max_cache_size": 400,
                "max_workers": 40,
                "file_name_meta_key": "new_file_key",
                "s3_key_generation_function": "tests.test_s3_downloader.s3_key_generation_function",
                "s3_bucket_name_env": "b",
            },
        }
        assert d.to_dict() == expected

    def test_run(self, tmp_path, mock_s3_storage):
        d = S3Downloader(file_root_path=str(tmp_path))
        d._storage = mock_s3_storage

        docs = [
            Document(meta={"file_id": str(uuid4()), "file_name": "a.txt"}),
            Document(meta={"file_id": str(uuid4()), "file_name": "b.txt"}),
        ]
        out = d.run(documents=docs)
        assert len(out["documents"]) == 2

    def test_run_with_extensions(self, tmp_path, mock_s3_storage):
        d = S3Downloader(file_root_path=str(tmp_path), file_extensions=[".txt"])
        d._storage = mock_s3_storage

        docs = [
            Document(meta={"file_id": str(uuid4()), "file_name": "a.txt"}),
            Document(meta={"file_id": str(uuid4()), "file_name": "b.pdf"}),
        ]

        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["file_name"] == "a.txt"

    def test_run_with_input_file_meta_key(self, tmp_path, mock_s3_storage):
        d = S3Downloader(file_root_path=str(tmp_path), file_name_meta_key="custom_file_key")
        d._storage = mock_s3_storage

        docs = [Document(meta={"file_id": str(uuid4()), "custom_file_key": "a.txt"})]

        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["custom_file_key"] == "a.txt"

    def test_run_with_s3_key_generation_function(self, tmp_path, mock_s3_storage):
        d = S3Downloader(
            file_root_path=str(tmp_path),
            s3_key_generation_function=s3_key_generation_function,
        )
        d._storage = mock_s3_storage

        docs = [Document(meta={"file_id": str(uuid4()), "file_name": "a.txt"})]
        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["file_name"] == "a.txt"

        mock_s3_storage.download.assert_called_once()
        assert mock_s3_storage.download.call_args.kwargs["key"] == "a.txt_suffix"

    def test_run_with_s3_key_generation_function_and_file_extensions(self, tmp_path, mock_s3_storage):
        d = S3Downloader(
            file_root_path=str(tmp_path),
            s3_key_generation_function=s3_key_generation_function,
            file_extensions=[".txt"],
        )
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

    def test_init_missing_file_root_path(self, monkeypatch):
        monkeypatch.delenv("FILE_ROOT_PATH", raising=False)
        with pytest.raises(ValueError, match="file_root_path"):
            S3Downloader()

    def test_init_file_root_path_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FILE_ROOT_PATH", str(tmp_path))
        d = S3Downloader()
        assert Path(d.file_root_path) == tmp_path

    def test_run_skips_document_missing_file_name_meta(self, tmp_path, mock_s3_storage):
        d = S3Downloader(file_root_path=str(tmp_path))
        d._storage = mock_s3_storage

        docs = [Document(meta={"file_id": str(uuid4())})]
        out = d.run(documents=docs)
        assert out["documents"] == []
        mock_s3_storage.download.assert_not_called()

    def test_run_file_already_cached(self, tmp_path, mock_s3_storage):
        existing_file = tmp_path / "cached.txt"
        existing_file.write_bytes(b"cached")

        d = S3Downloader(file_root_path=str(tmp_path))
        d._storage = mock_s3_storage

        out = d.run(documents=[Document(meta={"file_name": "cached.txt"})])
        assert out["documents"][0].meta["file_path"] == str(existing_file)
        mock_s3_storage.download.assert_not_called()

    def test_run_returns_empty_when_all_filtered(self, tmp_path, mock_s3_storage):
        d = S3Downloader(file_root_path=str(tmp_path), file_extensions=[".txt"])
        d._storage = mock_s3_storage

        out = d.run(documents=[Document(meta={"file_name": "a.pdf"})])
        assert out["documents"] == []
        mock_s3_storage.download.assert_not_called()

    def test_download_writes_to_temp_path_then_renames(self, tmp_path):
        d = S3Downloader(file_root_path=str(tmp_path))

        final_path = tmp_path / "test.pdf"
        captured_paths = []

        def fake_download(key, local_file_path: Path):
            captured_paths.append(Path(local_file_path))
            assert not final_path.exists(), "final path must not exist while download is in progress"
            Path(local_file_path).write_bytes(b"complete content")

        mock_storage = MagicMock(spec=S3Storage)
        mock_storage.download.side_effect = fake_download
        d._storage = mock_storage

        d.run(documents=[Document(meta={"file_name": "test.pdf"})])

        assert len(captured_paths) == 1
        assert captured_paths[0] != final_path
        assert captured_paths[0].name.startswith("test.pdf.tmp-")
        assert final_path.exists()
        assert final_path.read_bytes() == b"complete content"

    @pytest.mark.parametrize(
        ("file_name", "expected_warning"),
        [
            ("../escaped.txt", "outside of 'file_root_path'"),
            ("{tmp_path}/absolute.txt", "outside of 'file_root_path'"),
            ("sub/..", "Refusing to overwrite the download root directory"),
        ],
        ids=["traversal", "absolute", "root_itself"],
    )
    def test_run_skips_file_name_escaping_root(self, tmp_path, mock_s3_storage, caplog, file_name, expected_warning):
        # Attacker-controlled document metadata must not be able to write outside of file_root_path.
        root = tmp_path / "root"
        root.mkdir()
        d = S3Downloader(file_root_path=str(root))
        d._storage = mock_s3_storage

        out = d.run(documents=[Document(meta={"file_name": file_name.format(tmp_path=tmp_path)})])

        assert out["documents"] == []
        assert list(tmp_path.iterdir()) == [root]  # nothing was written outside of the root
        mock_s3_storage.download.assert_not_called()
        assert expected_warning in caplog.text

    def test_run_skips_only_the_escaping_document(self, tmp_path, mock_s3_storage):
        # One malicious document must not stop the batch, and nested names inside the root keep working.
        root = tmp_path / "root"
        root.mkdir()
        d = S3Downloader(file_root_path=str(root))
        d._storage = mock_s3_storage

        out = d.run(
            documents=[
                Document(meta={"file_name": "../escaped.txt"}),
                Document(meta={"file_name": "nested/dir/file.txt"}),
            ]
        )

        assert [doc.meta["file_path"] for doc in out["documents"]] == [str(root / "nested" / "dir" / "file.txt")]
        assert not (tmp_path / "escaped.txt").exists()

    def test_cleanup_cache_evicts_old_files(self, tmp_path, mock_s3_storage):
        d = S3Downloader(file_root_path=str(tmp_path), max_cache_size=1)
        d._storage = mock_s3_storage

        stale = tmp_path / "stale.txt"
        stale.write_bytes(b"stale")
        os.utime(stale, (0, 0))

        d.run(documents=[Document(meta={"file_name": "fresh.txt"})])

        assert not stale.exists()

    def test_from_dict_aws_region_name(self, tmp_path):
        """
        Test that aws_region_name as str value is correctly parsed
        """
        d = S3Downloader.from_dict(
            {
                "type": TYPE,
                "init_parameters": {
                    "aws_region_name": "my-fake-region",
                    "file_root_path": str(tmp_path),
                },
            }
        )
        assert d.aws_region_name == "my-fake-region"

        serialized = d.to_dict()
        assert serialized["init_parameters"]["aws_region_name"] == "my-fake-region"

    def test_from_dict_with_serialized_callable(self, tmp_path):
        data = {
            "type": TYPE,
            "init_parameters": {
                "file_root_path": str(tmp_path),
                "s3_key_generation_function": "tests.test_s3_downloader.s3_key_generation_function",
            },
        }
        d = S3Downloader.from_dict(data)
        assert d.s3_key_generation_function is s3_key_generation_function


class TestComponentLifecycle:
    @pytest.fixture
    def downloader(self, tmp_path, monkeypatch):
        monkeypatch.setenv("S3_DOWNLOADER_BUCKET", "bucket")
        return S3Downloader(file_root_path=str(tmp_path / "nested"))

    def test_warm_up_uses_resolved_credentials(self, downloader, mock_boto3_session, set_env_variables):
        downloader.warm_up()
        mock_boto3_session.assert_called_once_with(
            aws_access_key_id="some_fake_id",
            aws_secret_access_key="some_fake_key",
            aws_session_token="some_fake_token",
            region_name="fake_region",
            profile_name="some_fake_profile",
        )

    def test_key_resolved_at_warm_up_not_init(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MISSING_AWS_ACCESS_KEY", raising=False)
        downloader = S3Downloader(
            file_root_path=str(tmp_path),
            aws_access_key_id=Secret.from_env_var("MISSING_AWS_ACCESS_KEY"),
        )
        with pytest.raises(ValueError, match="MISSING_AWS_ACCESS_KEY"):
            downloader.warm_up()

    def test_sync_lifecycle(self, downloader):
        with patch(
            "haystack_integrations.components.downloaders.s3.s3_downloader.S3Storage.from_env"
        ) as storage_factory:
            storage = storage_factory.return_value
            downloader.warm_up()
            assert downloader._storage is storage
            assert downloader.file_root_path.is_dir()
            downloader.close()
            storage.close.assert_called_once_with()
            assert downloader._storage is None
            downloader.warm_up()
            assert storage_factory.call_count == 2

    def test_warm_up_is_idempotent(self, downloader):
        with patch(
            "haystack_integrations.components.downloaders.s3.s3_downloader.S3Storage.from_env"
        ) as storage_factory:
            downloader.warm_up()
            downloader.warm_up()
            storage_factory.assert_called_once()

    def test_close_is_safe_without_warm_up(self, downloader):
        downloader.close()
        assert downloader._storage is None


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("S3_DOWNLOADER_BUCKET", None),
    reason="Export an env var called `S3_DOWNLOADER_BUCKET` containing the S3 bucket to run this test.",
)
class TestS3DownloaderIntegration:
    def test_live_run(self, tmp_path, monkeypatch):
        d = S3Downloader(file_root_path=str(tmp_path))
        monkeypatch.setenv("S3_DOWNLOADER_PREFIX", "")
        docs = [
            Document(meta={"file_id": str(uuid4()), "file_name": "text-sample.txt"}),
            Document(meta={"file_id": str(uuid4()), "file_name": "document-sample.pdf"}),
        ]

        out = d.run(documents=docs)
        assert len(out["documents"]) == 2
        assert out["documents"][0].meta["file_name"] == "text-sample.txt"
        assert out["documents"][1].meta["file_name"] == "document-sample.pdf"

    def test_live_run_with_no_documents(self, tmp_path):
        d = S3Downloader(file_root_path=str(tmp_path))
        out = d.run(documents=[])
        assert len(out["documents"]) == 0

    def test_live_run_with_custom_meta_key(self, tmp_path, monkeypatch):
        d = S3Downloader(file_root_path=str(tmp_path), file_name_meta_key="custom_name")
        monkeypatch.setenv("S3_DOWNLOADER_PREFIX", "")
        docs = [Document(meta={"custom_name": "text-sample.txt"})]
        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["custom_name"] == "text-sample.txt"

    def test_live_run_with_prefix(self, tmp_path, monkeypatch):
        d = S3Downloader(file_root_path=str(tmp_path))
        monkeypatch.setenv("S3_DOWNLOADER_PREFIX", "subfolder/")
        docs = [Document(meta={"file_name": "employees.json"})]
        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["file_name"] == "employees.json"

    def test_live_run_with_s3_key_generation_function_and_file_extensions(self, tmp_path):
        # the file in the s3 bucket has this key: "dog.jpg_suffix"

        d = S3Downloader(
            file_root_path=str(tmp_path),
            file_extensions=[".jpg"],
            file_name_meta_key="file_name",
            s3_key_generation_function=s3_key_generation_function,
        )
        docs = [Document(meta={"file_name": "dog.jpg"})]
        out = d.run(documents=docs)
        assert len(out["documents"]) == 1
        assert out["documents"][0].meta["file_name"] == "dog.jpg"
