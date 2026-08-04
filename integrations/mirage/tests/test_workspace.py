# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
from haystack import default_from_dict, default_to_dict
from haystack.utils import Secret

from haystack_integrations.tools.mirage import MirageConfigError, MirageMount, MirageWorkspace
from haystack_integrations.tools.mirage import workspace as workspace_module
from haystack_integrations.tools.mirage.workspace import _resolve_config, _to_text


class _FakeTokenSource:
    """A config-only OAuth token source (shaped like OAuthRefreshTokenSource) for tests."""

    requires_subject_token = False

    def __init__(self, value="access-tok"):
        self.value = value

    def resolve(self):
        return self.value

    def to_dict(self):
        return default_to_dict(self, value=self.value)

    @classmethod
    def from_dict(cls, data):
        return default_from_dict(cls, data)


class _FakeSubjectTokenSource:
    """A per-request (subject-token) source, which the integration must reject at build time."""

    requires_subject_token = True

    def resolve(self, subject_token):
        return subject_token

    def to_dict(self):
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data):
        return default_from_dict(cls, data)


class _FakeResult:
    """Stand-in for a Mirage IOResult to unit-test _to_text without running a workspace."""

    def __init__(self, stdout=b"", stderr=None, exit_code=0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class TestToText:
    def test_decodes_stdout(self):
        assert _to_text(_FakeResult(stdout=b"hello\n")) == "hello\n"

    def test_appends_error_on_nonzero_exit(self):
        text = _to_text(_FakeResult(stdout=b"", stderr=b"boom", exit_code=1))
        assert "[exit code 1]" in text
        assert "boom" in text

    def test_passes_through_str(self):
        assert _to_text("already text") == "already text"

    def test_truncates(self):
        text = _to_text(_FakeResult(stdout=b"x" * 100), max_chars=10)
        assert text.startswith("xxxxxxxxxx")
        assert "truncated" in text


class TestMirageMount:
    def test_generic_construction(self):
        m = MirageMount(path="/local", resource="disk", config={"root": "/mnt/data"}, read_only=True)
        assert m.path == "/local"
        assert m.resource == "disk"
        assert m.config == {"root": "/mnt/data"}
        assert m.read_only is True

    def test_defaults(self):
        m = MirageMount(path="/data", resource="ram")
        assert m.config == {}
        assert m.read_only is False

    def test_available_resources_lists_registry_names(self):
        names = MirageMount.available_resources()
        assert isinstance(names, list)
        # a few backends we know exist in the Mirage registry
        assert {"ram", "disk", "s3", "gdrive"}.issubset(set(names))


class TestMirageWorkspaceConfig:
    def test_requires_mounts(self):
        with pytest.raises(MirageConfigError):
            MirageWorkspace(mounts=[])

    def test_rejects_duplicate_paths(self):
        with pytest.raises(MirageConfigError):
            MirageWorkspace(
                mounts=[
                    MirageMount(path="/data", resource="ram"),
                    MirageMount(path="/data", resource="ram"),
                ]
            )

    def test_unknown_resource_raises_on_build(self):
        ws = MirageWorkspace(mounts=[MirageMount(path="/x", resource="does_not_exist")])
        with pytest.raises(MirageConfigError):
            ws.warm_up()


class TestSerialization:
    def test_roundtrip_plain(self):
        ws = MirageWorkspace(
            mounts=[
                MirageMount(path="/data", resource="ram"),
                MirageMount(path="/s3", resource="s3", config={"bucket": "b"}, read_only=True),
            ],
            cache_limit="256MB",
        )
        data = ws.to_dict()
        restored = MirageWorkspace.from_dict(data)
        assert restored.cache_limit == "256MB"
        assert [m.path for m in restored.mounts] == ["/data", "/s3"]
        assert restored.mounts[1].read_only is True
        assert restored.mounts[1].config == {"bucket": "b"}

    def test_roundtrip_with_secret_does_not_leak_plaintext(self, monkeypatch):
        monkeypatch.setenv("GDRIVE_REFRESH_TOKEN", "super-secret-value")
        ws = MirageWorkspace(
            mounts=[
                MirageMount(
                    path="/drive",
                    resource="gdrive",
                    config={
                        "client_id": "client-123",
                        "refresh_token": Secret.from_env_var("GDRIVE_REFRESH_TOKEN"),
                    },
                )
            ]
        )
        data = ws.to_dict()
        # The secret must be serialized by reference, never as plaintext.
        assert "super-secret-value" not in str(data)
        restored = MirageWorkspace.from_dict(data)
        token = restored.mounts[0].config["refresh_token"]
        assert isinstance(token, Secret)
        assert token.resolve_value() == "super-secret-value"


class TestExecution:
    def test_run_ram_roundtrip(self):
        ws = MirageWorkspace(mounts=[MirageMount(path="/data", resource="ram")])
        ws.run('echo "hello mirage" > /data/a.txt')
        out = ws.run("cat /data/a.txt")
        assert "hello mirage" in out
        ws.close()

    def test_run_disk(self, tmp_path):
        ws = MirageWorkspace(mounts=[MirageMount(path="/local", resource="disk", config={"root": str(tmp_path)})])
        ws.run('echo "on disk" > /local/note.txt')
        out = ws.run("cat /local/note.txt")
        assert "on disk" in out
        # the file should really exist on disk
        assert (tmp_path / "note.txt").read_text().strip() == "on disk"
        ws.close()

    def test_read_only_mount_rejects_writes(self):
        ws = MirageWorkspace(mounts=[MirageMount(path="/ro", resource="ram", read_only=True)])
        out = ws.run('echo "nope" > /ro/x.txt')
        assert "read-only" in out
        ws.close()

    @pytest.mark.asyncio
    async def test_run_async(self):
        ws = MirageWorkspace(mounts=[MirageMount(path="/data", resource="ram")])
        await ws.run_async('echo "async hi" > /data/b.txt')
        out = await ws.run_async("cat /data/b.txt")
        assert "async hi" in out
        ws.close()


class TestConcurrentBuild:
    def test_ensure_live_builds_once_under_concurrency(self, monkeypatch):
        # Many threads hitting a cold workspace at once must build exactly one live workspace; a second
        # build would leak the losing thread's backend resources (the bug the build lock prevents).
        real_cls = workspace_module._MirageWorkspace
        build_count = 0
        count_lock = threading.Lock()

        def counting_ctor(*args, **kwargs):
            nonlocal build_count
            with count_lock:
                build_count += 1
            return real_cls(*args, **kwargs)

        monkeypatch.setattr(workspace_module, "_MirageWorkspace", counting_ctor)

        ws = MirageWorkspace(mounts=[MirageMount(path="/data", resource="ram")])
        n_threads = 8
        barrier = threading.Barrier(n_threads)
        results: list = []
        results_lock = threading.Lock()

        def build() -> None:
            barrier.wait()  # release all threads together to maximize contention
            live = ws._ensure_live()
            with results_lock:
                results.append(live)

        threads = [threading.Thread(target=build) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert build_count == 1
        assert all(r is results[0] for r in results)
        ws.close()


class TestTokenSourceConfig:
    def test_resolve_turns_config_only_source_into_callable(self):
        resolved = _resolve_config({"access_token": _FakeTokenSource("tok-123")})
        provider = resolved["access_token"]
        assert callable(provider)
        assert provider() == "tok-123"

    def test_resolve_rejects_subject_token_source(self):
        with pytest.raises(MirageConfigError):
            _resolve_config({"access_token": _FakeSubjectTokenSource()})

    def test_token_source_serialization_roundtrip(self):
        ws = MirageWorkspace(
            mounts=[
                MirageMount(
                    path="/onedrive",
                    resource="onedrive",
                    config={"access_token": _FakeTokenSource("secret-tok")},
                    read_only=True,
                )
            ]
        )
        data = ws.to_dict()
        serialized = data["mounts"][0]["config"]["access_token"]
        assert "__mirage_token_source__" in serialized

        restored = MirageWorkspace.from_dict(data)
        src = restored.mounts[0].config["access_token"]
        assert src.resolve() == "secret-tok"
        # and it resolves to a working token-provider callable at build time
        assert _resolve_config(restored.mounts[0].config)["access_token"]() == "secret-tok"
