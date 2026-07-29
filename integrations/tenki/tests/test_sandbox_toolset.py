# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from haystack.tools.errors import ToolInvocationError
from haystack.utils import Secret

from haystack_integrations.tools.tenki.bash_tool import RunBashCommandTool
from haystack_integrations.tools.tenki.list_directory_tool import ListDirectoryTool
from haystack_integrations.tools.tenki.read_file_tool import ReadFileTool
from haystack_integrations.tools.tenki.sandbox_toolset import TenkiToolset
from haystack_integrations.tools.tenki.tenki_sandbox import TenkiSandbox
from haystack_integrations.tools.tenki.write_file_tool import WriteFileTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sandbox(**kwargs) -> TenkiSandbox:
    """Create a TenkiSandbox with a dummy auth token for testing."""
    defaults = {"auth_token": Secret.from_token("test-auth-token")}
    defaults.update(kwargs)
    return TenkiSandbox(**defaults)


def _make_sandbox_mock() -> MagicMock:
    """Return a MagicMock that mimics the tenki_sandbox Sandbox object."""
    sandbox = MagicMock()
    sandbox.id = "sandbox-test-123"
    return sandbox


def _make_result(*, ok=True, exit_code=0, signal=None, reason=None, stdout="", stderr="") -> MagicMock:
    """Return a MagicMock mimicking tenki_sandbox.CommandResult."""
    result = MagicMock()
    result.ok = ok
    result.exit_code = exit_code
    result.signal = signal
    result.reason = reason
    result.stdout_text = stdout
    result.stderr_text = stderr
    return result


def _sandbox_with_mock() -> tuple[TenkiSandbox, MagicMock]:
    """Return a TenkiSandbox that already has a mocked underlying sandbox."""
    sb = _make_sandbox()
    mock = _make_sandbox_mock()
    sb._sandbox = mock
    return sb, mock


# ---------------------------------------------------------------------------
# TenkiSandbox -- initialisation
# ---------------------------------------------------------------------------


class TestTenkiSandboxInit:
    def test_class_defaults(self):
        """Verify the real class defaults, not values set by a helper."""
        sandbox = TenkiSandbox(auth_token=Secret.from_token("test-auth-token"))
        assert sandbox.name == "haystack"
        assert sandbox.base_url is None
        assert sandbox.cpu_cores is None
        assert sandbox.memory_mb is None
        assert sandbox.max_duration is None
        assert sandbox.idle_timeout_minutes is None
        assert sandbox.environment_vars == {}
        assert sandbox._sandbox is None

    def test_custom_parameters(self):
        sandbox = _make_sandbox(
            name="my-box",
            base_url="https://api.example.com",
            cpu_cores=4,
            memory_mb=8192,
            max_duration=3600,
            idle_timeout_minutes=60,
            environment_vars={"FOO": "bar"},
        )
        assert sandbox.name == "my-box"
        assert sandbox.base_url == "https://api.example.com"
        assert sandbox.cpu_cores == 4
        assert sandbox.memory_mb == 8192
        assert sandbox.max_duration == 3600
        assert sandbox.idle_timeout_minutes == 60
        assert sandbox.environment_vars == {"FOO": "bar"}


# ---------------------------------------------------------------------------
# TenkiSandbox -- warm_up
# ---------------------------------------------------------------------------


class TestTenkiSandboxWarmUp:
    @patch("haystack_integrations.tools.tenki.tenki_sandbox.tenki_import")
    @patch("haystack_integrations.tools.tenki.tenki_sandbox.Sandbox.create")
    def test_warm_up_creates_sandbox(self, mock_sandbox_create, mock_tenki_import):
        mock_tenki_import.check.return_value = None
        mock_instance = _make_sandbox_mock()
        mock_sandbox_create.return_value = mock_instance

        sb = _make_sandbox()
        sb.warm_up()

        # Only set values are passed; None kwargs are dropped so the SDK applies defaults.
        mock_sandbox_create.assert_called_once_with(wait=False, name="haystack", auth_token="test-auth-token")
        assert sb._sandbox is mock_instance

    @patch("haystack_integrations.tools.tenki.tenki_sandbox.tenki_import")
    @patch("haystack_integrations.tools.tenki.tenki_sandbox.Sandbox.create")
    def test_warm_up_passes_all_set_kwargs(self, mock_sandbox_create, mock_tenki_import):
        mock_tenki_import.check.return_value = None
        mock_sandbox_create.return_value = _make_sandbox_mock()

        sb = _make_sandbox(
            name="box",
            base_url="https://api.example.com",
            cpu_cores=2,
            memory_mb=4096,
            max_duration=1800,
            idle_timeout_minutes=45,
            environment_vars={"MY_VAR": "value"},
        )
        sb.warm_up()

        _, kwargs = mock_sandbox_create.call_args
        assert kwargs == {
            "wait": False,
            "name": "box",
            "auth_token": "test-auth-token",
            "base_url": "https://api.example.com",
            "cpu_cores": 2,
            "memory_mb": 4096,
            "max_duration": 1800,
            "idle_timeout_minutes": 45,
            "env": {"MY_VAR": "value"},
        }

    @patch("haystack_integrations.tools.tenki.tenki_sandbox.tenki_import")
    @patch("haystack_integrations.tools.tenki.tenki_sandbox.Sandbox.create")
    def test_warm_up_is_idempotent(self, mock_sandbox_create, mock_tenki_import):
        mock_tenki_import.check.return_value = None
        mock_sandbox_create.return_value = _make_sandbox_mock()

        sb = _make_sandbox()
        sb.warm_up()
        sb.warm_up()

        mock_sandbox_create.assert_called_once()

    @patch("haystack_integrations.tools.tenki.tenki_sandbox.tenki_import")
    @patch("haystack_integrations.tools.tenki.tenki_sandbox.Sandbox.create")
    def test_warm_up_raises_on_sandbox_error(self, mock_sandbox_create, mock_tenki_import):
        mock_tenki_import.check.return_value = None
        mock_sandbox_create.side_effect = Exception("connection refused")

        sb = _make_sandbox()
        with pytest.raises(RuntimeError, match="Failed to start Tenki sandbox"):
            sb.warm_up()
        assert sb._sandbox is None

    @patch("haystack_integrations.tools.tenki.tenki_sandbox.logger")
    @patch("haystack_integrations.tools.tenki.tenki_sandbox.tenki_import")
    @patch("haystack_integrations.tools.tenki.tenki_sandbox.Sandbox.create")
    def test_warm_up_is_cancellation_safe(self, mock_sandbox_create, mock_tenki_import, mock_logger):
        """If anything fails after the VM is created (here a simulated
        cancellation during the post-create log), the VM must be torn down and
        no handle retained — so a RUNNING sandbox is never leaked."""
        mock_tenki_import.check.return_value = None
        mock_instance = _make_sandbox_mock()
        mock_sandbox_create.return_value = mock_instance
        # First logger.info (pre-create) is fine; the second (post-create) raises.
        mock_logger.info.side_effect = [None, KeyboardInterrupt()]

        sb = _make_sandbox()
        with pytest.raises(KeyboardInterrupt):
            sb.warm_up()

        mock_instance.close_if_open.assert_called_once()
        assert sb._sandbox is None

    @patch("haystack_integrations.tools.tenki.tenki_sandbox.tenki_import")
    @patch("haystack_integrations.tools.tenki.tenki_sandbox.Sandbox.create")
    def test_warm_up_awaits_readiness_with_a_handle(self, mock_sandbox_create, mock_tenki_import):
        """The VM is created with wait=False so readiness is awaited with a handle in hand."""
        mock_tenki_import.check.return_value = None
        mock_instance = _make_sandbox_mock()
        mock_sandbox_create.return_value = mock_instance

        sb = _make_sandbox()
        sb.warm_up()

        assert mock_sandbox_create.call_args.kwargs["wait"] is False
        mock_instance.wait_ready.assert_called_once()
        assert sb._sandbox is mock_instance

    @patch("haystack_integrations.tools.tenki.tenki_sandbox.tenki_import")
    @patch("haystack_integrations.tools.tenki.tenki_sandbox.Sandbox.create")
    def test_warm_up_terminates_vm_when_readiness_fails(self, mock_sandbox_create, mock_tenki_import):
        """A readiness failure must tear down the already-provisioned microVM."""
        mock_tenki_import.check.return_value = None
        mock_instance = _make_sandbox_mock()
        mock_instance.wait_ready.side_effect = Exception("never became ready")
        mock_sandbox_create.return_value = mock_instance

        sb = _make_sandbox()
        with pytest.raises(RuntimeError, match="Failed to start Tenki sandbox"):
            sb.warm_up()

        mock_instance.close_if_open.assert_called_once()
        assert sb._sandbox is None

    @patch("haystack_integrations.tools.tenki.tenki_sandbox.tenki_import")
    @patch("haystack_integrations.tools.tenki.tenki_sandbox.Sandbox.create")
    def test_concurrent_warm_up_creates_a_single_vm(self, mock_sandbox_create, mock_tenki_import):
        """Racing callers must not each provision a VM and overwrite the handle."""
        mock_tenki_import.check.return_value = None
        callers = 8
        at_the_gate = threading.Barrier(callers)

        def _slow_create(**_kwargs):
            time.sleep(0.05)
            return _make_sandbox_mock()

        mock_sandbox_create.side_effect = _slow_create
        sb = _make_sandbox()

        def _warm_up():
            at_the_gate.wait(timeout=5)
            sb.warm_up()

        threads = [threading.Thread(target=_warm_up) for _ in range(callers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert mock_sandbox_create.call_count == 1
        assert sb._sandbox is not None


# ---------------------------------------------------------------------------
# TenkiSandbox -- close
# ---------------------------------------------------------------------------


class TestTenkiSandboxClose:
    def test_close_without_warm_up_is_noop(self):
        sb = _make_sandbox()
        sb.close()
        assert sb._sandbox is None

    def test_close_terminates_sandbox(self):
        sb, mock = _sandbox_with_mock()
        sb.close()
        mock.close_if_open.assert_called_once()
        assert sb._sandbox is None

    def test_close_does_not_swallow_error_and_keeps_handle(self):
        """A failed terminate must surface (not be reported as success) and the
        handle must be retained so close() can be retried."""
        sb, mock = _sandbox_with_mock()
        mock.close_if_open.side_effect = Exception("terminate failed")
        with pytest.raises(Exception, match="terminate failed"):
            sb.close()
        assert sb._sandbox is mock  # handle retained for retry


# ---------------------------------------------------------------------------
# TenkiSandbox -- handle validity
# ---------------------------------------------------------------------------


class TestTenkiSandboxRequireSandbox:
    def test_running_sandbox_is_returned_as_is(self):
        sb, mock = _sandbox_with_mock()
        mock.state = "running"

        assert sb._require_sandbox() is mock
        mock.refresh.assert_called_once()
        mock.resume.assert_not_called()

    def test_paused_sandbox_is_resumed_before_use(self):
        """An idle timeout can pause the VM while the handle stays non-None."""
        sb, mock = _sandbox_with_mock()
        mock.state = "paused"

        assert sb._require_sandbox() is mock
        mock.refresh.assert_called_once()
        mock.resume.assert_called_once()
        mock.wait_ready.assert_called_once()

    def test_resume_failure_surfaces_as_runtime_error(self):
        sb, mock = _sandbox_with_mock()
        mock.state = "paused"
        mock.resume.side_effect = Exception("resume rejected")

        with pytest.raises(RuntimeError, match="Failed to resume paused Tenki sandbox"):
            sb._require_sandbox()


# ---------------------------------------------------------------------------
# TenkiSandbox -- serialisation
# ---------------------------------------------------------------------------


class TestTenkiSandboxSerialisation:
    def _make_env_sandbox(self, **kwargs) -> TenkiSandbox:
        defaults = {"auth_token": Secret.from_env_var("TENKI_AUTH_TOKEN")}
        defaults.update(kwargs)
        return TenkiSandbox(**defaults)

    def test_to_dict_contains_expected_keys(self):
        sb = self._make_env_sandbox(name="my-box", cpu_cores=2, idle_timeout_minutes=30)
        data = sb.to_dict()

        assert "type" in data
        assert "data" in data
        assert data["data"]["name"] == "my-box"
        assert data["data"]["cpu_cores"] == 2
        assert data["data"]["idle_timeout_minutes"] == 30

    def test_to_dict_does_not_include_sandbox_instance(self):
        sb = self._make_env_sandbox()
        sb._sandbox = _make_sandbox_mock()
        data = sb.to_dict()

        assert "_sandbox" not in data["data"]
        assert "sandbox" not in data["data"]

    def test_from_dict_round_trip(self):
        original = self._make_env_sandbox(
            name="custom",
            cpu_cores=4,
            memory_mb=8192,
            max_duration=3600,
            idle_timeout_minutes=90,
            environment_vars={"KEY": "value"},
        )
        data = original.to_dict()
        restored = TenkiSandbox.from_dict(data)

        assert restored.name == "custom"
        assert restored.cpu_cores == 4
        assert restored.memory_mb == 8192
        assert restored.max_duration == 3600
        assert restored.idle_timeout_minutes == 90
        assert restored.environment_vars == {"KEY": "value"}
        assert restored._sandbox is None

    def test_to_dict_type_is_qualified_class_name(self):
        sb = self._make_env_sandbox()
        data = sb.to_dict()
        assert "TenkiSandbox" in data["type"]

    def test_to_dict_includes_stable_instance_id(self):
        sb = self._make_env_sandbox()
        data = sb.to_dict()
        assert data["data"]["instance_id"] == sb.instance_id

    def test_from_dict_preserves_instance_id(self):
        original = self._make_env_sandbox()
        restored = TenkiSandbox.from_dict(original.to_dict())
        assert restored.instance_id == original.instance_id

    def test_from_dict_dedupes_tools_sharing_one_sandbox(self):
        """Tools that shared one sandbox before serialization share it after round-trip."""
        TenkiSandbox._instances.clear()
        sandbox = self._make_env_sandbox(name="custom", cpu_cores=2)
        tool_a = RunBashCommandTool(sandbox=sandbox)
        tool_b = ReadFileTool(sandbox=sandbox)

        restored_a = RunBashCommandTool.from_dict(tool_a.to_dict())
        restored_b = ReadFileTool.from_dict(tool_b.to_dict())

        assert restored_a._tenki_sandbox is restored_b._tenki_sandbox
        assert restored_a._tenki_sandbox.instance_id == sandbox.instance_id

    def test_from_dict_distinct_sandboxes_remain_distinct(self):
        """Two separately-built sandboxes with identical config keep distinct identities."""
        TenkiSandbox._instances.clear()
        sb1 = self._make_env_sandbox(name="box", cpu_cores=1)
        sb2 = self._make_env_sandbox(name="box", cpu_cores=1)
        assert sb1.instance_id != sb2.instance_id

        restored_1 = TenkiSandbox.from_dict(sb1.to_dict())
        restored_2 = TenkiSandbox.from_dict(sb2.to_dict())

        assert restored_1 is not restored_2
        assert restored_1.instance_id == sb1.instance_id
        assert restored_2.instance_id == sb2.instance_id

    def test_from_dict_id_collision_with_mismatched_config_does_not_dedup(self, monkeypatch):
        """A crafted dict reusing another sandbox's id but with a different auth token
        must NOT receive the cached instance (no cross-tenant escalation), and must
        NOT evict the legitimate cache entry (no DoS)."""
        TenkiSandbox._instances.clear()
        monkeypatch.setenv("VICTIM_KEY", "victim-secret")
        monkeypatch.setenv("ATTACKER_KEY", "attacker-secret")

        legitimate_data = {
            "type": "haystack_integrations.tools.tenki.tenki_sandbox.TenkiSandbox",
            "data": {
                "instance_id": "shared-id",
                "auth_token": Secret.from_env_var("VICTIM_KEY").to_dict(),
                "base_url": None,
                "name": "haystack",
                "cpu_cores": None,
                "memory_mb": None,
                "max_duration": None,
                "idle_timeout_minutes": None,
                "environment_vars": {},
            },
        }
        legitimate = TenkiSandbox.from_dict(legitimate_data)
        assert TenkiSandbox._instances.get("shared-id") is legitimate

        attacker_data = {
            "type": "haystack_integrations.tools.tenki.tenki_sandbox.TenkiSandbox",
            "data": {
                "instance_id": "shared-id",
                "auth_token": Secret.from_env_var("ATTACKER_KEY").to_dict(),
                "base_url": None,
                "name": "haystack",
                "cpu_cores": None,
                "memory_mb": None,
                "max_duration": None,
                "idle_timeout_minutes": None,
                "environment_vars": {},
            },
        }
        attacker = TenkiSandbox.from_dict(attacker_data)

        assert attacker is not legitimate
        assert attacker.auth_token.resolve_value() == "attacker-secret"
        assert legitimate.auth_token.resolve_value() == "victim-secret"
        # Cache still points at the legitimate instance — attacker did not evict it.
        assert TenkiSandbox._instances.get("shared-id") is legitimate


# ---------------------------------------------------------------------------
# Tool classes -- structure
# ---------------------------------------------------------------------------


class TestToolClasses:
    def test_run_bash_command_tool_name_and_schema(self):
        tool = RunBashCommandTool(sandbox=_make_sandbox())
        assert tool.name == "run_bash_command"
        assert tool.description
        assert "command" in tool.parameters["required"]

    def test_read_file_tool_name_and_schema(self):
        tool = ReadFileTool(sandbox=_make_sandbox())
        assert tool.name == "read_file"
        assert tool.description
        assert "path" in tool.parameters["required"]

    def test_write_file_tool_name_and_schema(self):
        tool = WriteFileTool(sandbox=_make_sandbox())
        assert tool.name == "write_file"
        assert tool.description
        assert "path" in tool.parameters["required"]
        assert "content" in tool.parameters["required"]

    def test_list_directory_tool_name_and_schema(self):
        tool = ListDirectoryTool(sandbox=_make_sandbox())
        assert tool.name == "list_directory"
        assert tool.description
        assert "path" in tool.parameters["required"]

    def test_tool_stores_sandbox_reference(self):
        sb = _make_sandbox()
        tool = RunBashCommandTool(sandbox=sb)
        assert tool._tenki_sandbox is sb

    def test_tenki_toolset_contains_four_tools(self):
        ts = TenkiToolset(auth_token=Secret.from_token("test-auth-token"))
        assert len(ts) == 4
        names = {t.name for t in ts}
        assert names == {"run_bash_command", "read_file", "write_file", "list_directory"}

    def test_tenki_toolset_has_correct_tool_types(self):
        ts = TenkiToolset(auth_token=Secret.from_token("test-auth-token"))
        tool_types = {type(t) for t in ts}
        assert tool_types == {RunBashCommandTool, ReadFileTool, WriteFileTool, ListDirectoryTool}

    def test_tenki_toolset_shares_same_sandbox(self):
        ts = TenkiToolset(auth_token=Secret.from_token("test-auth-token"))
        assert all(t._tenki_sandbox is ts.sandbox for t in ts)

        mock = _make_sandbox_mock()
        mock.exec.return_value = _make_result(stdout="ok")
        ts.sandbox._sandbox = mock

        bash_tool = next(t for t in ts if t.name == "run_bash_command")
        bash_tool.invoke(command="echo ok")

        mock.exec.assert_called_once()

    def test_tenki_toolset_round_trip_preserves_instance_id(self):
        """Without the id, a restored toolset cannot rejoin the tools that shared its sandbox."""
        ts = TenkiToolset(auth_token=Secret.from_env_var("TENKI_AUTH_TOKEN"))
        restored = TenkiToolset.from_dict(ts.to_dict())

        assert restored.sandbox.instance_id == ts.sandbox.instance_id

    def test_tenki_toolset_default_auth_token(self):
        """TenkiToolset uses the env-var secret when auth_token is omitted."""
        ts = TenkiToolset()
        assert ts.sandbox.auth_token is not None

    def test_tools_from_same_sandbox_share_state(self):
        sb = _make_sandbox()
        bash_tool = RunBashCommandTool(sandbox=sb)
        read_tool = ReadFileTool(sandbox=sb)
        assert bash_tool._tenki_sandbox is read_tool._tenki_sandbox


# ---------------------------------------------------------------------------
# RunBashCommandTool behaviour
# ---------------------------------------------------------------------------


class TestRunBashCommandTool:
    def test_returns_formatted_output(self):
        sb, mock = _sandbox_with_mock()
        mock.exec.return_value = _make_result(ok=True, exit_code=0, stdout="hello world\n")
        tool = RunBashCommandTool(sandbox=sb)

        output = tool.invoke(command="echo hello world")

        assert "ok: True" in output
        assert "exit_code: 0" in output
        assert "hello world" in output
        mock.exec.assert_called_once_with("bash", "-lc", "echo hello world", timeout=60)

    def test_reports_nonzero_exit(self):
        sb, mock = _sandbox_with_mock()
        mock.exec.return_value = _make_result(ok=False, exit_code=42, stderr="boom")
        tool = RunBashCommandTool(sandbox=sb)

        output = tool.invoke(command="exit 42")

        assert "ok: False" in output
        assert "exit_code: 42" in output
        assert "boom" in output

    def test_signal_killed_is_not_reported_as_success(self):
        """A process terminated by a signal (exit_code 0 + signal) must read as
        ok: False and surface the signal — never as success."""
        sb, mock = _sandbox_with_mock()
        mock.exec.return_value = _make_result(ok=False, exit_code=0, signal="SIGKILL", reason="oom")
        tool = RunBashCommandTool(sandbox=sb)

        output = tool.invoke(command="trigger-oom")

        assert "ok: False" in output
        assert "signal: SIGKILL" in output
        assert "reason: oom" in output

    def test_passes_custom_timeout(self):
        sb, mock = _sandbox_with_mock()
        mock.exec.return_value = _make_result()
        tool = RunBashCommandTool(sandbox=sb)

        tool.invoke(command="sleep 5", timeout=30)

        mock.exec.assert_called_once_with("bash", "-lc", "sleep 5", timeout=30)

    @patch("haystack_integrations.tools.tenki.tenki_sandbox.tenki_import")
    @patch("haystack_integrations.tools.tenki.tenki_sandbox.Sandbox.create")
    def test_wraps_warm_up_failure(self, mock_sandbox_create, mock_tenki_import):
        mock_tenki_import.check.return_value = None
        mock_sandbox_create.side_effect = Exception("connection refused")
        tool = RunBashCommandTool(sandbox=_make_sandbox())
        with pytest.raises(ToolInvocationError, match="Failed to start Tenki sandbox"):
            tool.invoke(command="ls")

    def test_wraps_sandbox_exception(self):
        sb, mock = _sandbox_with_mock()
        mock.exec.side_effect = Exception("timeout")
        tool = RunBashCommandTool(sandbox=sb)
        with pytest.raises(ToolInvocationError, match="Failed to run bash command"):
            tool.invoke(command="sleep 1000")


# ---------------------------------------------------------------------------
# ReadFileTool behaviour
# ---------------------------------------------------------------------------


class TestReadFileTool:
    def test_returns_string(self):
        sb, mock = _sandbox_with_mock()
        mock.fs.read_text.return_value = "file content"
        tool = ReadFileTool(sandbox=sb)

        result = tool.invoke(path="/some/file.txt")

        assert result == "file content"
        mock.fs.read_text.assert_called_once_with("/some/file.txt")

    def test_wraps_sandbox_exception(self):
        sb, mock = _sandbox_with_mock()
        mock.fs.read_text.side_effect = Exception("file not found")
        tool = ReadFileTool(sandbox=sb)
        with pytest.raises(ToolInvocationError, match="Failed to read file"):
            tool.invoke(path="/nonexistent.txt")


# ---------------------------------------------------------------------------
# WriteFileTool behaviour
# ---------------------------------------------------------------------------


class TestWriteFileTool:
    def test_returns_confirmation(self):
        sb, mock = _sandbox_with_mock()
        tool = WriteFileTool(sandbox=sb)

        result = tool.invoke(path="/output/result.txt", content="hello")

        assert "/output/result.txt" in result
        mock.fs.write_text.assert_called_once_with("/output/result.txt", "hello")

    def test_wraps_sandbox_exception(self):
        sb, mock = _sandbox_with_mock()
        mock.fs.write_text.side_effect = Exception("permission denied")
        tool = WriteFileTool(sandbox=sb)
        with pytest.raises(ToolInvocationError, match="Failed to write file"):
            tool.invoke(path="/protected/file.txt", content="data")


# ---------------------------------------------------------------------------
# ListDirectoryTool behaviour
# ---------------------------------------------------------------------------


class TestListDirectoryTool:
    def _make_entry(self, path: str, is_dir: bool = False) -> MagicMock:
        entry = MagicMock()
        entry.path = path
        entry.is_dir = is_dir
        return entry

    def test_returns_names(self):
        sb, mock = _sandbox_with_mock()
        mock.fs.list.return_value = [
            self._make_entry("/home/user/file.txt"),
            self._make_entry("/home/user/subdir", is_dir=True),
        ]
        tool = ListDirectoryTool(sandbox=sb)

        result = tool.invoke(path="/home/user")

        assert "file.txt" in result
        assert "subdir/" in result
        mock.fs.list.assert_called_once_with("/home/user")

    def test_empty_directory(self):
        sb, mock = _sandbox_with_mock()
        mock.fs.list.return_value = []
        tool = ListDirectoryTool(sandbox=sb)

        result = tool.invoke(path="/empty")

        assert result == "(empty directory)"

    def test_wraps_sandbox_exception(self):
        sb, mock = _sandbox_with_mock()
        mock.fs.list.side_effect = Exception("not a directory")
        tool = ListDirectoryTool(sandbox=sb)
        with pytest.raises(ToolInvocationError, match="Failed to list directory"):
            tool.invoke(path="/nonexistent")
