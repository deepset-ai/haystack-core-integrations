# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.tools.errors import ToolInvocationError
from haystack.utils import Secret

from haystack_integrations.tools.e2b.bash_tool import RunBashCommandTool
from haystack_integrations.tools.e2b.e2b_sandbox import E2BSandbox
from haystack_integrations.tools.e2b.list_directory_tool import ListDirectoryTool
from haystack_integrations.tools.e2b.read_file_tool import ReadFileTool
from haystack_integrations.tools.e2b.sandbox_toolset import E2BToolset
from haystack_integrations.tools.e2b.write_file_tool import WriteFileTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sandbox(**kwargs) -> E2BSandbox:
    """Create an E2BSandbox with a dummy API key for testing."""
    defaults = {"api_key": Secret.from_token("test-api-key")}
    defaults.update(kwargs)
    return E2BSandbox(**defaults)


def _make_sandbox_mock() -> MagicMock:
    """Return a MagicMock that mimics the e2b Sandbox object."""
    sandbox = MagicMock()
    sandbox.sandbox_id = "sandbox-test-123"
    return sandbox


def _sandbox_with_mock() -> tuple[E2BSandbox, MagicMock]:
    """Return an E2BSandbox that already has a mocked underlying sandbox."""
    sb = _make_sandbox()
    mock = _make_sandbox_mock()
    sb._sandbox = mock
    return sb, mock


# ---------------------------------------------------------------------------
# E2BSandbox -- initialisation
# ---------------------------------------------------------------------------


class TestE2BSandboxInit:
    def test_class_defaults(self):
        """Verify the real class defaults, not values set by a helper."""
        sandbox = E2BSandbox(api_key=Secret.from_token("test-api-key"))
        assert sandbox.sandbox_template == "base"
        assert sandbox.timeout == 120
        assert sandbox.environment_vars == {}
        assert sandbox._sandbox is None

    def test_custom_parameters(self):
        sandbox = _make_sandbox(
            sandbox_template="my-template",
            timeout=600,
            environment_vars={"FOO": "bar"},
        )
        assert sandbox.sandbox_template == "my-template"
        assert sandbox.timeout == 600
        assert sandbox.environment_vars == {"FOO": "bar"}


# ---------------------------------------------------------------------------
# E2BSandbox -- warm_up
# ---------------------------------------------------------------------------


class TestE2BSandboxWarmUp:
    @patch("haystack_integrations.tools.e2b.e2b_sandbox.e2b_import")
    @patch("haystack_integrations.tools.e2b.e2b_sandbox.Sandbox.create")
    def test_warm_up_creates_sandbox(self, mock_sandbox_create, mock_e2b_import):
        mock_e2b_import.check.return_value = None
        mock_instance = _make_sandbox_mock()
        mock_sandbox_create.return_value = mock_instance

        sb = _make_sandbox(sandbox_template="base", timeout=120)
        sb.warm_up()

        mock_sandbox_create.assert_called_once_with(
            api_key="test-api-key",
            template="base",
            timeout=120,
            envs=None,
        )
        assert sb._sandbox is mock_instance

    @patch("haystack_integrations.tools.e2b.e2b_sandbox.e2b_import")
    @patch("haystack_integrations.tools.e2b.e2b_sandbox.Sandbox.create")
    def test_warm_up_passes_environment_vars(self, mock_sandbox_create, mock_e2b_import):
        mock_e2b_import.check.return_value = None
        mock_sandbox_create.return_value = _make_sandbox_mock()

        sb = _make_sandbox(environment_vars={"MY_VAR": "value"})
        sb.warm_up()

        _, kwargs = mock_sandbox_create.call_args
        assert kwargs["envs"] == {"MY_VAR": "value"}

    @patch("haystack_integrations.tools.e2b.e2b_sandbox.e2b_import")
    @patch("haystack_integrations.tools.e2b.e2b_sandbox.Sandbox.create")
    def test_warm_up_is_idempotent(self, mock_sandbox_create, mock_e2b_import):
        mock_e2b_import.check.return_value = None
        mock_sandbox_create.return_value = _make_sandbox_mock()

        sb = _make_sandbox()
        sb.warm_up()
        sb.warm_up()

        mock_sandbox_create.assert_called_once()

    @patch("haystack_integrations.tools.e2b.e2b_sandbox.e2b_import")
    @patch("haystack_integrations.tools.e2b.e2b_sandbox.Sandbox.create")
    def test_warm_up_raises_on_sandbox_error(self, mock_sandbox_create, mock_e2b_import):
        mock_e2b_import.check.return_value = None
        mock_sandbox_create.side_effect = Exception("connection refused")

        sb = _make_sandbox()
        with pytest.raises(RuntimeError, match="Failed to start E2B sandbox"):
            sb.warm_up()


# ---------------------------------------------------------------------------
# E2BSandbox -- close
# ---------------------------------------------------------------------------


class TestE2BSandboxClose:
    def test_close_without_warm_up_is_noop(self):
        sb = _make_sandbox()
        sb.close()
        assert sb._sandbox is None

    def test_close_kills_sandbox(self):
        sb, mock = _sandbox_with_mock()
        sb.close()
        mock.kill.assert_called_once()
        assert sb._sandbox is None

    def test_close_clears_sandbox_on_kill_error(self):
        sb, mock = _sandbox_with_mock()
        mock.kill.side_effect = Exception("kill failed")
        sb.close()  # must not raise
        assert sb._sandbox is None


# ---------------------------------------------------------------------------
# E2BSandbox -- serialisation
# ---------------------------------------------------------------------------


class TestE2BSandboxSerialisation:
    def _make_env_sandbox(self, **kwargs) -> E2BSandbox:
        defaults = {"api_key": Secret.from_env_var("E2B_API_KEY")}
        defaults.update(kwargs)
        return E2BSandbox(**defaults)

    def test_to_dict_contains_expected_keys(self):
        sb = self._make_env_sandbox(sandbox_template="my-template", timeout=600)
        data = sb.to_dict()

        assert "type" in data
        assert "data" in data
        assert data["data"]["sandbox_template"] == "my-template"
        assert data["data"]["timeout"] == 600

    def test_to_dict_does_not_include_sandbox_instance(self):
        sb = self._make_env_sandbox()
        sb._sandbox = _make_sandbox_mock()
        data = sb.to_dict()

        assert "_sandbox" not in data["data"]
        assert "sandbox" not in data["data"]

    def test_from_dict_round_trip(self):
        original = self._make_env_sandbox(
            sandbox_template="custom",
            timeout=900,
            environment_vars={"KEY": "value"},
        )
        data = original.to_dict()
        restored = E2BSandbox.from_dict(data)

        assert restored.sandbox_template == "custom"
        assert restored.timeout == 900
        assert restored.environment_vars == {"KEY": "value"}
        assert restored._sandbox is None

    def test_to_dict_type_is_qualified_class_name(self):
        sb = self._make_env_sandbox()
        data = sb.to_dict()
        assert "E2BSandbox" in data["type"]

    def test_to_dict_includes_stable_instance_id(self):
        sb = self._make_env_sandbox()
        data = sb.to_dict()
        assert data["data"]["instance_id"] == sb.instance_id

    def test_from_dict_preserves_instance_id(self):
        original = self._make_env_sandbox()
        restored = E2BSandbox.from_dict(original.to_dict())
        assert restored.instance_id == original.instance_id

    def test_from_dict_dedupes_tools_sharing_one_sandbox(self):
        """Tools that shared one sandbox before serialization share it after round-trip."""
        E2BSandbox._instances.clear()
        sandbox = self._make_env_sandbox(sandbox_template="custom", timeout=300)
        tool_a = RunBashCommandTool(sandbox=sandbox)
        tool_b = ReadFileTool(sandbox=sandbox)

        restored_a = RunBashCommandTool.from_dict(tool_a.to_dict())
        restored_b = ReadFileTool.from_dict(tool_b.to_dict())

        assert restored_a._e2b_sandbox is restored_b._e2b_sandbox
        assert restored_a._e2b_sandbox.instance_id == sandbox.instance_id

    def test_from_dict_distinct_sandboxes_remain_distinct(self):
        """Two separately-built sandboxes with identical config keep distinct identities."""
        E2BSandbox._instances.clear()
        sb1 = self._make_env_sandbox(sandbox_template="base", timeout=120)
        sb2 = self._make_env_sandbox(sandbox_template="base", timeout=120)
        assert sb1.instance_id != sb2.instance_id

        restored_1 = E2BSandbox.from_dict(sb1.to_dict())
        restored_2 = E2BSandbox.from_dict(sb2.to_dict())

        assert restored_1 is not restored_2
        assert restored_1.instance_id == sb1.instance_id
        assert restored_2.instance_id == sb2.instance_id

    def test_from_dict_id_collision_with_mismatched_config_does_not_dedup(self, monkeypatch):
        """A crafted dict reusing another sandbox's id but with a different api_key
        must NOT receive the cached instance (no cross-tenant escalation), and must
        NOT evict the legitimate cache entry (no DoS)."""
        E2BSandbox._instances.clear()
        monkeypatch.setenv("VICTIM_KEY", "victim-secret")
        monkeypatch.setenv("ATTACKER_KEY", "attacker-secret")

        legitimate_data = {
            "type": "haystack_integrations.tools.e2b.e2b_sandbox.E2BSandbox",
            "data": {
                "instance_id": "shared-id",
                "api_key": Secret.from_env_var("VICTIM_KEY").to_dict(),
                "sandbox_template": "base",
                "timeout": 120,
                "environment_vars": {},
            },
        }
        legitimate = E2BSandbox.from_dict(legitimate_data)
        assert E2BSandbox._instances.get("shared-id") is legitimate

        attacker_data = {
            "type": "haystack_integrations.tools.e2b.e2b_sandbox.E2BSandbox",
            "data": {
                "instance_id": "shared-id",
                "api_key": Secret.from_env_var("ATTACKER_KEY").to_dict(),
                "sandbox_template": "base",
                "timeout": 120,
                "environment_vars": {},
            },
        }
        attacker = E2BSandbox.from_dict(attacker_data)

        assert attacker is not legitimate
        assert attacker.api_key.resolve_value() == "attacker-secret"
        assert legitimate.api_key.resolve_value() == "victim-secret"
        # Cache still points at the legitimate instance — attacker did not evict it.
        assert E2BSandbox._instances.get("shared-id") is legitimate


# ---------------------------------------------------------------------------
# Tool classes -- structure
# ---------------------------------------------------------------------------


class TestToolClasses:
    def test_run_bash_command_tool_name_and_schema(self):
        sb = _make_sandbox()
        tool = RunBashCommandTool(sandbox=sb)
        assert tool.name == "run_bash_command"
        assert tool.description
        assert "command" in tool.parameters["required"]

    def test_read_file_tool_name_and_schema(self):
        sb = _make_sandbox()
        tool = ReadFileTool(sandbox=sb)
        assert tool.name == "read_file"
        assert tool.description
        assert "path" in tool.parameters["required"]

    def test_write_file_tool_name_and_schema(self):
        sb = _make_sandbox()
        tool = WriteFileTool(sandbox=sb)
        assert tool.name == "write_file"
        assert tool.description
        assert "path" in tool.parameters["required"]
        assert "content" in tool.parameters["required"]

    def test_list_directory_tool_name_and_schema(self):
        sb = _make_sandbox()
        tool = ListDirectoryTool(sandbox=sb)
        assert tool.name == "list_directory"
        assert tool.description
        assert "path" in tool.parameters["required"]

    def test_tool_stores_sandbox_reference(self):
        sb = _make_sandbox()
        tool = RunBashCommandTool(sandbox=sb)
        assert tool._e2b_sandbox is sb

    def test_e2b_toolset_contains_four_tools(self):
        ts = E2BToolset(api_key=Secret.from_token("test-api-key"))
        assert len(ts) == 4
        names = {t.name for t in ts}
        assert names == {"run_bash_command", "read_file", "write_file", "list_directory"}

    def test_e2b_toolset_has_correct_tool_types(self):
        ts = E2BToolset(api_key=Secret.from_token("test-api-key"))
        tool_types = {type(t) for t in ts}
        assert tool_types == {RunBashCommandTool, ReadFileTool, WriteFileTool, ListDirectoryTool}

    def test_e2b_toolset_shares_same_sandbox(self):
        ts = E2BToolset(api_key=Secret.from_token("test-api-key"))
        assert all(t._e2b_sandbox is ts.sandbox for t in ts)

        mock = _make_sandbox_mock()
        mock.commands.run.return_value = MagicMock(exit_code=0, stdout="ok", stderr="")
        ts.sandbox._sandbox = mock

        bash_tool = next(t for t in ts if t.name == "run_bash_command")
        bash_tool.invoke(command="echo ok")

        mock.commands.run.assert_called_once()

    def test_e2b_toolset_default_api_key(self):
        """E2BToolset uses E2B_API_KEY env var when api_key is omitted."""
        ts = E2BToolset()
        assert ts.sandbox.api_key is not None

    def test_tools_from_same_sandbox_share_state(self):
        """Tools instantiated with the same sandbox share state."""
        sb = _make_sandbox()
        bash_tool = RunBashCommandTool(sandbox=sb)
        read_tool = ReadFileTool(sandbox=sb)
        assert bash_tool._e2b_sandbox is read_tool._e2b_sandbox


# ---------------------------------------------------------------------------
# RunBashCommandTool behaviour
# ---------------------------------------------------------------------------


class TestRunBashCommandTool:
    def test_returns_formatted_output(self):
        sb, mock = _sandbox_with_mock()
        mock_result = MagicMock(exit_code=0, stdout="hello world\n", stderr="")
        mock.commands.run.return_value = mock_result
        tool = RunBashCommandTool(sandbox=sb)

        output = tool.invoke(command="echo hello world")

        assert "exit_code: 0" in output
        assert "hello world" in output
        mock.commands.run.assert_called_once_with("echo hello world", timeout=60)

    def test_passes_custom_timeout(self):
        sb, mock = _sandbox_with_mock()
        mock.commands.run.return_value = MagicMock(exit_code=0, stdout="", stderr="")
        tool = RunBashCommandTool(sandbox=sb)

        tool.invoke(command="sleep 5", timeout=30)

        mock.commands.run.assert_called_once_with("sleep 5", timeout=30)

    @patch("haystack_integrations.tools.e2b.e2b_sandbox.e2b_import")
    @patch("haystack_integrations.tools.e2b.e2b_sandbox.Sandbox.create")
    def test_wraps_warm_up_failure(self, mock_sandbox_create, mock_e2b_import):
        mock_e2b_import.check.return_value = None
        mock_sandbox_create.side_effect = Exception("connection refused")
        sb = _make_sandbox()
        tool = RunBashCommandTool(sandbox=sb)
        with pytest.raises(ToolInvocationError, match="Failed to start E2B sandbox"):
            tool.invoke(command="ls")

    def test_wraps_sandbox_exception(self):
        sb, mock = _sandbox_with_mock()
        mock.commands.run.side_effect = Exception("timeout")
        tool = RunBashCommandTool(sandbox=sb)
        with pytest.raises(ToolInvocationError, match="Failed to run bash command"):
            tool.invoke(command="sleep 1000")


# ---------------------------------------------------------------------------
# ReadFileTool behaviour
# ---------------------------------------------------------------------------


class TestReadFileTool:
    def test_returns_string(self):
        sb, mock = _sandbox_with_mock()
        mock.files.read.return_value = "file content"
        tool = ReadFileTool(sandbox=sb)

        result = tool.invoke(path="/some/file.txt")

        assert result == "file content"
        mock.files.read.assert_called_once_with("/some/file.txt")

    def test_decodes_bytes(self):
        sb, mock = _sandbox_with_mock()
        mock.files.read.return_value = b"binary content"
        tool = ReadFileTool(sandbox=sb)

        result = tool.invoke(path="/binary.bin")

        assert result == "binary content"

    @patch("haystack_integrations.tools.e2b.e2b_sandbox.e2b_import")
    @patch("haystack_integrations.tools.e2b.e2b_sandbox.Sandbox.create")
    def test_wraps_warm_up_failure(self, mock_sandbox_create, mock_e2b_import):
        mock_e2b_import.check.return_value = None
        mock_sandbox_create.side_effect = Exception("connection refused")
        sb = _make_sandbox()
        tool = ReadFileTool(sandbox=sb)
        with pytest.raises(ToolInvocationError, match="Failed to start E2B sandbox"):
            tool.invoke(path="/some/file.txt")

    def test_wraps_sandbox_exception(self):
        sb, mock = _sandbox_with_mock()
        mock.files.read.side_effect = Exception("file not found")
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
        mock.files.write.assert_called_once_with("/output/result.txt", "hello")

    @patch("haystack_integrations.tools.e2b.e2b_sandbox.e2b_import")
    @patch("haystack_integrations.tools.e2b.e2b_sandbox.Sandbox.create")
    def test_wraps_warm_up_failure(self, mock_sandbox_create, mock_e2b_import):
        mock_e2b_import.check.return_value = None
        mock_sandbox_create.side_effect = Exception("connection refused")
        sb = _make_sandbox()
        tool = WriteFileTool(sandbox=sb)
        with pytest.raises(ToolInvocationError, match="Failed to start E2B sandbox"):
            tool.invoke(path="/some/path.txt", content="content")

    def test_wraps_sandbox_exception(self):
        sb, mock = _sandbox_with_mock()
        mock.files.write.side_effect = Exception("permission denied")
        tool = WriteFileTool(sandbox=sb)
        with pytest.raises(ToolInvocationError, match="Failed to write file"):
            tool.invoke(path="/protected/file.txt", content="data")


# ---------------------------------------------------------------------------
# ListDirectoryTool behaviour
# ---------------------------------------------------------------------------


class TestListDirectoryTool:
    def _make_entry(self, name: str, is_dir: bool = False) -> MagicMock:
        entry = MagicMock()
        entry.name = name
        entry.is_dir = is_dir
        return entry

    def test_returns_names(self):
        sb, mock = _sandbox_with_mock()
        mock.files.list.return_value = [
            self._make_entry("file.txt"),
            self._make_entry("subdir", is_dir=True),
        ]
        tool = ListDirectoryTool(sandbox=sb)

        result = tool.invoke(path="/home/user")

        assert "file.txt" in result
        assert "subdir/" in result
        mock.files.list.assert_called_once_with("/home/user")

    def test_empty_directory(self):
        sb, mock = _sandbox_with_mock()
        mock.files.list.return_value = []
        tool = ListDirectoryTool(sandbox=sb)

        result = tool.invoke(path="/empty")

        assert result == "(empty directory)"

    @patch("haystack_integrations.tools.e2b.e2b_sandbox.e2b_import")
    @patch("haystack_integrations.tools.e2b.e2b_sandbox.Sandbox.create")
    def test_wraps_warm_up_failure(self, mock_sandbox_create, mock_e2b_import):
        mock_e2b_import.check.return_value = None
        mock_sandbox_create.side_effect = Exception("connection refused")
        sb = _make_sandbox()
        tool = ListDirectoryTool(sandbox=sb)
        with pytest.raises(ToolInvocationError, match="Failed to start E2B sandbox"):
            tool.invoke(path="/home")

    def test_wraps_sandbox_exception(self):
        sb, mock = _sandbox_with_mock()
        mock.files.list.side_effect = Exception("not a directory")
        tool = ListDirectoryTool(sandbox=sb)
        with pytest.raises(ToolInvocationError, match="Failed to list directory"):
            tool.invoke(path="/nonexistent")
