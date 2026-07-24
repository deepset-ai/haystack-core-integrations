# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for the Tenki sandbox tools.

These tests require a valid TENKI_AUTH_TOKEN (or TENKI_API_KEY) environment
variable and will spin up a real cloud sandbox on each run.
"""

import pytest

from haystack_integrations.tools.tenki import (
    ListDirectoryTool,
    ReadFileTool,
    RunBashCommandTool,
    TenkiSandbox,
    TenkiToolset,
    WriteFileTool,
)


@pytest.fixture(scope="module")
def sandbox():
    """Shared sandbox for the module — spun up once, torn down after all tests."""
    sb = TenkiSandbox()
    sb.warm_up()
    yield sb
    sb.close()


@pytest.mark.integration
class TestRunBashCommandToolIntegration:
    def test_echo_command(self, sandbox):
        tool = RunBashCommandTool(sandbox=sandbox)
        result = tool.invoke(command="echo 'hello from tenki'")
        assert "hello from tenki" in result
        assert "ok: True" in result
        assert "exit_code: 0" in result

    def test_exit_code_nonzero(self, sandbox):
        tool = RunBashCommandTool(sandbox=sandbox)
        result = tool.invoke(command="exit 42")
        assert "exit_code: 42" in result
        assert "ok: False" in result

    def test_stderr_captured(self, sandbox):
        tool = RunBashCommandTool(sandbox=sandbox)
        result = tool.invoke(command="echo error_msg >&2")
        assert "error_msg" in result


@pytest.mark.integration
class TestWriteAndReadFileToolIntegration:
    def test_write_then_read(self, sandbox):
        write_tool = WriteFileTool(sandbox=sandbox)
        read_tool = ReadFileTool(sandbox=sandbox)

        # Paths are relative to the sandbox session working directory. Tenki's fs
        # API is confined to the workdir, so absolute paths like /tmp are rejected.
        write_result = write_tool.invoke(path="test_haystack.txt", content="haystack tenki integration")
        assert "test_haystack.txt" in write_result

        read_result = read_tool.invoke(path="test_haystack.txt")
        assert read_result == "haystack tenki integration"


@pytest.mark.integration
class TestListDirectoryToolIntegration:
    def test_list_workdir(self, sandbox):
        tool = ListDirectoryTool(sandbox=sandbox)
        result = tool.invoke(path=".")
        # The session workdir is always listable; result is a newline-separated string
        # or "(empty directory)".
        assert isinstance(result, str)

    def test_lists_written_file(self, sandbox):
        write_tool = WriteFileTool(sandbox=sandbox)
        list_tool = ListDirectoryTool(sandbox=sandbox)

        write_tool.invoke(path="list_probe.txt", content="data")
        result = list_tool.invoke(path=".")
        assert "list_probe.txt" in result


@pytest.mark.integration
class TestTenkiToolsetIntegration:
    def test_toolset_warm_up_and_close(self):
        ts = TenkiToolset()
        ts.warm_up()
        # Verify sandbox is live by running a command through the bash tool
        bash_tool = next(t for t in ts if t.name == "run_bash_command")
        result = bash_tool.invoke(command="echo 'toolset ok'")
        assert "toolset ok" in result
        ts.close()

    def test_all_tools_share_sandbox(self):
        ts = TenkiToolset()
        ts.warm_up()

        write_tool = next(t for t in ts if t.name == "write_file")
        read_tool = next(t for t in ts if t.name == "read_file")
        bash_tool = next(t for t in ts if t.name == "run_bash_command")

        # Write via write_file, read back via bash — proves shared sandbox.
        # Relative path: bash runs in the same session workdir the fs API is scoped to.
        write_tool.invoke(path="shared_test.txt", content="shared sandbox state")
        bash_result = bash_tool.invoke(command="cat shared_test.txt")
        assert "shared sandbox state" in bash_result

        read_result = read_tool.invoke(path="shared_test.txt")
        assert read_result == "shared sandbox state"

        ts.close()
