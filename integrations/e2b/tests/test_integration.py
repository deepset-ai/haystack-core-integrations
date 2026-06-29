# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for the E2B sandbox tools.

These tests require a valid E2B_API_KEY environment variable and will
spin up a real cloud sandbox on each run.
"""

import pytest

from haystack_integrations.tools.e2b import (
    E2BSandbox,
    E2BToolset,
    ListDirectoryTool,
    ReadFileTool,
    RunBashCommandTool,
    WriteFileTool,
)


@pytest.fixture(scope="module")
def sandbox():
    """Shared sandbox for the module — spun up once, torn down after all tests."""
    sb = E2BSandbox()
    sb.warm_up()
    yield sb
    sb.close()


@pytest.mark.integration
class TestRunBashCommandToolIntegration:
    def test_echo_command(self, sandbox):
        tool = RunBashCommandTool(sandbox=sandbox)
        result = tool.invoke(command="echo 'hello from e2b'")
        assert "hello from e2b" in result
        assert "exit_code: 0" in result

    def test_exit_code_nonzero(self, sandbox):
        tool = RunBashCommandTool(sandbox=sandbox)
        result = tool.invoke(command="exit 42")
        assert "exit_code: 42" in result

    def test_stderr_captured(self, sandbox):
        tool = RunBashCommandTool(sandbox=sandbox)
        result = tool.invoke(command="echo error_msg >&2")
        assert "error_msg" in result


@pytest.mark.integration
class TestWriteAndReadFileToolIntegration:
    def test_write_then_read(self, sandbox):
        write_tool = WriteFileTool(sandbox=sandbox)
        read_tool = ReadFileTool(sandbox=sandbox)

        write_result = write_tool.invoke(path="/tmp/test_haystack.txt", content="haystack e2b integration")
        assert "/tmp/test_haystack.txt" in write_result

        read_result = read_tool.invoke(path="/tmp/test_haystack.txt")
        assert read_result == "haystack e2b integration"

    def test_write_creates_parent_dirs(self, sandbox):
        write_tool = WriteFileTool(sandbox=sandbox)
        read_tool = ReadFileTool(sandbox=sandbox)

        write_tool.invoke(path="/tmp/e2b_test_dir/nested/file.txt", content="nested content")
        result = read_tool.invoke(path="/tmp/e2b_test_dir/nested/file.txt")
        assert result == "nested content"


@pytest.mark.integration
class TestListDirectoryToolIntegration:
    def test_list_tmp(self, sandbox):
        tool = ListDirectoryTool(sandbox=sandbox)
        result = tool.invoke(path="/tmp")
        # /tmp always exists and is listable; result is a newline-separated string or "(empty directory)"
        assert isinstance(result, str)

    def test_lists_written_file(self, sandbox):
        write_tool = WriteFileTool(sandbox=sandbox)
        list_tool = ListDirectoryTool(sandbox=sandbox)

        write_tool.invoke(path="/tmp/e2b_list_test/myfile.txt", content="data")
        result = list_tool.invoke(path="/tmp/e2b_list_test")
        assert "myfile.txt" in result


@pytest.mark.integration
class TestE2BToolsetIntegration:
    def test_toolset_warm_up_and_close(self):
        ts = E2BToolset()
        ts.warm_up()
        # Verify sandbox is live by running a command through the bash tool
        bash_tool = next(t for t in ts if t.name == "run_bash_command")
        result = bash_tool.invoke(command="echo 'toolset ok'")
        assert "toolset ok" in result
        ts.close()

    def test_all_tools_share_sandbox(self):
        ts = E2BToolset()
        ts.warm_up()

        write_tool = next(t for t in ts if t.name == "write_file")
        read_tool = next(t for t in ts if t.name == "read_file")
        bash_tool = next(t for t in ts if t.name == "run_bash_command")

        # Write via write_file, read back via bash — proves shared sandbox
        write_tool.invoke(path="/tmp/shared_test.txt", content="shared sandbox state")
        bash_result = bash_tool.invoke(command="cat /tmp/shared_test.txt")
        assert "shared sandbox state" in bash_result

        read_result = read_tool.invoke(path="/tmp/shared_test.txt")
        assert read_result == "shared sandbox state"

        ts.close()
