# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.tools import Tool

from haystack_integrations.tools.mirage import (
    MirageCommandNotAllowedError,
    MirageMount,
    MirageShellTool,
    MirageWorkspace,
)


def _ws():
    return MirageWorkspace(mounts=[MirageMount(path="/data", resource="ram")])


class TestToolShape:
    def test_is_a_haystack_tool(self):
        tool = MirageShellTool(_ws())
        assert isinstance(tool, Tool)
        assert tool.name == "mirage_shell"

    def test_description_mentions_mounts(self):
        tool = MirageShellTool(_ws())
        assert "/data" in tool.description

    def test_command_parameter_is_required(self):
        tool = MirageShellTool(_ws())
        assert tool.parameters["required"] == ["command"]
        assert tool.parameters["properties"]["command"]["type"] == "string"


class TestGuard:
    def test_allowlist_blocks_unlisted_command(self):
        tool = MirageShellTool(_ws(), allowed_commands=["cat", "ls"])
        with pytest.raises(MirageCommandNotAllowedError):
            tool._invoke("rm -rf /data")

    def test_allowlist_checks_every_pipeline_segment(self):
        tool = MirageShellTool(_ws(), allowed_commands=["cat"])
        # 'wc' is not allowed, even though it is in the second pipe segment
        with pytest.raises(MirageCommandNotAllowedError):
            tool._invoke("cat /data/a.txt | wc -l")

    def test_allowlist_permits_listed_commands(self):
        tool = MirageShellTool(_ws(), allowed_commands=["echo", "cat"])
        tool._invoke('echo "hi" > /data/a.txt')
        assert "hi" in tool._invoke("cat /data/a.txt")
        tool.close()

    def test_denied_path_blocks_command(self):
        tool = MirageShellTool(_ws(), denied_paths=["/data/secret"])
        with pytest.raises(MirageCommandNotAllowedError):
            tool._invoke("cat /data/secret/keys.txt")


class TestAllowlistNestedCommands:
    """The allowlist must catch commands Mirage would run inside substitutions/subshells, not just
    the leading word of each segment. Mirage executes `$(...)`, backticks, `<(...)` and subshells, so
    a leading-word-only guard would let a disallowed command run inside an allowed wrapper."""

    @pytest.mark.parametrize(
        "command",
        [
            'ls "$(rm /data/x)"',  # command substitution
            "ls `rm /data/x`",  # backticks
            "cat < <(rm /data/x)",  # process substitution
            "echo hi && rm /data/x",  # && list
            "cat /data/a | rm /data/x",  # pipeline segment
            "for f in $(rm /data/x); do echo $f; done",  # substitution inside control flow
            "( rm /data/x )",  # subshell
        ],
    )
    def test_disallowed_command_nested_in_wrapper_is_rejected(self, command):
        tool = MirageShellTool(_ws(), allowed_commands=["ls", "cat", "echo", "for"])
        with pytest.raises(MirageCommandNotAllowedError):
            tool._guard(command)

    def test_allowed_command_nested_in_substitution_is_permitted(self):
        # `cat` is allowed, so `ls "$(cat ...)"` must pass the guard (both names are allowlisted).
        tool = MirageShellTool(_ws(), allowed_commands=["ls", "cat"])
        tool._guard('ls "$(cat /data/a.txt)"')  # does not raise

    def test_quoted_command_name_fails_closed(self):
        # A quoted command name (`"rm"`) resolves to `rm` at runtime but is not statically resolvable
        # to a bare allowlist entry, so the guard rejects it rather than trusting it.
        tool = MirageShellTool(_ws(), allowed_commands=["rm"])
        with pytest.raises(MirageCommandNotAllowedError):
            tool._guard('"rm" /data/x')

    def test_variable_command_name_fails_closed(self):
        tool = MirageShellTool(_ws(), allowed_commands=["ls"])
        with pytest.raises(MirageCommandNotAllowedError):
            tool._guard("$CMD /data/x")


class TestInvocation:
    def test_invoke_runs_command(self):
        tool = MirageShellTool(_ws())
        tool._invoke('echo "tool works" > /data/a.txt')
        assert "tool works" in tool._invoke("cat /data/a.txt")
        tool.close()

    def test_output_is_truncated(self):
        tool = MirageShellTool(_ws(), max_output_chars=20)
        out = tool._invoke("echo " + "y" * 200)
        assert out.endswith("characters]")
        assert len(out) <= 20 + len("\n... [output truncated to 20 characters]")
        tool.close()


class TestToolSerialization:
    def test_to_dict_shape(self):
        tool = MirageShellTool(_ws(), allowed_commands=["cat"], max_output_chars=123)
        data = tool.to_dict()
        assert data["type"].endswith("MirageShellTool")
        assert data["data"]["max_output_chars"] == 123
        assert data["data"]["allowed_commands"] == ["cat"]

    def test_roundtrip(self):
        tool = MirageShellTool(_ws(), name="fs", allowed_commands=["ls", "cat", "echo"], invocation_timeout=12.0)
        restored = MirageShellTool.from_dict(tool.to_dict())
        assert isinstance(restored, MirageShellTool)
        assert restored.name == "fs"
        assert restored._allowed_commands == ["ls", "cat", "echo"]
        assert restored._invocation_timeout == 12.0
        # the restored tool still works end-to-end
        restored._invoke('echo "rt" > /data/a.txt')
        assert "rt" in restored._invoke("cat /data/a.txt")
        restored.close()
