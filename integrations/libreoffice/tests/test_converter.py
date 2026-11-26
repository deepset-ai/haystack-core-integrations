# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.libreoffice import LibreOfficeFileConverter
from haystack_integrations.components.converters.libreoffice.converter import (
    _conversion_subprocess_kwargs,
    _kill_process_group,
    _reap_conversion_process,
)

TYPE = "haystack_integrations.components.converters.libreoffice.converter.LibreOfficeFileConverter"
CONVERTER_MODULE = "haystack_integrations.components.converters.libreoffice.converter"


def _converted_to(args: list[str]) -> str:
    """The output file type the converter asked soffice for."""
    return args[args.index("--convert-to") + 1]


async def _convert(converter: LibreOfficeFileConverter, is_async: bool, *args: Any, **kwargs: Any) -> Any:
    """Call `run_async` or `run`, so one test body can cover both."""
    return await converter.run_async(*args, **kwargs) if is_async else converter.run(*args, **kwargs)


def _configured_converter(output_file_type: str) -> LibreOfficeFileConverter:
    """A converter with an `output_file_type` set at init time, needing no LibreOffice installation."""
    with patch("shutil.which", return_value="/usr/bin/soffice"):
        return LibreOfficeFileConverter(output_file_type=output_file_type)


class TestInit:
    def test_init(self, mock_converter: LibreOfficeFileConverter) -> None:
        assert isinstance(mock_converter.soffice_path, str)
        assert mock_converter.output_file_type is None
        assert _configured_converter("pdf").output_file_type == "pdf"

    def test_raises_when_soffice_is_not_installed(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="LibreOffice"):
                LibreOfficeFileConverter()


class TestSerde:
    def test_to_dict(self, mock_converter: LibreOfficeFileConverter) -> None:
        assert mock_converter.to_dict() == {"type": TYPE, "init_parameters": {"output_file_type": None}}
        assert _configured_converter("pdf").to_dict() == {"type": TYPE, "init_parameters": {"output_file_type": "pdf"}}

    def test_from_dict(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeFileConverter.from_dict({"type": TYPE, "init_parameters": {}})
        assert isinstance(converter.soffice_path, str)
        # Dictionaries serialized before `output_file_type` was included still deserialize.
        assert converter.output_file_type is None

    def test_round_trip_keeps_converting_to_the_configured_type(
        self, fake_soffice: SimpleNamespace, tmp_path: Path
    ) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            deserialized = LibreOfficeFileConverter.from_dict(_configured_converter("pdf").to_dict())

        # The converter that came back out of the round trip still needs no `output_file_type` argument.
        assert deserialized.run([source])["output"][0].mime_type == "application/pdf"
        assert _converted_to(fake_soffice.calls[-1]) == "pdf"


class TestGetConversionArgs:
    def test_builds_the_soffice_command_line(self, mock_converter: LibreOfficeFileConverter, tmp_path: Path) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        output_path, args = mock_converter._get_conversion_args(source, tmp_path, "pdf")

        assert args == ["/usr/bin/soffice", "--headless", "--convert-to", "pdf", "--outdir", str(tmp_path), str(source)]
        assert output_path == tmp_path / "sample.pdf"

    def test_rejects_an_output_directory_that_does_not_exist(
        self, mock_converter: LibreOfficeFileConverter, tmp_path: Path
    ) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        with pytest.raises(OSError, match="must exist and be writable"):
            mock_converter._get_conversion_args(source, tmp_path / "missing", "pdf")


class TestValidateArgs:
    @pytest.mark.parametrize(
        ("output_file_type", "error"),
        [
            # .pdf is not a supported input type in SUPPORTED_TYPES
            ("docx", "input_file_type"),
            # .doc -> .png is not a valid conversion
            ("png", "is not supported for"),
            # .xyz is not one of the OUTPUT_FILE_TYPE values at all
            ("xyz", "is not supported and must be one of type"),
        ],
    )
    def test_rejects_unsupported_conversions(
        self, mock_converter: LibreOfficeFileConverter, output_file_type: str, error: str
    ) -> None:
        source = "test_file.pdf" if output_file_type == "docx" else "test_file.doc"
        with pytest.raises(ValueError, match=error):
            mock_converter.run([source], output_file_type=output_file_type)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["run", "run_async"])
class TestConvert:
    """`run` and `run_async` share their logic, so every case here covers both."""

    async def test_converts_each_source_in_order(
        self, mock_converter: LibreOfficeFileConverter, fake_soffice: SimpleNamespace, tmp_path: Path, is_async: bool
    ) -> None:
        first, second = tmp_path / "first.doc", tmp_path / "second.ppt"
        first.write_bytes(b"doc")
        second.write_bytes(b"ppt")

        result = await _convert(mock_converter, is_async, [first, second], output_file_type="pdf")

        assert [stream.data for stream in result["output"]] == [b"converted bytes"] * 2
        assert [stream.mime_type for stream in result["output"]] == ["application/pdf"] * 2
        assert len(fake_soffice.calls) == 2

    async def test_writes_a_bytestream_to_a_temporary_file_before_converting(
        self, mock_converter: LibreOfficeFileConverter, fake_soffice: SimpleNamespace, is_async: bool
    ) -> None:
        result = await _convert(mock_converter, is_async, [ByteStream(data=b"raw doc bytes")], output_file_type="pdf")

        assert [stream.data for stream in result["output"]] == [b"converted bytes"]
        # The source handed to soffice is the temporary file, not the original path.
        assert Path(fake_soffice.calls[-1][-1]).name == "input"

    async def test_rejects_a_source_that_does_not_exist(
        self, mock_converter: LibreOfficeFileConverter, is_async: bool
    ) -> None:
        with pytest.raises(FileNotFoundError):
            await _convert(mock_converter, is_async, ["nonexistent_file.doc"], output_file_type="docx")

    async def test_uses_the_output_file_type_from_init(
        self, fake_soffice: SimpleNamespace, tmp_path: Path, is_async: bool
    ) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        result = await _convert(_configured_converter("pdf"), is_async, [source])

        assert result["output"][0].mime_type == "application/pdf"
        assert _converted_to(fake_soffice.calls[-1]) == "pdf"

    async def test_argument_overrides_the_output_file_type_from_init(
        self, fake_soffice: SimpleNamespace, tmp_path: Path, is_async: bool
    ) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        await _convert(_configured_converter("docx"), is_async, [source], output_file_type="pdf")

        assert _converted_to(fake_soffice.calls[-1]) == "pdf"

    async def test_requires_an_output_file_type_somewhere(
        self, mock_converter: LibreOfficeFileConverter, is_async: bool
    ) -> None:
        with pytest.raises(ValueError, match="output_file_type must be provided"):
            await _convert(mock_converter, is_async, ["sample.doc"])

    @pytest.mark.parametrize("as_bytestream", [False, True])
    async def test_raises_when_soffice_exits_non_zero(
        self,
        mock_converter: LibreOfficeFileConverter,
        fake_soffice: SimpleNamespace,
        tmp_path: Path,
        is_async: bool,
        as_bytestream: bool,
    ) -> None:
        path = tmp_path / "sample.doc"
        path.write_bytes(b"doc")
        fake_soffice.returncode = 1
        source = ByteStream(data=b"raw doc bytes") if as_bytestream else path

        with pytest.raises(subprocess.CalledProcessError):
            await _convert(mock_converter, is_async, [source], output_file_type="pdf")


class TestAsyncCancellation:
    """Cancelling `run_async` must reap the soffice child before its workspace disappears."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("as_bytestream", [False, True], ids=["path", "bytestream"])
    @pytest.mark.parametrize("cancel_again", [False, True], ids=["once", "again"])
    async def test_cancelling_reaps_the_child_before_removing_its_workspace(
        self,
        mock_converter: LibreOfficeFileConverter,
        tmp_path: Path,
        as_bytestream: bool,
        cancel_again: bool,
    ) -> None:
        # A real interpreter process stands in for soffice and blocks until it is killed.
        # No LibreOffice binary is involved.
        started = asyncio.Event()
        children: list[asyncio.subprocess.Process] = []
        tasks: list[asyncio.Task[Any]] = []
        reaped_before_cleanup: list[bool] = []
        workspaces: list[str] = []

        async def launch(*_args: str, **_kwargs: object) -> asyncio.subprocess.Process:
            # Same session/group setup as production, so the group kill reaches this child.
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "import time; time.sleep(3600)",
                **_conversion_subprocess_kwargs(),
            )
            children.append(child)
            started.set()
            return child

        class _Workspace(TemporaryDirectory):
            def __enter__(self) -> str:
                path = super().__enter__()
                workspaces.append(path)
                return path

            def __exit__(self, *args: object) -> bool | None:
                # Cleanup is the moment the workspace goes away; the child must already be gone.
                reaped_before_cleanup.append(bool(children) and children[0].returncode is not None)
                return super().__exit__(*args)

        path = tmp_path / "sample.doc"
        path.write_bytes(b"doc")
        source: Path | ByteStream = ByteStream(data=b"raw doc bytes") if as_bytestream else path

        def kill_and_cancel(process: asyncio.subprocess.Process) -> None:
            _kill_process_group(process)
            # Second cancellation arrives while the coroutine is reaping the leader,
            # which is the case that drops an unshielded `process.wait()`.
            tasks[0].cancel()

        patches = [
            patch(f"{CONVERTER_MODULE}.create_subprocess_exec", side_effect=launch),
            patch(f"{CONVERTER_MODULE}.TemporaryDirectory", _Workspace),
        ]
        if cancel_again:
            patches.append(patch(f"{CONVERTER_MODULE}._kill_process_group", side_effect=kill_and_cancel))
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            task = asyncio.create_task(mock_converter.run_async([source], output_file_type="pdf"))
            tasks.append(task)
            await started.wait()
            task.cancel()
            try:
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert reaped_before_cleanup == [True]
                assert children[0].returncode is not None
                assert workspaces and not Path(workspaces[0]).exists()
            finally:
                # If the converter leaked the child, don't leave the sleeper behind.
                if children and children[0].returncode is None:
                    _stop_process_tree(children[0])
                    await children[0].wait()

    @pytest.mark.asyncio
    async def test_reap_leaves_a_child_that_has_already_exited(self) -> None:
        # Cancellation can arrive after the child has already been reaped. That path
        # must return without waiting again.
        child = await asyncio.create_subprocess_exec(sys.executable, "-c", "import sys; sys.exit(0)")
        await child.wait()
        assert child.returncode == 0

        async def wait() -> int:
            msg = "already exited"
            raise AssertionError(msg)

        child.wait = wait  # type: ignore[method-assign]
        await _reap_conversion_process(child)
        assert child.returncode == 0

    @pytest.mark.asyncio
    async def test_cancelling_kills_a_grandchild_started_by_the_launcher(
        self, mock_converter: LibreOfficeFileConverter, tmp_path: Path
    ) -> None:
        # soffice is a launcher: the direct child starts soffice.bin and waits.
        # Killing only the launcher used to leave that grandchild running.
        pidfile = tmp_path / "grandchild.pid"
        children: list[asyncio.subprocess.Process] = []
        path = tmp_path / "sample.doc"
        path.write_bytes(b"doc")

        async def launch(*_args: str, **kwargs: Any) -> asyncio.subprocess.Process:
            child = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _LAUNCHER,
                str(pidfile),
                **kwargs,
            )
            children.append(child)
            return child

        with patch(f"{CONVERTER_MODULE}.create_subprocess_exec", side_effect=launch):
            task = asyncio.create_task(mock_converter.run_async([path], output_file_type="pdf"))
            try:
                grandchild_pid = await _wait_for_pid(pidfile)
                task.cancel()
                done, _pending = await asyncio.wait({task}, timeout=5)
                if task not in done:
                    pytest.fail("cancelled conversion did not finish")
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert children[0].returncode is not None
                _wait_until_stopped(grandchild_pid)
            finally:
                if children and children[0].returncode is None:
                    _stop_process_tree(children[0])
                    await children[0].wait()
                text = pidfile.read_text(encoding="utf-8").strip() if pidfile.exists() else ""
                if text and _pid_is_running(int(text)):
                    _stop_pid(int(text))


def _stop_pid(pid: int) -> None:
    """Best-effort kill for a grandchild whose leader is already gone."""
    if sys.platform == "win32":
        subprocess.run(  # noqa: S603
            ["taskkill", "/F", "/PID", str(pid)],  # noqa: S607
            check=False,
            capture_output=True,
        )
        return
    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def _stop_process_tree(process: asyncio.subprocess.Process) -> None:
    """Stop a leaked leader and anything it started, matching the production kill."""
    if sys.platform == "win32":
        subprocess.run(  # noqa: S603
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],  # noqa: S607
            check=False,
            capture_output=True,
        )
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


def _pid_is_running(pid: int) -> bool:
    """True when `pid` still exists. A zombie counts until init reaps it; callers poll."""
    if sys.platform == "win32":
        result = subprocess.run(  # noqa: S603
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
        )
        return any(part == str(pid) for line in result.stdout.splitlines() for part in line.split())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _wait_for_pid(pidfile: Path) -> int:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if pidfile.exists():
            text = pidfile.read_text(encoding="utf-8").strip()
            if text:
                return int(text)
        await asyncio.sleep(0.02)
    msg = "launcher did not record the grandchild pid"
    raise AssertionError(msg)


def _wait_until_stopped(pid: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return
        time.sleep(0.02)
    msg = f"process {pid} was still running"
    raise AssertionError(msg)


# Stands in for the soffice launcher: start a long-running grandchild, record its pid, and wait.
_LAUNCHER = """
import subprocess
import sys

grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    handle.write(str(grandchild.pid))
grandchild.wait()
"""


class TestResolveMimeType:
    @pytest.mark.parametrize(
        ("guessed", "output_file_type", "expected"),
        [
            ("application/pdf", "pdf", "application/pdf"),
            (None, "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            (None, "unknown", None),
        ],
    )
    def test_falls_back_to_the_mapping_when_the_guess_fails(
        self,
        mock_converter: LibreOfficeFileConverter,
        guessed: str | None,
        output_file_type: str,
        expected: str | None,
    ) -> None:
        with patch("mimetypes.guess_type", return_value=(guessed, None)):
            assert mock_converter._resolve_mime_type(Path(f"test.{output_file_type}"), output_file_type) == expected


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["run", "run_async"])
class TestRunIntegration:
    """End-to-end against a real LibreOffice installation, for both `run` and `run_async`."""

    @pytest.mark.parametrize("as_bytestream", [False, True])
    async def test_converts_office_files_to_pdf(
        self,
        real_converter: LibreOfficeFileConverter,
        test_files_path: Path,
        is_async: bool,
        as_bytestream: bool,
    ) -> None:
        paths = [
            test_files_path / "doc" / "sample_doc.doc",
            test_files_path / "ppt" / "sample_ppt.ppt",
            test_files_path / "xls" / "basic_tables_two_sheets.xls",
        ]
        sources = [ByteStream(data=paths[0].read_bytes())] if as_bytestream else paths

        result = await _convert(real_converter, is_async, sources, output_file_type="pdf")

        output = result["output"]
        assert len(output) == len(sources)
        for stream in output:
            assert isinstance(stream, ByteStream)
            assert len(stream.data) > 0
            assert stream.mime_type == "application/pdf"
