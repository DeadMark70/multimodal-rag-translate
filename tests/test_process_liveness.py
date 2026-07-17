"""Cross-platform process-liveness regression tests."""

import os
import subprocess
import sys
from types import SimpleNamespace

import core.process_liveness as process_liveness
from core.process_liveness import is_process_alive


def test_live_child_process_remains_alive_after_liveness_check() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert is_process_alive(child.pid) is True
        assert child.poll() is None
    finally:
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def test_current_process_is_alive_without_signalling_it() -> None:
    assert is_process_alive(os.getpid()) is True


def test_terminated_process_is_not_alive() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=5)

    assert is_process_alive(child.pid) is False


def test_non_positive_process_ids_are_not_alive() -> None:
    assert is_process_alive(0) is False
    assert is_process_alive(-1) is False


def test_windows_access_denied_means_process_is_live() -> None:
    closed: list[object] = []

    assert (
        process_liveness._windows_process_is_alive_with_api(
            123,
            open_process=lambda *_args: None,
            get_exit_code_process=lambda *_args: False,
            close_handle=closed.append,
            get_last_error=lambda: 5,
            exit_code_factory=lambda: SimpleNamespace(value=0),
            exit_code_reference=lambda code: code,
        )
        is True
    )
    assert closed == []


def test_windows_exit_code_query_failure_means_process_is_not_live() -> None:
    closed: list[object] = []
    handle = object()

    assert (
        process_liveness._windows_process_is_alive_with_api(
            123,
            open_process=lambda *_args: handle,
            get_exit_code_process=lambda *_args: False,
            close_handle=closed.append,
            get_last_error=lambda: 0,
            exit_code_factory=lambda: SimpleNamespace(value=0),
            exit_code_reference=lambda code: code,
        )
        is False
    )
    assert closed == [handle]


def test_windows_acquired_handle_is_closed_after_live_query() -> None:
    closed: list[object] = []
    handle = object()

    def report_still_active(_handle: object, exit_code: SimpleNamespace) -> bool:
        exit_code.value = 259
        return True

    assert (
        process_liveness._windows_process_is_alive_with_api(
            123,
            open_process=lambda *_args: handle,
            get_exit_code_process=report_still_active,
            close_handle=closed.append,
            get_last_error=lambda: 0,
            exit_code_factory=lambda: SimpleNamespace(value=0),
            exit_code_reference=lambda code: code,
        )
        is True
    )
    assert closed == [handle]
