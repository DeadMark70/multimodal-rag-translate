"""Cross-platform process-liveness regression tests."""

import os
import subprocess
import sys

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
