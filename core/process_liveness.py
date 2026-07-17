"""Cross-platform, non-destructive process liveness checks."""

from __future__ import annotations

import os
from typing import Any, Callable


def is_process_alive(process_id: int) -> bool:
    """Return whether ``process_id`` identifies a currently running process."""
    if process_id <= 0:
        return False
    if process_id == os.getpid():
        return True
    if os.name == "nt":
        return _windows_process_is_alive(process_id)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_process_is_alive(process_id: int) -> bool:
    """Query a Windows process handle without sending it a signal."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    get_exit_code_process = kernel32.GetExitCodeProcess
    get_exit_code_process.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    get_exit_code_process.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    return _windows_process_is_alive_with_api(
        process_id,
        open_process=open_process,
        get_exit_code_process=get_exit_code_process,
        close_handle=close_handle,
        get_last_error=ctypes.get_last_error,
        exit_code_factory=wintypes.DWORD,
        exit_code_reference=ctypes.byref,
    )


def _windows_process_is_alive_with_api(
    process_id: int,
    *,
    open_process: Callable[[int, bool, int], Any],
    get_exit_code_process: Callable[[Any, Any], bool],
    close_handle: Callable[[Any], Any],
    get_last_error: Callable[[], int],
    exit_code_factory: Callable[[], Any],
    exit_code_reference: Callable[[Any], Any],
) -> bool:
    """Run the Windows liveness decision against an injectable Win32 API."""
    process_query_limited_information = 0x1000
    error_access_denied = 5
    still_active = 259
    handle = open_process(process_query_limited_information, False, process_id)
    if not handle:
        return get_last_error() == error_access_denied
    try:
        exit_code = exit_code_factory()
        if not get_exit_code_process(handle, exit_code_reference(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        close_handle(handle)
