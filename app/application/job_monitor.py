"""Read-only worker activity for the live progress console."""
import os
import subprocess


def parse_cpu_time(value: str) -> float:
    days, separator, clock = value.strip().rpartition('-')
    parts = [float(part) for part in (clock if separator else value).split(':')]
    seconds = sum(part * 60**index for index, part in enumerate(reversed(parts)))
    return seconds + (int(days) * 86400 if separator else 0)


def _windows_activity(pid: int) -> dict:
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, *(ctypes.POINTER(wintypes.FILETIME),) * 4)
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        # Access denied does not prove the process has exited.
        return {'alive': ctypes.get_last_error() == 5, 'cpu_seconds': None}
    try:
        code = wintypes.DWORD()
        alive = bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        cpu = None
        if kernel.GetProcessTimes(handle, *(ctypes.byref(item) for item in (created, exited, system, user))):
            cpu = sum((item.dwHighDateTime << 32) + item.dwLowDateTime for item in (system, user)) / 1e7
        return {'alive': alive, 'cpu_seconds': cpu}
    finally:
        kernel.CloseHandle(handle)


def process_activity(pid: int) -> dict:
    if os.name == 'nt':
        return _windows_activity(pid)
    try:
        result = subprocess.run(['ps', '-p', str(int(pid)), '-o', 'stat=', '-o', 'time='],
            capture_output=True, text=True, timeout=1., check=False)
        if result.returncode == 1 and not result.stdout.strip():
            return {'alive': False, 'cpu_seconds': None}
        if result.returncode == 0 and result.stdout.strip():
            status, clock = result.stdout.strip().split(maxsplit=1)
            return {'alive': not status.startswith(('Z', 'X')), 'cpu_seconds': parse_cpu_time(clock)}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    try:
        os.kill(pid, 0)
        return {'alive': True, 'cpu_seconds': None}
    except PermissionError:
        return {'alive': True, 'cpu_seconds': None}
    except ProcessLookupError:
        return {'alive': False, 'cpu_seconds': None}
