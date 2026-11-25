"""Windows API helpers to exclude an overlay window from screen capture / screen sharing."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

_user32 = ctypes.WinDLL("user32", use_last_error=True)

_user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
_user32.SetWindowDisplayAffinity.restype = wintypes.BOOL

_user32.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_user32.GetWindowDisplayAffinity.restype = wintypes.BOOL

# Handle 64-bit vs 32-bit Windows GetWindowLongPtrW / SetWindowLongPtrW
if hasattr(_user32, "GetWindowLongPtrW"):
    _GetWindowLong = _user32.GetWindowLongPtrW
    _SetWindowLong = _user32.SetWindowLongPtrW
    _LONG_PTR = ctypes.c_ssize_t
else:
    _GetWindowLong = _user32.GetWindowLongW
    _SetWindowLong = _user32.SetWindowLongW
    _LONG_PTR = ctypes.c_long

_GetWindowLong.argtypes = [wintypes.HWND, ctypes.c_int]
_GetWindowLong.restype = _LONG_PTR

_SetWindowLong.argtypes = [wintypes.HWND, ctypes.c_int, _LONG_PTR]
_SetWindowLong.restype = _LONG_PTR


def apply_capture_exclusion(hwnd: int) -> bool:
    """
    Exclude window from screen capture (Zoom, Meet, Teams, Discord, OBS, Snipping Tool).
    Requires Windows 10 2004+ or Windows 11.
    Returns True if WDA_EXCLUDEFROMCAPTURE was successfully applied.
    """
    ok = bool(_user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), wintypes.DWORD(WDA_EXCLUDEFROMCAPTURE)))
    if not ok:
        # Fallback to WDA_MONITOR (shows black rectangle on older Windows builds)
        ok = bool(_user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), wintypes.DWORD(WDA_MONITOR)))
    return ok


def is_capture_excluded(hwnd: int) -> bool:
    affinity = wintypes.DWORD(0)
    if _user32.GetWindowDisplayAffinity(wintypes.HWND(hwnd), ctypes.byref(affinity)):
        return affinity.value in (WDA_EXCLUDEFROMCAPTURE, WDA_MONITOR)
    return False


def set_click_through(hwnd: int, enabled: bool) -> None:
    """Toggle mouse click-through via WS_EX_TRANSPARENT."""
    style = _GetWindowLong(wintypes.HWND(hwnd), GWL_EXSTYLE)
    if enabled:
        style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
    else:
        style &= ~WS_EX_TRANSPARENT
    _SetWindowLong(wintypes.HWND(hwnd), GWL_EXSTYLE, style)
