"""Fast in-memory screen / active-window capture (excludes WDA_EXCLUDEFROMCAPTURE overlay)."""

from __future__ import annotations

import ctypes
import io
from ctypes import wintypes

import mss
from PIL import Image, ImageGrab

MAX_IMAGE_WIDTH = 1920
JPEG_QUALITY = 82

DWMWA_EXTENDED_FRAME_BOUNDS = 9
GW_HWNDNEXT = 2

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)


def _get_cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT()
    if _user32.GetCursorPos(ctypes.byref(pt)):
        return int(pt.x), int(pt.y)
    return 0, 0


def _get_window_title(hwnd: int) -> str:
    length = _user32.GetWindowTextLengthW(wintypes.HWND(hwnd))
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(wintypes.HWND(hwnd), buf, length + 1)
    return buf.value.strip()


def _get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    hr = _dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd),
        wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
        ctypes.byref(rect),
        ctypes.sizeof(rect),
    )
    if hr != 0:
        if not _user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return None
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w < 50 or h < 50:
        return None
    return (int(rect.left), int(rect.top), int(w), int(h))


def _find_target_window(exclude_hwnd: int = 0) -> tuple[tuple[int, int, int, int], str] | None:
    """Find foreground window rect & title, skipping our own overlay if it has focus."""
    hwnd = int(_user32.GetForegroundWindow())
    visited = 0
    while hwnd and visited < 30:
        visited += 1
        if hwnd != exclude_hwnd and _user32.IsWindowVisible(wintypes.HWND(hwnd)):
            title = _get_window_title(hwnd)
            if title and title not in ("Helper", "Program Manager"):
                rect = _get_window_rect(hwnd)
                if rect is not None:
                    return rect, title
        hwnd = int(_user32.GetWindow(wintypes.HWND(hwnd), GW_HWNDNEXT))
    return None


def _grab_region(region: dict[str, int]) -> Image.Image:
    try:
        with mss.mss() as sct:
            raw = sct.grab(region)
            return Image.frombytes("RGB", raw.size, raw.rgb)
    except Exception:
        bbox = (
            region["left"],
            region["top"],
            region["left"] + region["width"],
            region["top"] + region["height"],
        )
        return ImageGrab.grab(bbox=bbox, all_screens=True).convert("RGB")


def _grab_monitor_under_cursor() -> Image.Image:
    cx, cy = _get_cursor_pos()
    try:
        with mss.mss() as sct:
            monitors = sct.monitors[1:] or [sct.monitors[0]]
            target = monitors[0]
            for m in monitors:
                if m["left"] <= cx < m["left"] + m["width"] and m["top"] <= cy < m["top"] + m["height"]:
                    target = m
                    break
            raw = sct.grab(target)
            return Image.frombytes("RGB", raw.size, raw.rgb)
    except Exception:
        return ImageGrab.grab().convert("RGB")


def capture_screen_jpeg(
    mode: str = "monitor",
    exclude_hwnd: int = 0,
) -> tuple[bytes, tuple[int, int], str]:
    """
    Capture either the full monitor under the mouse cursor ('monitor') or only the
    active foreground window ('active_window').
    Returns (jpeg_bytes, (width, height), source_label).
    """
    label = "Экран"
    img: Image.Image | None = None

    if mode == "active_window":
        found = _find_target_window(exclude_hwnd=exclude_hwnd)
        if found is not None:
            (left, top, width, height), title = found
            region = {"left": left, "top": top, "width": width, "height": height}
            img = _grab_region(region)
            short_title = title if len(title) <= 28 else title[:26] + "…"
            label = f"Окно: {short_title}"

    if img is None:
        img = _grab_monitor_under_cursor()

    if img.width > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / float(img.width)
        new_size = (MAX_IMAGE_WIDTH, max(1, int(img.height * ratio)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue(), img.size, label
