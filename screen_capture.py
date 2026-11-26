"""Fast in-memory screen capture (automatically excludes WDA_EXCLUDEFROMCAPTURE overlay)."""

from __future__ import annotations

import ctypes
import io
from ctypes import wintypes

import mss
from PIL import Image, ImageGrab

MAX_IMAGE_WIDTH = 1920
JPEG_QUALITY = 82


def _get_cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT()
    if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
        return int(pt.x), int(pt.y)
    return 0, 0


def _grab_pil_image() -> Image.Image:
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


def capture_screen_jpeg() -> tuple[bytes, tuple[int, int]]:
    """
    Capture the monitor currently under the mouse cursor (falling back to primary)
    and return (jpeg_bytes, (width, height)).
    """
    img = _grab_pil_image()

    if img.width > MAX_IMAGE_WIDTH:
        ratio = MAX_IMAGE_WIDTH / float(img.width)
        new_size = (MAX_IMAGE_WIDTH, max(1, int(img.height * ratio)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue(), img.size
