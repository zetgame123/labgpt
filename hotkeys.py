"""Native Windows RegisterHotKey manager (works identically on RU/EN keyboard layouts)."""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes
from typing import Callable, Optional

DEFAULT_HOTKEY_ANSWER = "Ctrl+Shift+Space"
DEFAULT_HOTKEY_TOGGLE = "Ctrl+Shift+H"
DEFAULT_HOTKEY_CLEAR = "Ctrl+Shift+C"
DEFAULT_HOTKEY_CLICK_THROUGH = "Ctrl+Shift+T"

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
_user32.RegisterHotKey.restype = wintypes.BOOL

_user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.UnregisterHotKey.restype = wintypes.BOOL

_user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
_user32.GetMessageW.restype = wintypes.BOOL

_user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
_user32.PostThreadMessageW.restype = wintypes.BOOL

_user32.VkKeyScanW.argtypes = [wintypes.WCHAR]
_user32.VkKeyScanW.restype = ctypes.c_short

_kernel32.GetCurrentThreadId.argtypes = []
_kernel32.GetCurrentThreadId.restype = wintypes.DWORD

_MODIFIER_MAP = {
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "shift": MOD_SHIFT,
    "alt": MOD_ALT,
    "meta": MOD_WIN,
    "win": MOD_WIN,
    "cmd": MOD_WIN,
}

_VK_MAP = {
    "backspace": 0x08,
    "tab": 0x09,
    "return": 0x0D,
    "enter": 0x0D,
    "pause": 0x13,
    "capslock": 0x14,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "pgup": 0x21,
    "pageup": 0x21,
    "pgdown": 0x22,
    "pagedown": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "ins": 0x2D,
    "insert": 0x2D,
    "del": 0x2E,
    "delete": 0x2E,
    "numlock": 0x90,
    "scrolllock": 0x91,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "f13": 0x7C, "f14": 0x7D, "f15": 0x7E, "f16": 0x7F,
    "f17": 0x80, "f18": 0x81, "f19": 0x82, "f20": 0x83,
    "f21": 0x84, "f22": 0x85, "f23": 0x86, "f24": 0x87,
}

# Cyrillic letter -> corresponding physical Latin VK key (so even if recorded on RU layout it works everywhere)
_RU_TO_LATIN_VK = {
    "й": "Q", "ц": "W", "у": "E", "к": "R", "е": "T", "н": "Y", "г": "U", "ш": "I", "щ": "O", "з": "P",
    "ф": "A", "ы": "S", "в": "D", "а": "F", "п": "G", "р": "H", "о": "J", "л": "K", "д": "L",
    "я": "Z", "ч": "X", "с": "C", "м": "V", "и": "B", "т": "N", "ь": "M",
}


def parse_qt_hotkey(seq_str: str) -> Optional[tuple[int, int]]:
    """
    Parse a Qt QKeySequence portable string (e.g. 'Ctrl+Shift+T', 'Alt+F2')
    into (win32_modifiers, virtual_key_code). Returns None if invalid.
    """
    raw = (seq_str or "").strip()
    if not raw:
        return None

    chord = raw.split(",")[0].strip()
    tokens: list[str] = []
    if chord.endswith("++"):
        prefix = chord[:-2]
        if prefix:
            tokens.extend(t for t in prefix.split("+") if t)
        tokens.append("+")
    else:
        tokens = [t for t in chord.split("+") if t]

    mods = 0
    vk = 0
    for tok in tokens:
        t = tok.strip()
        low = t.lower()
        if low in _MODIFIER_MAP:
            mods |= _MODIFIER_MAP[low]
        elif low in _VK_MAP:
            vk = _VK_MAP[low]
        elif len(t) == 1:
            if low in _RU_TO_LATIN_VK:
                vk = ord(_RU_TO_LATIN_VK[low])
            elif "a" <= low <= "z":
                vk = ord(t.upper())
            elif "0" <= low <= "9":
                vk = ord(t)
            else:
                scan = _user32.VkKeyScanW(t)
                if scan != -1:
                    vk = scan & 0xFF

    if vk == 0:
        return None
    return (mods | MOD_NOREPEAT, vk)


class HotkeyManager:
    """Registers global Windows hotkeys on a dedicated Win32 message-loop thread."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._ready = threading.Event()
        self.last_error: str = ""

    def register(self, bindings: list[tuple[str, Callable[[], None]]]) -> None:
        self.stop()
        self.last_error = ""

        parsed: list[tuple[int, int, str, Callable[[], None]]] = []
        for qt_seq, cb in bindings:
            res = parse_qt_hotkey(qt_seq)
            if res is not None:
                mods, vk = res
                parsed.append((mods, vk, qt_seq, cb))

        if not parsed:
            return

        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(parsed,),
            name="Win32Hotkeys",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=2.0)

    def _run_loop(self, parsed: list[tuple[int, int, str, Callable[[], None]]]) -> None:
        self._thread_id = int(_kernel32.GetCurrentThreadId())
        callbacks: dict[int, Callable[[], None]] = {}
        registered_ids: list[int] = []
        errors: list[str] = []

        # Force creation of this thread's message queue before signaling ready
        msg = wintypes.MSG()
        for idx, (mods, vk, label, cb) in enumerate(parsed, start=1):
            if _user32.RegisterHotKey(None, idx, mods, vk):
                callbacks[idx] = cb
                registered_ids.append(idx)
            else:
                err_code = ctypes.get_last_error()
                errors.append(f"{label} (err={err_code})")

        if errors:
            self.last_error = "Не удалось зарегистрировать: " + ", ".join(errors)

        self._ready.set()

        try:
            while True:
                res = _user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if res <= 0:  # 0 = WM_QUIT, -1 = error
                    break
                if msg.message == WM_HOTKEY:
                    cb = callbacks.get(int(msg.wParam))
                    if cb is not None:
                        try:
                            cb()
                        except Exception:
                            pass
        finally:
            for hk_id in registered_ids:
                _user32.UnregisterHotKey(None, hk_id)
            self._thread_id = None

    def stop(self) -> None:
        tid = self._thread_id
        if tid is not None:
            _user32.PostThreadMessageW(wintypes.DWORD(tid), WM_QUIT, 0, 0)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self._thread_id = None
