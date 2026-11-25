"""Global hotkey manager with live reload and Qt QKeySequence -> pynput conversion."""

from __future__ import annotations

from typing import Callable, Optional

from pynput import keyboard

DEFAULT_HOTKEY_ANSWER = "Ctrl+Shift+Space"
DEFAULT_HOTKEY_TOGGLE = "Ctrl+Shift+H"
DEFAULT_HOTKEY_CLEAR = "Ctrl+Shift+C"
DEFAULT_HOTKEY_CLICK_THROUGH = "Ctrl+Shift+T"

_MODIFIER_MAP = {
    "ctrl": "<ctrl>",
    "control": "<ctrl>",
    "shift": "<shift>",
    "alt": "<alt>",
    "meta": "<cmd>",
    "win": "<cmd>",
    "cmd": "<cmd>",
}

_SPECIAL_KEY_MAP = {
    "space": "<space>",
    "tab": "<tab>",
    "return": "<enter>",
    "enter": "<enter>",
    "esc": "<esc>",
    "escape": "<esc>",
    "backspace": "<backspace>",
    "del": "<delete>",
    "delete": "<delete>",
    "ins": "<insert>",
    "insert": "<insert>",
    "home": "<home>",
    "end": "<end>",
    "pgup": "<page_up>",
    "pageup": "<page_up>",
    "pgdown": "<page_down>",
    "pagedown": "<page_down>",
    "up": "<up>",
    "down": "<down>",
    "left": "<left>",
    "right": "<right>",
    "pause": "<pause>",
    "capslock": "<caps_lock>",
    "numlock": "<num_lock>",
    "scrolllock": "<scroll_lock>",
}


def qt_seq_to_pynput(seq_str: str) -> str:
    """
    Convert a Qt QKeySequence portable string (e.g. 'Ctrl+Shift+Space', 'Alt+F2')
    into pynput GlobalHotKeys syntax (e.g. '<ctrl>+<shift>+<space>', '<alt>+<f2>').
    """
    raw = (seq_str or "").strip()
    if not raw:
        return ""

    # Take only the first chord if multiple were somehow recorded
    chord = raw.split(",")[0].strip()

    # Handle literal '+' key carefully (e.g. 'Ctrl++')
    tokens: list[str] = []
    if chord.endswith("++"):
        prefix = chord[:-2]
        if prefix:
            tokens.extend(t for t in prefix.split("+") if t)
        tokens.append("+")
    else:
        tokens = [t for t in chord.split("+") if t]

    converted: list[str] = []
    for tok in tokens:
        low = tok.strip().lower()
        if low in _MODIFIER_MAP:
            converted.append(_MODIFIER_MAP[low])
        elif low in _SPECIAL_KEY_MAP:
            converted.append(_SPECIAL_KEY_MAP[low])
        elif low.startswith("f") and low[1:].isdigit() and 1 <= int(low[1:]) <= 24:
            converted.append(f"<{low}>")
        elif len(low) == 1:
            converted.append(low)
        else:
            converted.append(f"<{low}>")

    return "+".join(converted)


class HotkeyManager:
    def __init__(self) -> None:
        self._listener: Optional[keyboard.GlobalHotKeys] = None
        self.last_error: str = ""

    def register(self, bindings: list[tuple[str, Callable[[], None]]]) -> None:
        """Stop previous listener and register a new list of (qt_seq_str, callback)."""
        self.stop()
        self.last_error = ""

        hotkey_map: dict[str, Callable[[], None]] = {}
        for qt_seq, cb in bindings:
            pynput_seq = qt_seq_to_pynput(qt_seq)
            if pynput_seq:
                hotkey_map[pynput_seq] = cb

        if not hotkey_map:
            return

        try:
            self._listener = keyboard.GlobalHotKeys(hotkey_map)
            self._listener.start()
        except Exception as exc:
            self.last_error = str(exc)
            self._listener = None

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
