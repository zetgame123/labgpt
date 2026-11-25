"""Entry point for the stealth interview audio assistant."""

from __future__ import annotations

import sys

from pynput import keyboard
from PyQt6.QtWidgets import QApplication

from ai_client import GroqAssistant
from audio_capture import SystemAudioBuffer
from config import AppConfig
from context_store import ContextStore
from overlay_ui import OverlayWindow


def main() -> int:
    config = AppConfig.load()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    audio_buffer = SystemAudioBuffer(
        buffer_seconds=config.buffer_seconds,
        preferred_device=config.audio_device,
    )
    audio_buffer.start()

    context_store = ContextStore()
    assistant = GroqAssistant(config, context_store=context_store)
    window = OverlayWindow(
        config=config,
        audio_buffer=audio_buffer,
        assistant=assistant,
        context_store=context_store,
    )
    window.show()

    hotkeys = keyboard.GlobalHotKeys(
        {
            "<ctrl>+<shift>+<space>": lambda: window.sig_hotkey_answer.emit(),
            "<ctrl>+<shift>+h": lambda: window.sig_hotkey_toggle_visible.emit(),
            "<ctrl>+<shift>+c": lambda: window.sig_hotkey_clear.emit(),
            "<ctrl>+<shift>+t": lambda: window.sig_hotkey_click_through.emit(),
        }
    )
    hotkeys.start()

    try:
        return app.exec()
    finally:
        hotkeys.stop()
        audio_buffer.stop()


if __name__ == "__main__":
    raise SystemExit(main())
