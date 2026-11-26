"""Entry point for the stealth interview audio assistant."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from ai_client import AIAssistant
from audio_capture import SystemAudioBuffer
from config import AppConfig
from context_store import ContextStore
from hotkeys import HotkeyManager
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
    assistant = AIAssistant(config, context_store=context_store)
    hotkey_manager = HotkeyManager()
    window = OverlayWindow(
        config=config,
        audio_buffer=audio_buffer,
        assistant=assistant,
        context_store=context_store,
        hotkey_manager=hotkey_manager,
    )
    window.show()

    try:
        return app.exec()
    finally:
        hotkey_manager.stop()
        audio_buffer.stop()


if __name__ == "__main__":
    raise SystemExit(main())
