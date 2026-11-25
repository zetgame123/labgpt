from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, set_key

from hotkeys import (
    DEFAULT_HOTKEY_ANSWER,
    DEFAULT_HOTKEY_CLEAR,
    DEFAULT_HOTKEY_CLICK_THROUGH,
    DEFAULT_HOTKEY_TOGGLE,
)

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"
DEFAULT_BUFFER_SECONDS = 40
DEFAULT_USER_ROLE = "Software Engineer"


@dataclass
class AppConfig:
    api_key: str = ""
    model: str = DEFAULT_MODEL
    stt_model: str = DEFAULT_STT_MODEL
    buffer_seconds: int = DEFAULT_BUFFER_SECONDS
    use_web_search: bool = False
    user_role: str = DEFAULT_USER_ROLE
    opacity: float = 0.88
    audio_device: str = ""
    hotkey_answer: str = DEFAULT_HOTKEY_ANSWER
    hotkey_toggle: str = DEFAULT_HOTKEY_TOGGLE
    hotkey_clear: str = DEFAULT_HOTKEY_CLEAR
    hotkey_click_through: str = DEFAULT_HOTKEY_CLICK_THROUGH

    @classmethod
    def load(cls) -> "AppConfig":
        values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}

        def _get(name: str, default: str = "") -> str:
            val = values.get(name)
            if val is None:
                val = os.getenv(name, default)
            return (val or default).strip()

        try:
            buf_sec = int(_get("AUDIO_BUFFER_SECONDS", str(DEFAULT_BUFFER_SECONDS)))
            buf_sec = max(5, min(120, buf_sec))
        except ValueError:
            buf_sec = DEFAULT_BUFFER_SECONDS

        try:
            opacity = float(_get("WINDOW_OPACITY", "0.88"))
            opacity = max(0.3, min(1.0, opacity))
        except ValueError:
            opacity = 0.88

        use_search = _get("USE_WEB_SEARCH", "0").lower() in ("1", "true", "yes", "on")

        return cls(
            api_key=_get("GROQ_API_KEY", ""),
            model=_get("GROQ_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
            stt_model=_get("GROQ_STT_MODEL", DEFAULT_STT_MODEL) or DEFAULT_STT_MODEL,
            buffer_seconds=buf_sec,
            use_web_search=use_search,
            user_role=_get("USER_ROLE", DEFAULT_USER_ROLE) or DEFAULT_USER_ROLE,
            opacity=opacity,
            audio_device=_get("AUDIO_DEVICE", ""),
            hotkey_answer=_get("HOTKEY_ANSWER", DEFAULT_HOTKEY_ANSWER) or DEFAULT_HOTKEY_ANSWER,
            hotkey_toggle=_get("HOTKEY_TOGGLE", DEFAULT_HOTKEY_TOGGLE) or DEFAULT_HOTKEY_TOGGLE,
            hotkey_clear=_get("HOTKEY_CLEAR", DEFAULT_HOTKEY_CLEAR) or DEFAULT_HOTKEY_CLEAR,
            hotkey_click_through=_get("HOTKEY_CLICK_THROUGH", DEFAULT_HOTKEY_CLICK_THROUGH)
            or DEFAULT_HOTKEY_CLICK_THROUGH,
        )

    def save(self) -> None:
        if not ENV_PATH.exists():
            ENV_PATH.write_text("", encoding="utf-8")
        set_key(str(ENV_PATH), "GROQ_API_KEY", self.api_key)
        set_key(str(ENV_PATH), "GROQ_MODEL", self.model)
        set_key(str(ENV_PATH), "GROQ_STT_MODEL", self.stt_model)
        set_key(str(ENV_PATH), "AUDIO_BUFFER_SECONDS", str(self.buffer_seconds))
        set_key(str(ENV_PATH), "USE_WEB_SEARCH", "1" if self.use_web_search else "0")
        set_key(str(ENV_PATH), "USER_ROLE", self.user_role)
        set_key(str(ENV_PATH), "WINDOW_OPACITY", f"{self.opacity:.2f}")
        set_key(str(ENV_PATH), "AUDIO_DEVICE", self.audio_device)
        set_key(str(ENV_PATH), "HOTKEY_ANSWER", self.hotkey_answer)
        set_key(str(ENV_PATH), "HOTKEY_TOGGLE", self.hotkey_toggle)
        set_key(str(ENV_PATH), "HOTKEY_CLEAR", self.hotkey_clear)
        set_key(str(ENV_PATH), "HOTKEY_CLICK_THROUGH", self.hotkey_click_through)
