"""Background energy-VAD transcriber keeping a rolling 5-minute dialogue transcript."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

from ai_client import AIAssistant
from audio_capture import SystemAudioBuffer

SPEECH_RMS_THRESHOLD = 0.008
SILENCE_END_SECONDS = 1.1
MAX_PHRASE_SECONDS = 18.0
PRE_ROLL_SECONDS = 0.35
TRANSCRIPT_WINDOW_SECONDS = 300  # 5 minutes

# Common Whisper hallucinations on background noise / silence
_HALLUCINATIONS = {
    "субтитры сделал dimatorzok",
    "субтитры создавал dimatorzok",
    "редактор субтитров а.синецкая",
    "спасибо за просмотр",
    "спасибо за просмотр!",
    "продолжение следует...",
    "thank you.",
    "thank you for watching.",
    "thanks for watching!",
    "you",
    "...",
}


def _is_hallucination(text: str) -> bool:
    norm = text.strip().lower().strip(".,!?… ")
    if len(norm) < 2:
        return True
    if norm in _HALLUCINATIONS or text.strip().lower() in _HALLUCINATIONS:
        return True
    if "субтитры" in norm and len(norm) < 45:
        return True
    return False


class LiveTranscriber:
    """
    Watches SystemAudioBuffer with a lightweight energy VAD, sends finished speech
    phrases to Whisper in the background, and stores a rolling 5-minute transcript.
    """

    def __init__(
        self,
        audio_buffer: SystemAudioBuffer,
        assistant: AIAssistant,
        enabled: bool = True,
        window_seconds: int = TRANSCRIPT_WINDOW_SECONDS,
    ) -> None:
        self.audio_buffer = audio_buffer
        self.assistant = assistant
        self._enabled = enabled
        self._window_seconds = window_seconds

        self._lock = threading.Lock()
        self._transcribe_lock = threading.Lock()
        self._entries: deque[tuple[float, str]] = deque()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._in_speech = False
        self._speech_start: float = 0.0
        self._last_voice_ts: float = 0.0
        self._last_transcribed_ts: float = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def entry_count(self) -> int:
        with self._lock:
            self._prune_locked(time.monotonic())
            return len(self._entries)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self._in_speech = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._vad_loop, name="LiveVAD", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
        self._in_speech = False
        self._last_transcribed_ts = time.monotonic()

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

    def _transcribe_segment(self, start_ts: float, end_ts: float) -> None:
        if not self.assistant.has_stt_key():
            return
        with self._transcribe_lock:
            actual_start = max(start_ts, self._last_transcribed_ts)
            if end_ts - actual_start < 0.5:
                return
            wav_bytes, duration = self.audio_buffer.get_wav_bytes_between(actual_start, end_ts)
            self._last_transcribed_ts = end_ts
            if not wav_bytes or duration < 0.5:
                return
            try:
                text = self.assistant.transcribe_audio(wav_bytes).strip()
            except Exception:
                return
            if text and not _is_hallucination(text):
                now = time.monotonic()
                with self._lock:
                    self._entries.append((now, text))
                    self._prune_locked(now)

    def _vad_loop(self) -> None:
        while self._running:
            time.sleep(0.15)
            if not self._enabled or not self.assistant.has_stt_key():
                self._in_speech = False
                continue

            now = time.monotonic()
            rms = self.audio_buffer.current_rms

            if rms >= SPEECH_RMS_THRESHOLD:
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_start = max(self._last_transcribed_ts, now - PRE_ROLL_SECONDS)
                self._last_voice_ts = now
                if now - self._speech_start >= MAX_PHRASE_SECONDS:
                    seg_start = self._speech_start
                    self._speech_start = now
                    threading.Thread(
                        target=self._transcribe_segment,
                        args=(seg_start, now),
                        daemon=True,
                    ).start()
            elif self._in_speech:
                if now - self._last_voice_ts >= SILENCE_END_SECONDS:
                    seg_start = self._speech_start
                    seg_end = now
                    self._in_speech = False
                    threading.Thread(
                        target=self._transcribe_segment,
                        args=(seg_start, seg_end),
                        daemon=True,
                    ).start()

    def flush_and_get_transcript(self) -> str:
        """
        If speech is currently ongoing, immediately transcribe the tail so nothing
        is missed, then return the rolling 5-minute transcript text.
        """
        if self._enabled and self._in_speech:
            now = time.monotonic()
            seg_start = self._speech_start
            self._in_speech = False
            self._transcribe_segment(seg_start, now)

        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            lines = [text for _ts, text in self._entries]
        return "\n".join(lines).strip()
