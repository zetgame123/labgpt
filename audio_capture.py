"""Continuous WASAPI loopback audio ring buffer for capturing system audio on Windows."""

from __future__ import annotations

import io
import threading
import time
import wave
from collections import deque
from typing import Optional

import numpy as np
import pyaudiowpatch as pyaudio

TARGET_SAMPLE_RATE = 16000


def list_loopback_devices() -> list[str]:
    """Return names of all available Windows WASAPI loopback devices."""
    devices: list[str] = []
    p: Optional[pyaudio.PyAudio] = None
    try:
        p = pyaudio.PyAudio()
        seen: set[str] = set()
        for dev in p.get_loopback_device_info_generator():
            name = str(dev.get("name", "")).strip()
            if name and name not in seen:
                seen.add(name)
                devices.append(name)
    except Exception:
        pass
    finally:
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass
    return devices


class SystemAudioBuffer:
    """
    Captures Windows system output audio via WASAPI loopback into an in-memory
    ring buffer holding the last `buffer_seconds` of audio.
    """

    def __init__(self, buffer_seconds: int = 40, preferred_device: str = "") -> None:
        self.buffer_seconds = max(5, buffer_seconds)
        self._preferred_device: str = preferred_device.strip()
        self._restart_requested: bool = False
        self._lock = threading.Lock()
        self._chunks: deque[tuple[float, np.ndarray]] = deque()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._device_name: str = "Не подключено"
        self._last_error: str = ""
        self._current_rms: float = 0.0
        self._sample_rate: int = TARGET_SAMPLE_RATE

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def current_rms(self) -> float:
        return self._current_rms

    @property
    def is_running(self) -> bool:
        return self._running

    def set_buffer_seconds(self, seconds: int) -> None:
        with self._lock:
            self.buffer_seconds = max(5, min(120, seconds))
            self._prune_locked(time.monotonic())

    def set_device(self, device_name: str) -> None:
        """Switch capture to a specific WASAPI loopback device (empty = default output)."""
        new_dev = device_name.strip()
        if new_dev == self._preferred_device:
            return
        self._preferred_device = new_dev
        self._restart_requested = True
        self.clear()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._last_error = ""
        self._thread = threading.Thread(target=self._capture_loop, name="WASAPILoopback", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.buffer_seconds
        while self._chunks and self._chunks[0][0] < cutoff:
            self._chunks.popleft()

    def _downsample_to_mono_16k(self, pcm_bytes: bytes, channels: int, src_rate: int) -> np.ndarray:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        if channels > 1:
            usable = (len(audio) // channels) * channels
            audio = audio[:usable].reshape(-1, channels).mean(axis=1).astype(np.int16)

        if src_rate == TARGET_SAMPLE_RATE or len(audio) == 0:
            return audio

        # Fast linear interpolation resample to 16 kHz (more than enough for speech)
        target_len = max(1, int(round(len(audio) * TARGET_SAMPLE_RATE / src_rate)))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        resampled = np.interp(x_new, x_old, audio.astype(np.float32))
        return np.clip(resampled, -32768, 32767).astype(np.int16)

    def _select_loopback_device(self, p: pyaudio.PyAudio) -> dict:
        preferred = self._preferred_device
        all_loopbacks = list(p.get_loopback_device_info_generator())
        if not all_loopbacks:
            raise RuntimeError("Не найдено WASAPI Loopback устройство вывода звука.")

        if preferred:
            for dev in all_loopbacks:
                if dev["name"] == preferred:
                    return dev
            for dev in all_loopbacks:
                if preferred.lower() in dev["name"].lower():
                    return dev

        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        if default_speakers.get("isLoopbackDevice", False):
            return default_speakers

        for dev in all_loopbacks:
            if default_speakers["name"] in dev["name"]:
                return dev
        return all_loopbacks[0]

    def _capture_loop(self) -> None:
        while self._running:
            self._restart_requested = False
            p: Optional[pyaudio.PyAudio] = None
            stream = None
            try:
                p = pyaudio.PyAudio()
                loopback_device = self._select_loopback_device(p)

                channels = max(1, int(loopback_device["maxInputChannels"]))
                src_rate = int(loopback_device["defaultSampleRate"])
                chunk_frames = max(512, int(src_rate * 0.1))  # 100 ms chunks

                self._device_name = loopback_device["name"]
                self._last_error = ""

                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=src_rate,
                    frames_per_buffer=chunk_frames,
                    input=True,
                    input_device_index=loopback_device["index"],
                )

                while self._running and not self._restart_requested:
                    data = stream.read(chunk_frames, exception_on_overflow=False)
                    mono_16k = self._downsample_to_mono_16k(data, channels, src_rate)
                    if len(mono_16k) == 0:
                        continue

                    rms = float(np.sqrt(np.mean(mono_16k.astype(np.float32) ** 2))) / 32768.0
                    self._current_rms = rms

                    now = time.monotonic()
                    with self._lock:
                        self._chunks.append((now, mono_16k))
                        self._prune_locked(now)

            except Exception as exc:
                self._last_error = str(exc)
                self._device_name = "Ошибка аудио"
                self._current_rms = 0.0
                time.sleep(2.0)
            finally:
                try:
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
                try:
                    if p is not None:
                        p.terminate()
                except Exception:
                    pass

    def get_wav_bytes(self, last_seconds: Optional[int] = None) -> tuple[bytes, float]:
        """
        Return (wav_bytes, duration_seconds) for the buffered system audio.
        Returns (b"", 0.0) if buffer is empty or silent.
        """
        now = time.monotonic()
        window = last_seconds if last_seconds is not None else self.buffer_seconds
        cutoff = now - window

        with self._lock:
            selected = [chunk for ts, chunk in self._chunks if ts >= cutoff]

        if not selected:
            return b"", 0.0

        combined = np.concatenate(selected)
        duration = len(combined) / float(TARGET_SAMPLE_RATE)
        if duration < 0.3:
            return b"", 0.0

        # Check if audio is essentially pure digital silence
        peak = int(np.max(np.abs(combined)))
        if peak < 25:
            return b"", 0.0

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(combined.tobytes())
        return buf.getvalue(), duration

    def get_wav_bytes_between(self, start_ts: float, end_ts: Optional[float] = None) -> tuple[bytes, float]:
        """Return (wav_bytes, duration_seconds) for buffered audio in [start_ts, end_ts]."""
        hi = end_ts if end_ts is not None else time.monotonic()
        with self._lock:
            selected = [chunk for ts, chunk in self._chunks if start_ts <= ts <= hi]

        if not selected:
            return b"", 0.0

        combined = np.concatenate(selected)
        duration = len(combined) / float(TARGET_SAMPLE_RATE)
        if duration < 0.4:
            return b"", 0.0

        if int(np.max(np.abs(combined))) < 35:
            return b"", 0.0

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SAMPLE_RATE)
            wf.writeframes(combined.tobytes())
        return buf.getvalue(), duration
