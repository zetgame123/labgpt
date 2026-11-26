"""Unified AI client supporting Groq (fast/free) and OpenAI (large projects + screen vision)."""

from __future__ import annotations

import base64
from typing import Any, Generator

from groq import Groq
from openai import OpenAI

from config import AppConfig
from context_store import ContextStore


def _build_system_prompt(user_role: str) -> str:
    return (
        f"Ты — быстрый и точный ассистент-подсказчик для кандидата на техническом собеседовании "
        f"(позиция: {user_role}).\n"
        "Правила ответа:\n"
        "1. Если передана расшифровка речи с собеседования, сначала выдели последний прозвучавший вопрос "
        "собеседника в одну короткую строку формата: **Вопрос:** <суть вопроса>.\n"
        "2. Если передан скриншот экрана, сразу определи, что на нём (задача, тест, код, ошибка, вопрос) "
        "и дай точное решение / объяснение.\n"
        "3. Дай чёткий, практичный и структурированный ответ, который легко быстро прочитать глазами и пересказать своими словами:\n"
        "   - Короткая суть в 1–2 предложениях.\n"
        "   - Ключевые тезисы списком (3–5 пунктов максимум).\n"
        "   - Если нужен код или команда — минимальный чистый пример.\n"
        "4. Не лей воду, не пиши приветствий и оговорок. Отвечай на том же языке, на котором задан вопрос (обычно русский).\n"
        "5. НИКОГДА не используй LaTeX-разметку (`\\(...\\)`, `\\[...\\]`, `$$...$$`, `\\frac`, `\\rightarrow`, `\\begin{array}` и т.п.) — окно её не рендерит. "
        "Для формул, логики и математики пиши только чистый читаемый Unicode (`→`, `↔`, `⇒`, `¬`, `∧`, `∨`, `∀`, `∃`, `⊢`, `≤`, `≥`, `≠`, `∈`, `⊆`, `∅`, `∞`, `²`, `³`, дроби через `/`, а многострочные правила вывода — в обычном текстовом блоке ```).\n"
        "6. Если в расшифровке нет явного вопроса, кратко резюмируй последнюю тему и дай 2–3 полезных тезиса по ней."
    )


class AIAssistant:
    def __init__(self, config: AppConfig, context_store: ContextStore | None = None) -> None:
        self.config = config
        self.context_store = context_store
        self._groq_client: Groq | None = None
        self._openai_client: OpenAI | None = None
        self._history: list[tuple[str, str]] = []  # (user_summary, assistant_reply)
        self._rebuild_clients()

    def _rebuild_clients(self) -> None:
        self._groq_client = Groq(api_key=self.config.api_key) if self.config.api_key else None
        if self.config.openai_api_key:
            kwargs: dict[str, Any] = {"api_key": self.config.openai_api_key}
            if self.config.openai_base_url:
                kwargs["base_url"] = self.config.openai_base_url
            self._openai_client = OpenAI(**kwargs)
        else:
            self._openai_client = None

    def update_config(self, config: AppConfig) -> None:
        self.config = config
        self._rebuild_clients()

    def clear_history(self) -> None:
        self._history.clear()

    def _ensure_groq(self) -> Groq:
        if not self.config.api_key:
            raise ValueError(
                "Не задан GROQ_API_KEY. Нажми ⚙ и вставь ключ с console.groq.com/keys (или переключи провайдера на OpenAI)."
            )
        if self._groq_client is None:
            self._rebuild_clients()
        assert self._groq_client is not None
        return self._groq_client

    def _ensure_openai(self) -> OpenAI:
        if not self.config.openai_api_key:
            raise ValueError(
                "Не задан OPENAI_API_KEY. Нажми ⚙ и вставь ключ с platform.openai.com/api-keys."
            )
        if self._openai_client is None:
            self._rebuild_clients()
        assert self._openai_client is not None
        return self._openai_client

    def _build_messages(
        self,
        user_content: Any,
        query_hint: str = "",
        max_context_chars: int | None = None,
    ) -> list[dict[str, Any]]:
        system_text = _build_system_prompt(self.config.user_role)
        if self.context_store and self.context_store.file_count > 0:
            budget = max_context_chars if max_context_chars is not None else self.config.max_context_chars
            ctx = self.context_store.build_context_prompt(
                query=query_hint or (user_content if isinstance(user_content, str) else ""),
                max_chars=budget,
            )
            if ctx:
                system_text = f"{system_text}\n\n{ctx}"

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_text}]
        for q, a in self._history[-4:]:
            messages.append({"role": "user", "content": q[:300]})
            messages.append({"role": "assistant", "content": a[:600]})
        messages.append({"role": "user", "content": user_content})
        return messages

    def transcribe_audio(self, wav_bytes: bytes) -> str:
        # Always prefer free, ultra-fast Groq Whisper when GROQ_API_KEY is present —
        # this also lets OpenAI-compatible providers without /audio/transcriptions
        # (like DeepSeek API or OpenRouter) work seamlessly with voice input!
        if self.config.api_key:
            client = self._ensure_groq()
            transcription = client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model=self.config.stt_model,
                response_format="text",
                temperature=0.0,
            )
            return str(transcription).strip()

        client = self._ensure_openai()
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes, "audio/wav"),
            model=self.config.openai_stt_model,
            response_format="text",
        )
        return str(transcription).strip()

    def _stream_groq(self, messages: list[dict[str, Any]], history_label: str) -> Generator[str, None, None]:
        client = self._ensure_groq()
        model = self.config.model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_completion_tokens": 1024,
            "stream": True,
        }
        if self.config.use_web_search and model.startswith("openai/gpt-oss"):
            kwargs["tools"] = [{"type": "browser_search"}]

        stream = client.chat.completions.create(**kwargs)
        full_parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                full_parts.append(content)
                yield content

        full_reply = "".join(full_parts).strip()
        if full_reply:
            self._history.append((history_label, full_reply))

    def _stream_openai(self, messages: list[dict[str, Any]], history_label: str) -> Generator[str, None, None]:
        client = self._ensure_openai()
        model = self.config.openai_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": 1800,
            "stream": True,
        }
        # Reasoning models (o1/o3/o4, deepseek-reasoner) reject custom temperature
        if not model.lower().startswith(("o1", "o3", "o4")) and "reasoner" not in model.lower():
            kwargs["temperature"] = 0.3

        stream = client.chat.completions.create(**kwargs)
        full_parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                full_parts.append(content)
                yield content

        full_reply = "".join(full_parts).strip()
        if full_reply:
            self._history.append((history_label, full_reply))

    def _stream_completion(self, user_content: str, history_label: str) -> Generator[str, None, None]:
        messages = self._build_messages(user_content, query_hint=history_label)
        if self.config.provider == "openai":
            yield from self._stream_openai(messages, history_label)
        else:
            yield from self._stream_groq(messages, history_label)

    def ask_from_audio_stream(self, wav_bytes: bytes, extra_note: str = "") -> Generator[str, None, None]:
        transcript = self.transcribe_audio(wav_bytes)
        if not transcript:
            yield "**Речь не распознана.**\n\nВ буфере не было отчётливой речи — попробуй ещё раз, когда собеседник договорит."
            return

        user_content = (
            f"Расшифровка последних секунд разговора на собеседовании:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
            "Выдели последний вопрос собеседника и дай краткий, точный ответ."
        )
        if extra_note.strip():
            user_content += f"\nДополнительная заметка кандидата: {extra_note.strip()}"

        yield from self._stream_completion(user_content, history_label=transcript)

    def ask_from_text_stream(self, question: str) -> Generator[str, None, None]:
        clean_q = question.strip()
        yield from self._stream_completion(
            f"Вопрос на собеседовании: {clean_q}",
            history_label=clean_q,
        )

    def ask_from_screen_stream(self, jpeg_bytes: bytes, extra_note: str = "") -> Generator[str, None, None]:
        """Analyze a screenshot using OpenAI Vision (works even when active provider is Groq)."""
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        prompt = (
            "Проанализируй скриншот экрана. Если на экране задача, тест, код, ошибка или вопрос — "
            "дай чёткий, краткий и правильный ответ / решение."
        )
        if extra_note.strip():
            prompt += f"\nУточнение пользователя: {extra_note.strip()}"

        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            },
        ]
        # Always use OpenAI's large context budget for screen + project analysis
        messages = self._build_messages(
            user_content,
            query_hint=extra_note or "анализ экрана кода задачи",
            max_context_chars=120_000,
        )
        yield from self._stream_openai(messages, history_label=extra_note or "(анализ экрана)")


# Backwards-compatible alias
GroqAssistant = AIAssistant
