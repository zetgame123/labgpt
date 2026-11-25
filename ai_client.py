"""Groq API client: ultra-fast Whisper STT + streaming LLM answers (with optional web search)."""

from __future__ import annotations

from typing import Generator

from groq import Groq

from config import AppConfig
from context_store import ContextStore


def _build_system_prompt(user_role: str) -> str:
    return (
        f"Ты — быстрый и точный ассистент-подсказчик для кандидата на техническом собеседовании "
        f"(позиция: {user_role}).\n"
        "Правила ответа:\n"
        "1. Если передана расшифровка речи с собеседования, сначала выдели последний прозвучавший вопрос "
        "собеседника в одну короткую строку формата: **Вопрос:** <суть вопроса>.\n"
        "2. Затем сразу дай чёткий, практичный и структурированный ответ, который легко быстро прочитать глазами и пересказать своими словами:\n"
        "   - Короткая суть в 1–2 предложениях.\n"
        "   - Ключевые тезисы списком (3–5 пунктов максимум).\n"
        "   - Если нужен код или команда — минимальный чистый пример.\n"
        "3. Не лей воду, не пиши приветствий и оговорок. Отвечай на том же языке, на котором задан вопрос (обычно русский).\n"
        "4. НИКОГДА не используй LaTeX-разметку (`\\(...\\)`, `\\[...\\]`, `$$...$$`, `\\frac`, `\\rightarrow`, `\\begin{array}` и т.п.) — окно её не рендерит. "
        "Для формул, логики и математики пиши только чистый читаемый Unicode (`→`, `↔`, `⇒`, `¬`, `∧`, `∨`, `∀`, `∃`, `⊢`, `≤`, `≥`, `≠`, `∈`, `⊆`, `∅`, `∞`, `²`, `³`, дроби через `/`, а многострочные правила вывода — в обычном текстовом блоке ```).\n"
        "5. Если в расшифровке нет явного вопроса, кратко резюмируй последнюю тему и дай 2–3 полезных тезиса по ней."
    )


class GroqAssistant:
    def __init__(self, config: AppConfig, context_store: ContextStore | None = None) -> None:
        self.config = config
        self.context_store = context_store
        self._client: Groq | None = None
        self._history: list[tuple[str, str]] = []  # (user_summary, assistant_reply)
        self._rebuild_client()

    def _rebuild_client(self) -> None:
        if self.config.api_key:
            self._client = Groq(api_key=self.config.api_key)
        else:
            self._client = None

    def update_config(self, config: AppConfig) -> None:
        self.config = config
        self._rebuild_client()

    def clear_history(self) -> None:
        self._history.clear()

    def _ensure_client(self) -> Groq:
        if not self.config.api_key:
            raise ValueError(
                "Не задан GROQ_API_KEY. Нажми на иконку настроек (⚙) и вставь ключ с console.groq.com/keys"
            )
        if self._client is None:
            self._rebuild_client()
        assert self._client is not None
        return self._client

    def _build_messages(self, user_content: str, query_hint: str = "") -> list[dict[str, str]]:
        system_text = _build_system_prompt(self.config.user_role)
        if self.context_store and self.context_store.file_count > 0:
            ctx = self.context_store.build_context_prompt(query=query_hint or user_content)
            if ctx:
                system_text = f"{system_text}\n\n{ctx}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_text}
        ]
        for q, a in self._history[-4:]:
            messages.append({"role": "user", "content": q[:300]})
            messages.append({"role": "assistant", "content": a[:600]})
        messages.append({"role": "user", "content": user_content})
        return messages

    def transcribe_audio(self, wav_bytes: bytes) -> str:
        client = self._ensure_client()
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=self.config.stt_model,
            response_format="text",
            temperature=0.0,
        )
        return str(transcription).strip()

    def _stream_completion(self, user_content: str, history_label: str) -> Generator[str, None, None]:
        client = self._ensure_client()
        model = self.config.model
        messages = self._build_messages(user_content, query_hint=history_label)

        kwargs: dict = {
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
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                full_parts.append(content)
                yield content

        full_reply = "".join(full_parts).strip()
        if full_reply:
            self._history.append((history_label, full_reply))

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
