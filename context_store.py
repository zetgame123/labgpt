"""Manages user-selected lab/project files and builds token-budgeted context for Groq."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from config import BASE_DIR

CONTEXT_STATE_PATH = BASE_DIR / "context_files.json"

# Keeps prompt comfortably inside Groq's free-tier 8,000 TPM limit
MAX_CONTEXT_CHARS = 14000
CHUNK_CHARS = 1600
CHUNK_OVERLAP = 200

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "bin",
    "obj",
    "target",
}

ALLOWED_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".java", ".kt", ".go", ".rs", ".php", ".rb", ".swift",
    ".html", ".htm", ".css", ".scss", ".less", ".vue", ".svelte",
    ".sql", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".env",
    ".md", ".txt", ".rst", ".csv", ".sh", ".bat", ".ps1", ".ipynb",
}

_WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]{2,}")


@dataclass
class LoadedFile:
    path: str
    name: str
    content: str

    @property
    def chars(self) -> int:
        return len(self.content)


def _read_ipynb(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    cells = data.get("cells", [])
    parts: list[str] = []
    for idx, cell in enumerate(cells, 1):
        ctype = cell.get("cell_type", "code")
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else str(src)
        text = text.strip()
        if not text:
            continue
        parts.append(f"# --- Cell {idx} ({ctype}) ---\n{text}")
    return "\n\n".join(parts)


def _read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        return _read_ipynb(path)
    raw = path.read_bytes()
    # Skip obvious binary files
    if b"\x00" in raw[:4096]:
        raise ValueError("Бинарный файл не поддерживается")
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


class ContextStore:
    def __init__(self) -> None:
        self._files: list[LoadedFile] = []
        self._load_saved_paths()

    @property
    def files(self) -> list[LoadedFile]:
        return list(self._files)

    @property
    def total_chars(self) -> int:
        return sum(f.chars for f in self._files)

    @property
    def file_count(self) -> int:
        return len(self._files)

    def _save_paths(self) -> None:
        try:
            paths = [f.path for f in self._files]
            CONTEXT_STATE_PATH.write_text(
                json.dumps(paths, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _load_saved_paths(self) -> None:
        if not CONTEXT_STATE_PATH.exists():
            return
        try:
            paths = json.loads(CONTEXT_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(paths, list):
                self.add_files([str(p) for p in paths], save=False)
        except Exception:
            pass

    def add_files(self, paths: Iterable[str], save: bool = True) -> tuple[int, list[str]]:
        """Add file paths. Returns (added_count, error_messages)."""
        existing = {Path(f.path).resolve() for f in self._files}
        added = 0
        errors: list[str] = []

        for raw_path in paths:
            p = Path(raw_path)
            if not p.is_file():
                continue
            resolved = p.resolve()
            if resolved in existing:
                continue
            try:
                content = _read_text_file(resolved)
                self._files.append(LoadedFile(path=str(resolved), name=resolved.name, content=content))
                existing.add(resolved)
                added += 1
            except Exception as exc:
                errors.append(f"{p.name}: {exc}")

        if save and added > 0:
            self._save_paths()
        return added, errors

    def add_folder(self, folder_path: str) -> tuple[int, list[str]]:
        root = Path(folder_path)
        if not root.is_dir():
            return 0, ["Папка не найдена"]
        collected: list[str] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.suffix.lower() in ALLOWED_EXTS and p.stat().st_size <= 512_000:
                collected.append(str(p))
        return self.add_files(collected)

    def remove_paths(self, paths: Iterable[str]) -> None:
        to_remove = {str(Path(p).resolve()) for p in paths}
        self._files = [f for f in self._files if f.path not in to_remove]
        self._save_paths()

    def clear(self) -> None:
        self._files.clear()
        self._save_paths()

    def reload_all(self) -> None:
        """Refresh file contents from disk in case the user edited a lab file."""
        refreshed: list[LoadedFile] = []
        for lf in self._files:
            p = Path(lf.path)
            if not p.is_file():
                continue
            try:
                refreshed.append(LoadedFile(path=lf.path, name=lf.name, content=_read_text_file(p)))
            except Exception:
                refreshed.append(lf)
        self._files = refreshed

    def build_context_prompt(self, query: str = "", max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """
        Build a concise context block from loaded files.
        If all files fit in `max_chars`, include them in full; otherwise rank and pick
        the most relevant chunks for `query`.
        """
        if not self._files:
            return ""

        self.reload_all()
        if not self._files:
            return ""

        file_list = ", ".join(f.name for f in self._files)
        header = (
            f"Контекст из прикреплённых файлов пользователя ({len(self._files)} шт.: {file_list}).\n"
            "Если вопрос касается этих файлов / лабораторных работ, опирайся именно на их код и содержимое:\n\n"
        )

        # Fast path: everything fits in full
        if self.total_chars + len(header) <= max_chars:
            blocks = [f"=== Файл: {f.name} ===\n{f.content.strip()}" for f in self._files]
            return header + "\n\n".join(blocks)

        # Smart chunk selection when files exceed budget
        query_tokens = _tokenize(query)
        scored_chunks: list[tuple[float, str, str]] = []  # (score, filename, chunk_text)

        for lf in self._files:
            text = lf.content
            name_tokens = _tokenize(lf.name)
            pos = 0
            chunk_idx = 0
            while pos < len(text):
                end = min(len(text), pos + CHUNK_CHARS)
                chunk = text[pos:end].strip()
                if chunk:
                    chunk_tokens = _tokenize(chunk) | name_tokens
                    overlap = len(query_tokens & chunk_tokens) if query_tokens else 0
                    # Slight bonus for start of each file so structure/imports are preserved
                    score = overlap * 3.0 + (1.0 if chunk_idx == 0 else 0.0)
                    scored_chunks.append((score, lf.name, chunk))
                if end >= len(text):
                    break
                pos = end - CHUNK_OVERLAP
                chunk_idx += 1

        scored_chunks.sort(key=lambda item: item[0], reverse=True)

        selected: list[str] = []
        used = len(header)
        for _score, fname, chunk in scored_chunks:
            block = f"=== Фрагмент из {fname} ===\n{chunk}"
            if used + len(block) + 2 > max_chars:
                continue
            selected.append(block)
            used += len(block) + 2
            if used >= max_chars - 300:
                break

        return header + "\n\n".join(selected)
