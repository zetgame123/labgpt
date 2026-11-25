"""Stealth PyQt6 overlay window for displaying streamed AI interview answers."""

from __future__ import annotations

import re
import threading
from typing import Optional

from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizeGrip,
    QSlider,
    QSpinBox,
    QSystemTrayIcon,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QMenu,
)

from ai_client import GroqAssistant
from audio_capture import SystemAudioBuffer, list_loopback_devices
from config import AppConfig
from context_store import ContextStore, MAX_CONTEXT_CHARS
from hotkeys import (
    DEFAULT_HOTKEY_ANSWER,
    DEFAULT_HOTKEY_CLEAR,
    DEFAULT_HOTKEY_CLICK_THROUGH,
    DEFAULT_HOTKEY_TOGGLE,
    HotkeyManager,
)
from win_stealth import apply_capture_exclusion, is_capture_excluded, set_click_through

_CODE_SEGMENT_RE = re.compile(r"(```[\s\S]*?(?:```|$)|`[^`\n]*`)")


def _sanitize_markdown_for_qt(md: str) -> str:
    """
    Escape '<' outside of fenced/inline code spans so Qt's Markdown renderer
    does not swallow generics like Partial<T> or Pick<T, K> as raw HTML tags.
    """
    parts = _CODE_SEGMENT_RE.split(md)
    for i in range(0, len(parts), 2):  # even indices are non-code prose
        parts[i] = parts[i].replace("<", "&lt;")
    return "".join(parts)

DARK_STYLESHEET = """
QWidget#RootContainer {
    background-color: rgba(18, 20, 26, 238);
    border: 1px solid rgba(255, 255, 255, 35);
    border-radius: 12px;
}
QWidget#HeaderBar {
    background-color: rgba(28, 31, 40, 240);
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 22);
}
QLabel {
    color: #E6E9F0;
    font-family: 'Segoe UI', sans-serif;
}
QPushButton {
    background-color: rgba(255, 255, 255, 14);
    color: #DCE1EB;
    border: 1px solid rgba(255, 255, 255, 20);
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 12px;
    font-family: 'Segoe UI', sans-serif;
}
QPushButton:hover {
    background-color: rgba(255, 255, 255, 28);
}
QPushButton:checked {
    background-color: rgba(74, 158, 255, 90);
    border-color: rgba(74, 158, 255, 180);
    color: #FFFFFF;
}
QPushButton#PrimaryBtn {
    background-color: rgba(56, 139, 253, 200);
    border: 1px solid rgba(110, 175, 255, 180);
    color: #FFFFFF;
    font-weight: 600;
    padding: 6px 12px;
}
QPushButton#PrimaryBtn:hover {
    background-color: rgba(74, 158, 255, 230);
}
QPushButton#PrimaryBtn:disabled {
    background-color: rgba(80, 90, 110, 120);
    color: rgba(255, 255, 255, 120);
}
QTextBrowser {
    background-color: transparent;
    color: #ECEFF4;
    border: none;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13.5px;
    selection-background-color: rgba(74, 158, 255, 120);
}
QLineEdit {
    background-color: rgba(255, 255, 255, 14);
    color: #F3F5FA;
    border: 1px solid rgba(255, 255, 255, 28);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    font-family: 'Segoe UI', sans-serif;
}
QLineEdit:focus {
    border-color: rgba(74, 158, 255, 190);
}
QScrollBar:vertical {
    background: transparent;
    width: 7px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 45);
    border-radius: 3px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowTitle("Настройки Interview Helper")
        self.setMinimumWidth(440)
        self.setStyleSheet(
            """
            QDialog { background-color: #16181F; color: #E6E9F0; font-family: 'Segoe UI'; }
            QLabel { color: #DCE1EB; font-size: 12.5px; }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #222632; color: #F3F5FA;
                border: 1px solid #3A4052; border-radius: 6px; padding: 5px 8px;
            }
            QCheckBox { color: #DCE1EB; }
            QPushButton {
                background-color: #2B3040; color: #FFFFFF;
                border: 1px solid #3E455B; border-radius: 6px; padding: 6px 14px;
            }
            QPushButton:hover { background-color: #3A4156; }
            """
        )

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        self.api_key_edit = QLineEdit(config.api_key)
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.api_key_edit.setPlaceholderText("gsk_... (console.groq.com/keys)")
        form.addRow("Groq API Key:", self.api_key_edit)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems([
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
        ])
        self.model_combo.setCurrentText(config.model)
        form.addRow("Модель:", self.model_combo)

        self.role_edit = QLineEdit(config.user_role)
        self.role_edit.setPlaceholderText("Например: Python Backend разработчик")
        form.addRow("Твоя роль:", self.role_edit)

        device_row = QHBoxLayout()
        device_row.setSpacing(6)
        self.device_combo = QComboBox()
        self._populate_devices(config.audio_device)
        device_row.addWidget(self.device_combo, stretch=1)

        refresh_dev_btn = QPushButton("↻")
        refresh_dev_btn.setFixedWidth(32)
        refresh_dev_btn.setToolTip("Обновить список аудиоустройств")
        refresh_dev_btn.clicked.connect(
            lambda: self._populate_devices(self.device_combo.currentData() or "")
        )
        device_row.addWidget(refresh_dev_btn)
        form.addRow("Источник звука:", device_row)

        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(10, 120)
        self.buffer_spin.setSuffix(" сек")
        self.buffer_spin.setValue(config.buffer_seconds)
        form.addRow("Буфер звука:", self.buffer_spin)

        self.search_check = QCheckBox("Веб-поиск (browser_search для gpt-oss)")
        self.search_check.setChecked(config.use_web_search)
        form.addRow("", self.search_check)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(35, 100)
        self.opacity_slider.setValue(int(config.opacity * 100))
        form.addRow("Прозрачность:", self.opacity_slider)

        hk_sep = QLabel("Горячие клавиши (нажми на поле и задай комбинацию):")
        hk_sep.setStyleSheet("color: #93C5FD; font-weight: 600; margin-top: 6px;")
        form.addRow(hk_sep)

        self.hk_answer_edit = self._make_hotkey_edit(config.hotkey_answer)
        form.addRow("Ответить:", self.hk_answer_edit)

        self.hk_toggle_edit = self._make_hotkey_edit(config.hotkey_toggle)
        form.addRow("Скрыть / показать:", self.hk_toggle_edit)

        self.hk_clear_edit = self._make_hotkey_edit(config.hotkey_clear)
        form.addRow("Очистить:", self.hk_clear_edit)

        self.hk_clickthrough_edit = self._make_hotkey_edit(config.hotkey_click_through)
        form.addRow("Клики сквозь:", self.hk_clickthrough_edit)

        reset_hk_btn = QPushButton("Сбросить клавиши по умолчанию")
        reset_hk_btn.clicked.connect(self._reset_hotkeys)
        form.addRow("", reset_hk_btn)

        layout.addLayout(form)

        hint = QLabel(
            "Ключ создаётся бесплатно на console.groq.com/keys. Речь распознаётся через whisper-large-v3-turbo."
        )
        hint.setStyleSheet("color: #8E97AB; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _make_hotkey_edit(seq_str: str) -> QKeySequenceEdit:
        edit = QKeySequenceEdit(QKeySequence(seq_str))
        edit.setMaximumSequenceLength(1)
        edit.setStyleSheet(
            "QKeySequenceEdit, QKeySequenceEdit QLineEdit { "
            "background-color: #222632; color: #F3F5FA; "
            "border: 1px solid #3A4052; border-radius: 6px; padding: 4px 8px; }"
        )
        return edit

    def _reset_hotkeys(self) -> None:
        self.hk_answer_edit.setKeySequence(QKeySequence(DEFAULT_HOTKEY_ANSWER))
        self.hk_toggle_edit.setKeySequence(QKeySequence(DEFAULT_HOTKEY_TOGGLE))
        self.hk_clear_edit.setKeySequence(QKeySequence(DEFAULT_HOTKEY_CLEAR))
        self.hk_clickthrough_edit.setKeySequence(QKeySequence(DEFAULT_HOTKEY_CLICK_THROUGH))

    def _populate_devices(self, selected_device: str) -> None:
        self.device_combo.clear()
        self.device_combo.addItem("По умолчанию (системный вывод)", "")
        devices = list_loopback_devices()
        for dev_name in devices:
            self.device_combo.addItem(dev_name, dev_name)
        if selected_device:
            idx = self.device_combo.findData(selected_device)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
            else:
                self.device_combo.addItem(f"{selected_device} (не найдено)", selected_device)
                self.device_combo.setCurrentIndex(self.device_combo.count() - 1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        apply_capture_exclusion(int(self.winId()))
        self.raise_()
        self.activateWindow()

    def apply_to(self, config: AppConfig) -> None:
        config.api_key = self.api_key_edit.text().strip()
        config.model = self.model_combo.currentText().strip() or "openai/gpt-oss-120b"
        config.user_role = self.role_edit.text().strip() or "Software Engineer"
        config.audio_device = str(self.device_combo.currentData() or "").strip()
        config.buffer_seconds = self.buffer_spin.value()
        config.use_web_search = self.search_check.isChecked()
        config.opacity = self.opacity_slider.value() / 100.0

        def _seq(edit: QKeySequenceEdit, fallback: str) -> str:
            text = edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText).strip()
            return text or fallback

        config.hotkey_answer = _seq(self.hk_answer_edit, DEFAULT_HOTKEY_ANSWER)
        config.hotkey_toggle = _seq(self.hk_toggle_edit, DEFAULT_HOTKEY_TOGGLE)
        config.hotkey_clear = _seq(self.hk_clear_edit, DEFAULT_HOTKEY_CLEAR)
        config.hotkey_click_through = _seq(self.hk_clickthrough_edit, DEFAULT_HOTKEY_CLICK_THROUGH)


class ContextFilesDialog(QDialog):
    def __init__(self, store: ContextStore, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.store = store
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowTitle("Файлы контекста (лабы / код)")
        self.setMinimumSize(480, 340)
        self.setStyleSheet(
            """
            QDialog { background-color: #16181F; color: #E6E9F0; font-family: 'Segoe UI'; }
            QLabel { color: #DCE1EB; font-size: 12px; }
            QListWidget {
                background-color: #1E222D; color: #ECEFF4;
                border: 1px solid #3A4052; border-radius: 6px; padding: 4px;
                font-size: 12.5px;
            }
            QListWidget::item { padding: 4px 6px; border-radius: 4px; }
            QListWidget::item:selected { background-color: rgba(56, 139, 253, 130); }
            QPushButton {
                background-color: #2B3040; color: #FFFFFF;
                border: 1px solid #3E455B; border-radius: 6px; padding: 6px 12px;
            }
            QPushButton:hover { background-color: #3A4156; }
            """
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        btn_row = QHBoxLayout()
        self.add_files_btn = QPushButton("+ Файлы")
        self.add_folder_btn = QPushButton("+ Папку")
        self.remove_btn = QPushButton("Удалить")
        self.clear_btn = QPushButton("Очистить")
        btn_row.addWidget(self.add_files_btn)
        btn_row.addWidget(self.add_folder_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.remove_btn)
        btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.list_widget, stretch=1)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #8E97AB; font-size: 11.5px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.accept)
        close_box.button(QDialogButtonBox.StandardButton.Close).setText("Готово")
        layout.addWidget(close_box)

        self.add_files_btn.clicked.connect(self._on_add_files)
        self.add_folder_btn.clicked.connect(self._on_add_folder)
        self.remove_btn.clicked.connect(self._on_remove_selected)
        self.clear_btn.clicked.connect(self._on_clear)

        self._refresh_list()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        apply_capture_exclusion(int(self.winId()))
        self.raise_()
        self.activateWindow()

    def _refresh_list(self, note: str = "") -> None:
        self.list_widget.clear()
        for lf in self.store.files:
            kb = lf.chars / 1024.0
            item = QListWidgetItem(f"{lf.name}  ({kb:.1f} КБ)  —  {lf.path}")
            item.setData(Qt.ItemDataRole.UserRole, lf.path)
            self.list_widget.addItem(item)

        total = self.store.total_chars
        mode = (
            "все файлы передаются целиком"
            if total <= MAX_CONTEXT_CHARS
            else f"автовыбор релевантных фрагментов до {MAX_CONTEXT_CHARS // 1000}K симв. (чтобы не превышать лимит Groq)"
        )
        summary = f"Файлов: {self.store.file_count} | Объём: {total:,} симв. (~{total // 4:,} токенов) — {mode}."
        if note:
            summary = f"{note}\n{summary}"
        self.info_label.setText(summary)

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Выбери файлы лаб / кода",
            "",
            "Код и текст (*.py *.ts *.tsx *.js *.jsx *.cpp *.c *.h *.cs *.java *.go *.rs *.sql *.html *.css *.json *.md *.txt *.ipynb);;Все файлы (*.*)",
        )
        if paths:
            added, errs = self.store.add_files(paths)
            note = f"Добавлено: {added}" + (f" | Пропущено: {'; '.join(errs[:2])}" if errs else "")
            self._refresh_list(note)

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выбери папку с лабами")
        if folder:
            added, errs = self.store.add_folder(folder)
            note = f"Добавлено из папки: {added}" + (f" | Ошибки: {'; '.join(errs[:2])}" if errs else "")
            self._refresh_list(note)

    def _on_remove_selected(self) -> None:
        selected = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.list_widget.selectedItems()
        ]
        if selected:
            self.store.remove_paths(selected)
            self._refresh_list()

    def _on_clear(self) -> None:
        self.store.clear()
        self._refresh_list("Контекст очищен.")


class OverlayWindow(QWidget):
    # Cross-thread signals
    sig_stream_start = pyqtSignal(str)
    sig_stream_chunk = pyqtSignal(str)
    sig_stream_done = pyqtSignal()
    sig_stream_error = pyqtSignal(str)

    # Global hotkey triggers
    sig_hotkey_answer = pyqtSignal()
    sig_hotkey_toggle_visible = pyqtSignal()
    sig_hotkey_clear = pyqtSignal()
    sig_hotkey_click_through = pyqtSignal()

    def __init__(
        self,
        config: AppConfig,
        audio_buffer: SystemAudioBuffer,
        assistant: GroqAssistant,
        context_store: ContextStore,
        hotkey_manager: Optional[HotkeyManager] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.audio_buffer = audio_buffer
        self.assistant = assistant
        self.context_store = context_store
        self.hotkey_manager = hotkey_manager

        self._drag_pos: Optional[QPoint] = None
        self._busy = False
        self._click_through = False
        self._current_markdown = ""
        self._stealth_ok = False

        self._init_window_flags()
        self._build_ui()
        self._connect_signals()
        self._setup_tray()

        self.resize(540, 420)
        self._position_top_right()
        self.setWindowOpacity(self.config.opacity)

        # Audio meter refresh timer
        self._meter_timer = QTimer(self)
        self._meter_timer.timeout.connect(self._update_audio_status)
        self._meter_timer.start(180)

        self._render_welcome()
        self._update_files_badge()
        self._apply_hotkeys()

    def _apply_hotkeys(self) -> None:
        self.answer_btn.setText(f"Ответить ({self.config.hotkey_answer})")
        self.hide_btn.setToolTip(f"Свернуть ({self.config.hotkey_toggle})")
        self.clear_btn.setToolTip(f"Очистить ответы и контекст ({self.config.hotkey_clear})")
        self.clickthrough_btn.setToolTip(
            f"Пропускать клики мыши сквозь окно ({self.config.hotkey_click_through})"
        )
        self.tray.setToolTip(f"Interview Helper ({self.config.hotkey_toggle} — показать/скрыть)")
        if hasattr(self, "act_tray_toggle"):
            self.act_tray_toggle.setText(f"Показать / Скрыть ({self.config.hotkey_toggle})")

        if self.hotkey_manager is not None:
            self.hotkey_manager.register(
                [
                    (self.config.hotkey_answer, lambda: self.sig_hotkey_answer.emit()),
                    (self.config.hotkey_toggle, lambda: self.sig_hotkey_toggle_visible.emit()),
                    (self.config.hotkey_clear, lambda: self.sig_hotkey_clear.emit()),
                    (self.config.hotkey_click_through, lambda: self.sig_hotkey_click_through.emit()),
                ]
            )

    def _update_files_badge(self) -> None:
        count = self.context_store.file_count
        if count > 0:
            self.files_btn.setText(f"Файлы ({count})")
            self.files_btn.setChecked(True)
        else:
            self.files_btn.setText("Файлы")
            self.files_btn.setChecked(False)

    def open_context_files(self) -> None:
        was_click_through = self._click_through
        if was_click_through:
            self.clickthrough_btn.setChecked(False)

        dlg = ContextFilesDialog(self.context_store, self)
        dlg.exec()
        self._update_files_badge()

        if was_click_through:
            self.clickthrough_btn.setChecked(True)

    def _init_window_flags(self) -> None:
        # Tool + Frameless + StaysOnTop keeps it off the taskbar & Alt-Tab and above all windows
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowTitle("Helper")

    def _position_top_right(self) -> None:
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = geo.right() - self.width() - 24
            y = geo.top() + 24
            self.move(x, y)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.root_container = QWidget()
        self.root_container.setObjectName("RootContainer")
        self.root_container.setStyleSheet(DARK_STYLESHEET)
        outer.addWidget(self.root_container)

        root_layout = QVBoxLayout(self.root_container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Header bar (draggable)
        self.header = QWidget()
        self.header.setObjectName("HeaderBar")
        self.header.setFixedHeight(40)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(12, 4, 8, 4)
        header_layout.setSpacing(6)

        self.audio_dot = QLabel("●")
        self.audio_dot.setStyleSheet("color: #6B7280; font-size: 13px;")
        self.audio_dot.setToolTip("Статус захвата системного звука")
        header_layout.addWidget(self.audio_dot)

        self.status_label = QLabel("Слушаю звук...")
        self.status_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #D8DEE9;")
        header_layout.addWidget(self.status_label)

        self.stealth_badge = QLabel("Скрыт")
        self.stealth_badge.setStyleSheet(
            "background-color: rgba(16, 185, 129, 45); color: #6EE7B7; "
            "border: 1px solid rgba(16, 185, 129, 120); border-radius: 4px; "
            "padding: 1px 6px; font-size: 10.5px; font-weight: 600;"
        )
        self.stealth_badge.setToolTip("Окно исключено из захвата экрана (WDA_EXCLUDEFROMCAPTURE)")
        header_layout.addWidget(self.stealth_badge)

        header_layout.addStretch()

        self.files_btn = QPushButton("Файлы")
        self.files_btn.setCheckable(True)
        self.files_btn.setToolTip("Прикрепить файлы лаб / кода в контекст")
        header_layout.addWidget(self.files_btn)

        self.search_btn = QPushButton("Поиск")
        self.search_btn.setCheckable(True)
        self.search_btn.setChecked(self.config.use_web_search)
        self.search_btn.setToolTip("Вкл/выкл веб-поиск (browser_search)")
        header_layout.addWidget(self.search_btn)

        self.clickthrough_btn = QPushButton("Сквозь")
        self.clickthrough_btn.setCheckable(True)
        self.clickthrough_btn.setToolTip("Пропускать клики мыши сквозь окно (Ctrl+Shift+T)")
        header_layout.addWidget(self.clickthrough_btn)

        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.setToolTip("Очистить ответы и контекст (Ctrl+Shift+C)")
        header_layout.addWidget(self.clear_btn)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedWidth(30)
        self.settings_btn.setToolTip("Настройки API и буфера")
        header_layout.addWidget(self.settings_btn)

        self.hide_btn = QPushButton("—")
        self.hide_btn.setFixedWidth(28)
        self.hide_btn.setToolTip("Свернуть (Ctrl+Shift+H)")
        header_layout.addWidget(self.hide_btn)

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedWidth(28)
        self.close_btn.setToolTip("Выход")
        header_layout.addWidget(self.close_btn)

        root_layout.addWidget(self.header)

        # Output area
        self.output = QTextBrowser()
        self.output.setOpenExternalLinks(True)
        self.output.document().setDefaultStyleSheet(
            """
            body { color: #ECEFF4; font-family: 'Segoe UI', sans-serif; font-size: 13.5px; line-height: 1.45; }
            p { margin: 5px 0; }
            ul, ol { margin: 4px 0 6px 0; padding-left: 18px; }
            li { margin: 3px 0; }
            strong { color: #FFFFFF; }
            code { background-color: rgba(255,255,255,0.12); padding: 1px 4px; border-radius: 3px; font-family: 'Consolas', monospace; font-size: 12.5px; color: #93C5FD; }
            pre { background-color: rgba(0,0,0,0.45); padding: 8px; border-radius: 6px; font-family: 'Consolas', monospace; font-size: 12px; color: #E5E7EB; }
            a { color: #60A5FA; }
            """
        )
        content_wrap = QVBoxLayout()
        content_wrap.setContentsMargins(12, 8, 12, 6)
        content_wrap.addWidget(self.output)
        root_layout.addLayout(content_wrap, stretch=1)

        # Bottom controls
        footer_line = QFrame()
        footer_line.setFrameShape(QFrame.Shape.HLine)
        footer_line.setStyleSheet("color: rgba(255, 255, 255, 18);")
        root_layout.addWidget(footer_line)

        footer = QHBoxLayout()
        footer.setContentsMargins(10, 8, 6, 8)
        footer.setSpacing(6)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Дописать вопрос вручную (Enter)...")
        footer.addWidget(self.input_edit, stretch=1)

        self.answer_btn = QPushButton("Ответить (Ctrl+Shift+Space)")
        self.answer_btn.setObjectName("PrimaryBtn")
        footer.addWidget(self.answer_btn)

        self.size_grip = QSizeGrip(self.root_container)
        self.size_grip.setFixedSize(14, 14)
        footer.addWidget(self.size_grip, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        root_layout.addLayout(footer)

    def _setup_tray(self) -> None:
        self.tray = QSystemTrayIcon(self)
        icon = self.style().standardIcon(self.style().StandardPixmap.SP_ComputerIcon)
        self.tray.setIcon(icon)
        self.tray.setToolTip("Interview Helper (Ctrl+Shift+H — показать/скрыть)")

        menu = QMenu()
        self.act_tray_toggle = QAction(f"Показать / Скрыть ({self.config.hotkey_toggle})", self)
        self.act_tray_toggle.triggered.connect(self.toggle_visibility)
        menu.addAction(self.act_tray_toggle)

        act_settings = QAction("Настройки...", self)
        act_settings.triggered.connect(self.open_settings)
        menu.addAction(act_settings)

        menu.addSeparator()
        act_quit = QAction("Выход", self)
        act_quit.triggered.connect(QApplication.quit)
        menu.addAction(act_quit)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self.toggle_visibility()
            if reason == QSystemTrayIcon.ActivationReason.Trigger
            else None
        )
        self.tray.show()

    def _connect_signals(self) -> None:
        self.files_btn.clicked.connect(self.open_context_files)
        self.answer_btn.clicked.connect(self.trigger_audio_answer)
        self.input_edit.returnPressed.connect(self.trigger_text_answer)
        self.clear_btn.clicked.connect(self.clear_all)
        self.settings_btn.clicked.connect(self.open_settings)
        self.hide_btn.clicked.connect(self.hide)
        self.close_btn.clicked.connect(QApplication.quit)
        self.search_btn.toggled.connect(self._on_search_toggled)
        self.clickthrough_btn.toggled.connect(self._on_click_through_toggled)

        self.sig_stream_start.connect(self._on_stream_start)
        self.sig_stream_chunk.connect(self._on_stream_chunk)
        self.sig_stream_done.connect(self._on_stream_done)
        self.sig_stream_error.connect(self._on_stream_error)

        self.sig_hotkey_answer.connect(self.trigger_audio_answer)
        self.sig_hotkey_toggle_visible.connect(self.toggle_visibility)
        self.sig_hotkey_clear.connect(self.clear_all)
        self.sig_hotkey_click_through.connect(self.toggle_click_through)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        hwnd = int(self.winId())
        self._stealth_ok = apply_capture_exclusion(hwnd) and is_capture_excluded(hwnd)
        if self._stealth_ok:
            self.stealth_badge.setText("Скрыт от демки")
            self.stealth_badge.setStyleSheet(
                "background-color: rgba(16, 185, 129, 45); color: #6EE7B7; "
                "border: 1px solid rgba(16, 185, 129, 120); border-radius: 4px; "
                "padding: 1px 6px; font-size: 10.5px; font-weight: 600;"
            )
        else:
            self.stealth_badge.setText("Видим!")
            self.stealth_badge.setStyleSheet(
                "background-color: rgba(239, 68, 68, 60); color: #FCA5A5; "
                "border: 1px solid rgba(239, 68, 68, 150); border-radius: 4px; "
                "padding: 1px 6px; font-size: 10.5px; font-weight: 600;"
            )

    def _render_welcome(self) -> None:
        key_notice = (
            ""
            if self.config.api_key
            else "\n> **Нужен ключ:** нажми **⚙** сверху и вставь бесплатный `GROQ_API_KEY` с [console.groq.com/keys](https://console.groq.com/keys).\n"
        )
        self._current_markdown = (
            f"### Готов к работе\n"
            f"{key_notice}\n"
            f"- **{self.config.hotkey_answer}** — ответить на последний вопрос из звука (буфер {self.config.buffer_seconds} сек)\n"
            f"- **{self.config.hotkey_toggle}** — скрыть / показать оверлей\n"
            f"- **{self.config.hotkey_clear}** — очистить экран и контекст\n"
            f"- **{self.config.hotkey_click_through}** — режим кликов сквозь окно\n\n"
            f"*Окно не захватывается при демонстрации экрана (Zoom, Meet, Teams, Discord, OBS).*"
        )
        self.output.setMarkdown(self._current_markdown)

    def _update_audio_status(self) -> None:
        if self.audio_buffer.last_error:
            self.audio_dot.setStyleSheet("color: #EF4444; font-size: 13px;")
            self.audio_dot.setToolTip(f"Ошибка аудио: {self.audio_buffer.last_error}")
            if not self._busy:
                self.status_label.setText("Нет аудио")
            return

        rms = self.audio_buffer.current_rms
        self.audio_dot.setToolTip(f"Устройство: {self.audio_buffer.device_name}")
        if rms > 0.004:
            self.audio_dot.setStyleSheet("color: #34D399; font-size: 13px;")  # active sound
        else:
            self.audio_dot.setStyleSheet("color: #60A5FA; font-size: 13px;")  # listening idle

        if not self._busy:
            short_dev = self.audio_buffer.device_name
            if len(short_dev) > 22:
                short_dev = short_dev[:20] + "…"
            self.status_label.setText(short_dev)

    def _on_search_toggled(self, checked: bool) -> None:
        self.config.use_web_search = checked
        self.assistant.update_config(self.config)
        self.config.save()

    def _on_click_through_toggled(self, checked: bool) -> None:
        self._click_through = checked
        set_click_through(int(self.winId()), checked)

    def toggle_click_through(self) -> None:
        self.clickthrough_btn.setChecked(not self.clickthrough_btn.isChecked())

    def toggle_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()

    def clear_all(self) -> None:
        self.assistant.clear_history()
        self.audio_buffer.clear()
        self._render_welcome()

    def open_settings(self) -> None:
        was_click_through = self._click_through
        if was_click_through:
            self.clickthrough_btn.setChecked(False)

        # Pause global hotkeys while recording new keybindings
        if self.hotkey_manager is not None:
            self.hotkey_manager.stop()

        dlg = SettingsDialog(self.config, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            dlg.apply_to(self.config)
            self.config.save()
            self.assistant.update_config(self.config)
            self.audio_buffer.set_buffer_seconds(self.config.buffer_seconds)
            self.audio_buffer.set_device(self.config.audio_device)
            self.search_btn.setChecked(self.config.use_web_search)
            self.setWindowOpacity(self.config.opacity)
            if not self._busy:
                self._render_welcome()

        self._apply_hotkeys()

        if was_click_through:
            self.clickthrough_btn.setChecked(True)

    def trigger_audio_answer(self) -> None:
        if self._busy:
            return
        if not self.isVisible():
            self.show()

        wav_bytes, duration = self.audio_buffer.get_wav_bytes()
        extra_note = self.input_edit.text().strip()
        if extra_note:
            self.input_edit.clear()

        if not wav_bytes:
            self.output.setMarkdown(
                "**В буфере тишина.**\n\nУбедись, что звук собеседника идёт на устройство вывода по умолчанию, и попробуй ещё раз."
            )
            return

        self._start_worker(
            header=f"Анализирую последние {duration:.0f} сек звука...",
            fn=lambda: self.assistant.ask_from_audio_stream(wav_bytes, extra_note=extra_note),
        )

    def trigger_text_answer(self) -> None:
        if self._busy:
            return
        question = self.input_edit.text().strip()
        if not question:
            return
        self.input_edit.clear()
        self._start_worker(
            header="Думаю...",
            fn=lambda: self.assistant.ask_from_text_stream(question),
        )

    def _start_worker(self, header: str, fn) -> None:
        self._busy = True
        self.answer_btn.setEnabled(False)
        self.sig_stream_start.emit(header)

        def _run() -> None:
            try:
                for chunk in fn():
                    self.sig_stream_chunk.emit(chunk)
                self.sig_stream_done.emit()
            except Exception as exc:
                self.sig_stream_error.emit(str(exc))

        threading.Thread(target=_run, name="GeminiWorker", daemon=True).start()

    def _on_stream_start(self, status_text: str) -> None:
        self.status_label.setText(status_text)
        self._current_markdown = ""
        self.output.setMarkdown(f"*{status_text}*")

    def _on_stream_chunk(self, chunk: str) -> None:
        self._current_markdown += chunk
        self.output.setMarkdown(_sanitize_markdown_for_qt(self._current_markdown))
        sb = self.output.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_stream_done(self) -> None:
        self._busy = False
        self.answer_btn.setEnabled(True)
        if not self._current_markdown.strip():
            self.output.setMarkdown("*Пустой ответ от модели.*")

    def _on_stream_error(self, err: str) -> None:
        self._busy = False
        self.answer_btn.setEnabled(True)
        self.output.setMarkdown(f"**Ошибка:**\n\n`{err}`")

    # Drag window by header bar
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.header.geometry().contains(event.pos()):
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
