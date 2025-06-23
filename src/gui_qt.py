"""
Современный GUI на PyQt6 для ChatYandexGPT
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QAction
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QProgressBar, QMessageBox,
    QSplitter, QDialog
)


class LiveTranscriptionWidget(QWidget):
    """Виджет для live-транскрипции"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.text_edit = None
        self.title_label = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(2)  # Уменьшаем промежутки между элементами
        layout.setContentsMargins(10, 5, 10, 5)  # Добавляем отступы

        # Заголовок
        self.title_label = QLabel("🎯 Распознанный текст:")
        self.title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.title_label)

        # Текстовое поле
        self.text_edit = QTextEdit()
        self.text_edit.setMaximumHeight(60)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Arial", 11))
        layout.addWidget(self.text_edit)

    def update_text(self, text: str, is_final: bool = False):
        """Обновление текста транскрипции"""
        self.text_edit.clear()
        if text:
            self.text_edit.insertPlainText(text)

            if is_final:
                self.title_label.setText("✅ Финальный текст:")
                self.text_edit.setStyleSheet("color: #E0E0E0;")
            else:
                self.title_label.setText("🎯 Распознавание...")
                self.text_edit.setStyleSheet("color: #87CEEB;")
        else:
            self.title_label.setText("🎯 Распознанный текст:")
            self.text_edit.setStyleSheet("color: #B0B0B0;")

    def clear(self):
        """Очистка текста"""
        self.text_edit.clear()
        self.title_label.setText("🎯 Распознанный текст:")
        self.text_edit.setStyleSheet("color: #B0B0B0;")


class VoskStatusWidget(QWidget):
    """Виджет для отображения статуса модели Vosk"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)  # Уменьшаем вертикальные отступы с 5 до 2

        # Иконка статуса
        self.status_icon = QLabel("🔄")
        self.status_icon.setFont(QFont("Arial", 12))
        layout.addWidget(self.status_icon)

        # Текст статуса
        self.status_label = QLabel("Модель Vosk: Загрузка...")
        self.status_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.status_label)

        # Детали
        self.details_label = QLabel("")
        self.details_label.setFont(QFont("Arial", 9))
        self.details_label.setStyleSheet("color: #B0B0B0;")
        layout.addWidget(self.details_label)

        layout.addStretch()

    def update_status(self, status: str, is_loaded: bool = False, details: str = ""):
        """Обновление статуса"""
        if is_loaded:
            self.status_icon.setText("✅")
            self.status_label.setText(f"Модель Vosk: Готово")
            self.status_label.setStyleSheet("color: #4A7C59;")
        elif "Ошибка" in status:
            self.status_icon.setText("❌")
            self.status_label.setText(f"Модель Vosk: {status}")
            self.status_label.setStyleSheet("color: #8B5A5A;")
        else:
            self.status_icon.setText("🔄")
            self.status_label.setText(f"Модель Vosk: {status}")
            self.status_label.setStyleSheet("color: #FFA500;")

        self.details_label.setText(details)


class PromptStatusWidget(QWidget):
    """Виджет для отображения статуса промпта и модели"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)  # Уменьшаем вертикальные отступы с 5 до 2

        # Статус промпта
        self.prompt_label = QLabel("Помощник: Программист-помощник")
        self.prompt_label.setFont(QFont("Arial", 10))
        self.prompt_label.setStyleSheet("color: #B0B0B0; font-style: italic;")
        layout.addWidget(self.prompt_label)

        # Разделитель
        separator = QLabel("|")
        separator.setStyleSheet("color: #6C6C6C; margin: 0 10px;")
        layout.addWidget(separator)

        # Статус модели
        self.model_label = QLabel("Модель: YandexGPT Lite")
        self.model_label.setFont(QFont("Arial", 10))
        self.model_label.setStyleSheet("color: #B0B0B0; font-style: italic;")
        layout.addWidget(self.model_label)

        layout.addStretch()

    def update_prompt(self, prompt_name: str):
        """Обновление статуса промпта"""
        self.prompt_label.setText(f"Помощник: {prompt_name}")

    def update_model(self, model_name: str):
        """Обновление статуса модели"""
        self.model_label.setText(f"Модель: {model_name}")


class RecordingButtonsWidget(QWidget):
    """Виджет с кнопками записи"""

    recording_started = pyqtSignal(str)  # Сигнал начала записи
    recording_stopped = pyqtSignal()  # Сигнал остановки записи
    recording_cancelled = pyqtSignal()  # Сигнал отмены записи

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(3)  # Уменьшаем промежутки между элементами

        # Кнопки записи
        button_layout = QHBoxLayout()

        # Кнопка записи с микрофона
        self.mic_button = QPushButton("🎤 Запись с микрофона")
        self.mic_button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.mic_button.setMinimumHeight(40)
        self.mic_button.clicked.connect(lambda: self.recording_started.emit('mic'))
        button_layout.addWidget(self.mic_button)

        # Кнопка записи с компьютера
        self.computer_button = QPushButton("💻 Запись с компьютера")
        self.computer_button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.computer_button.setMinimumHeight(40)
        self.computer_button.clicked.connect(lambda: self.recording_started.emit('computer'))
        button_layout.addWidget(self.computer_button)

        layout.addLayout(button_layout)

        # Индикатор записи (уменьшаем отступ)
        self.recording_label = QLabel("")
        self.recording_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.recording_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_label.setStyleSheet("color: #8B5A5A;")
        layout.addWidget(self.recording_label)

        # Кнопка отмены (уменьшаем отступ сверху)
        self.cancel_button = QPushButton("❌ Отменить запись")
        self.cancel_button.setFont(QFont("Arial", 10))
        self.cancel_button.setMinimumHeight(35)
        self.cancel_button.clicked.connect(self.recording_cancelled.emit)
        self.cancel_button.setEnabled(False)
        layout.addWidget(self.cancel_button)

        # Применяем стили
        self.apply_styles()

    def apply_styles(self):
        """Применение стилей к кнопкам"""
        button_style = """
        QPushButton {
            background-color: #4C4C4C;
            color: #E0E0E0;
            border: none;
            border-radius: 5px;
            padding: 8px;
        }
        QPushButton:hover {
            background-color: #6C6C6C;
        }
        QPushButton:pressed {
            background-color: #2C2C2C;
        }
        QPushButton:disabled {
            background-color: #3C3C3C;
            color: #808080;
        }
        """

        self.mic_button.setStyleSheet(button_style)
        self.computer_button.setStyleSheet(button_style)
        self.cancel_button.setStyleSheet(button_style.replace("#4C4C4C", "#8B5A5A"))

    def update_recording_status(self, is_recording: bool, device_type: str = None):
        """Обновление статуса записи извне"""
        if is_recording:
            # Обновляем кнопки
            if device_type == 'mic':
                self.mic_button.setText("⏹️ Остановить запись (микрофон)")
                self.mic_button.setStyleSheet("""
                    QPushButton {
                        background-color: #8B5A5A;
                        color: #E0E0E0;
                        border: none;
                        border-radius: 5px;
                        padding: 8px;
                    }
                    QPushButton:hover {
                        background-color: #A07070;
                    }
                """)
                self.computer_button.setEnabled(False)
                # Подключаем сигнал остановки к кнопке микрофона
                try:
                    self.mic_button.clicked.disconnect()
                except TypeError:
                    pass  # Сигнал не был подключен
                self.mic_button.clicked.connect(self.recording_stopped.emit)
            else:
                self.computer_button.setText("⏹️ Остановить запись (компьютер)")
                self.computer_button.setStyleSheet("""
                    QPushButton {
                        background-color: #8B5A5A;
                        color: #E0E0E0;
                        border: none;
                        border-radius: 5px;
                        padding: 8px;
                    }
                    QPushButton:hover {
                        background-color: #A07070;
                    }
                """)
                self.mic_button.setEnabled(False)
                # Подключаем сигнал остановки к кнопке компьютера
                try:
                    self.computer_button.clicked.disconnect()
                except TypeError:
                    pass  # Сигнал не был подключен
                self.computer_button.clicked.connect(self.recording_stopped.emit)

            self.recording_label.setText("🔴 ЗАПИСЬ...")
            self.cancel_button.setEnabled(True)
        else:
            # Возвращаем кнопки в исходное состояние
            self.mic_button.setText("🎤 Запись с микрофона")
            self.computer_button.setText("💻 Запись с компьютера")
            self.mic_button.setEnabled(True)
            self.computer_button.setEnabled(True)

            # Восстанавливаем сигналы начала записи
            try:
                self.mic_button.clicked.disconnect()
            except TypeError:
                pass  # Сигнал не был подключен
            self.mic_button.clicked.connect(lambda: self.recording_started.emit('mic'))

            try:
                self.computer_button.clicked.disconnect()
            except TypeError:
                pass  # Сигнал не был подключен
            self.computer_button.clicked.connect(lambda: self.recording_started.emit('computer'))

            # Применяем обычные стили
            self.apply_styles()

            self.recording_label.setText("")
            self.cancel_button.setEnabled(False)


class ChatWidget(QWidget):
    """Виджет для отображения чата"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Текстовое поле для чата
        self.chat_text = QTextEdit()
        self.chat_text.setReadOnly(True)
        self.chat_text.setFont(QFont("Consolas", 10))
        self.chat_text.setMinimumHeight(400)
        self.chat_text.setStyleSheet("""
            QTextEdit {
                background-color: #3C3C3C;
                color: #E0E0E0;
                border: none;
                border-radius: 5px;
                padding: 10px;
                line-height: 1.4;
            }
            QTextEdit QScrollBar:vertical {
                background-color: #2C2C2C;
                width: 12px;
                border-radius: 6px;
            }
            QTextEdit QScrollBar::handle:vertical {
                background-color: #6C6C6C;
                border-radius: 6px;
                min-height: 20px;
            }
            QTextEdit QScrollBar::handle:vertical:hover {
                background-color: #8C8C8C;
            }
        """)
        layout.addWidget(self.chat_text)

    def add_message(self, user_text: str, ai_response: str):
        """Добавление сообщения в чат с форматированием кода"""
        cursor = self.chat_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)

        # Добавляем сообщение пользователя
        cursor.insertHtml(f'<p style="color: #E0E0E0; margin: 5px 0;"><b>👤 User:</b> {user_text}</p>')

        # Форматируем ответ AI с выделением кода
        formatted_response = self._format_code_blocks(ai_response)
        cursor.insertHtml(f'<p style="color: #E0E0E0; margin: 5px 0;"><b>🤖 AI:</b> {formatted_response}</p>')
        cursor.insertHtml('<hr style="margin: 10px 0; border: 1px solid #6C6C6C;">')

        # Прокручиваем вниз
        self.chat_text.setTextCursor(cursor)
        self.chat_text.ensureCursorVisible()

    def _format_code_blocks(self, text: str) -> str:
        """Форматирование блоков кода в тексте"""
        import re

        # Заменяем переносы строк на HTML
        text = text.replace('\n', '<br>')

        # Находим блоки кода (между тройными обратными кавычками)
        code_block_pattern = r'```(\w+)?\s*\n(.*?)\n```'

        def replace_code_block(match):
            language = match.group(1) or 'text'
            code_content = match.group(2)

            # Применяем базовую подсветку синтаксиса
            highlighted_code = self._highlight_syntax(code_content, language)

            # Форматируем код с отступами и подсветкой
            formatted_code = f'<div style="background-color: #1E1E1E; border: 1px solid #6C6C6C; border-radius: 5px; padding: 10px; margin: 10px 0; font-family: Consolas, monospace; font-size: 11px; color: #D4D4D4; white-space: pre-wrap;">'
            formatted_code += f'<div style="color: #569CD6; font-weight: bold; margin-bottom: 5px;">{language.upper()}</div>'
            formatted_code += f'<div style="color: #D4D4D4;">{highlighted_code}</div>'
            formatted_code += '</div>'

            return formatted_code

        # Применяем форматирование к блокам кода
        text = re.sub(code_block_pattern, replace_code_block, text, flags=re.DOTALL)

        # Находим встроенный код (между одинарными обратными кавычками)
        inline_code_pattern = r'`([^`]+)`'

        def replace_inline_code(match):
            code_content = match.group(1)
            return f'<span style="background-color: #3C3C3C; color: #D4D4D4; font-family: Consolas, monospace; padding: 2px 4px; border-radius: 3px; font-size: 11px;">{code_content}</span>'

        # Применяем форматирование к встроенному коду
        text = re.sub(inline_code_pattern, replace_inline_code, text)

        return text

    def _highlight_syntax(self, code: str, language: str) -> str:
        """Базовая подсветка синтаксиса для популярных языков"""
        import re

        # Цвета для подсветки
        colors = {
            'keyword': '#569CD6',  # Синий для ключевых слов
            'string': '#CE9178',  # Оранжевый для строк
            'comment': '#6A9955',  # Зеленый для комментариев
            'number': '#B5CEA8',  # Светло-зеленый для чисел
            'function': '#DCDCAA',  # Желтый для функций
            'default': '#D4D4D4'  # Белый для обычного текста
        }

        # Паттерны для разных языков
        patterns = {
            'python': {
                'keyword': r'\b(def|class|import|from|as|if|else|elif|for|while|try|except|finally|with|return|True|False|None|and|or|not|in|is|lambda|yield|async|await)\b',
                'string': r'"[^"]*"|\'[^\']*\'',
                'comment': r'#.*$',
                'function': r'\b\w+(?=\()',
                'number': r'\b\d+\.?\d*\b',
            },
            'javascript': {
                'keyword': r'\b(function|var|let|const|if|else|for|while|try|catch|finally|return|class|extends|import|export|async|await|new|this|super)\b',
                'string': r'"[^"]*"|\'[^\']*\'|`[^`]*`',
                'comment': r'//.*$|/\*.*?\*/',
                'function': r'\b\w+(?=\()',
                'number': r'\b\d+\.?\d*\b',
            },
            'java': {
                'keyword': r'\b(public|private|protected|static|final|class|interface|extends|implements|if|else|for|while|try|catch|finally|return|new|this|super|import|package)\b',
                'string': r'"[^"]*"',
                'comment': r'//.*$|/\*.*?\*/',
                'function': r'\b\w+(?=\()',
                'number': r'\b\d+\.?\d*\b',
            },
            'cpp': {
                'keyword': r'\b(int|float|double|char|bool|void|class|struct|enum|public|private|protected|static|const|virtual|template|namespace|using|include|#include|#define|#ifdef|#endif)\b',
                'string': r'"[^"]*"',
                'comment': r'//.*$|/\*.*?\*/',
                'function': r'\b\w+(?=\()',
                'number': r'\b\d+\.?\d*\b',
            }
        }

        # Если язык не поддерживается, возвращаем как есть
        if language.lower() not in patterns:
            return code

        lang_patterns = patterns[language.lower()]
        highlighted_code = code

        # Применяем подсветку в обратном порядке (от длинных к коротким)
        for token_type, pattern in lang_patterns.items():
            color = colors.get(token_type, colors['default'])

            def replace_token(match):
                return f'<span style="color: {color};">{match.group(0)}</span>'

            highlighted_code = re.sub(pattern, replace_token, highlighted_code, flags=re.MULTILINE)

        return highlighted_code

    def clear(self):
        """Очистка чата"""
        self.chat_text.clear()


class TextInputWidget(QWidget):
    """Виджет для ввода текста"""

    text_sent = pyqtSignal(str)  # Сигнал отправки текста
    clear_button_clicked = pyqtSignal()  # Сигнал нажатия на кнопку очистки

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Поле ввода текста
        self.text_edit = QTextEdit()
        self.text_edit.setMaximumHeight(80)
        self.text_edit.setFont(QFont("Arial", 11))
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #3C3C3C;
                color: #E0E0E0;
                border: none;
                border-radius: 5px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.text_edit)

        # Кнопки управления
        button_layout = QHBoxLayout()

        # Кнопка отправки
        self.send_button = QPushButton("📤 Отправить текст")
        self.send_button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.send_button.setMinimumHeight(35)
        self.send_button.clicked.connect(self.send_text)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4A7C59;
                color: #E0E0E0;
                border: none;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #5A8C69;
            }
            QPushButton:pressed {
                background-color: #3A6C49;
            }
        """)
        button_layout.addWidget(self.send_button)

        # Кнопка очистки
        self.clear_button = QPushButton("🗑️ Очистить")
        self.clear_button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.clear_button.setMinimumHeight(35)
        self.clear_button.clicked.connect(self.clear_button_clicked.emit)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #8B7355;
                color: #E0E0E0;
                border: none;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #9B8365;
            }
            QPushButton:pressed {
                background-color: #7B6345;
            }
        """)
        button_layout.addWidget(self.clear_button)

        layout.addLayout(button_layout)

    def send_text(self):
        """Отправка текста"""
        text = self.text_edit.toPlainText().strip()
        if text:
            self.text_sent.emit(text)
            self.text_edit.clear()

    def clear(self):
        """Очистка поля ввода"""
        self.text_edit.clear()


class ProgressDialog(QDialog):
    """Диалог прогресса"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Обработка")
        self.setFixedSize(300, 100)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Метка сообщения
        self.message_label = QLabel("Обработка...")
        self.message_label.setFont(QFont("Arial", 12))
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message_label)

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Бесконечный прогресс
        layout.addWidget(self.progress_bar)

    def update_message(self, message: str):
        """Обновление сообщения"""
        self.message_label.setText(message)


class ModernGUI(QMainWindow):
    """Современный GUI на PyQt6"""

    # Сигналы для безопасного обновления GUI из других потоков
    update_chat_signal = pyqtSignal(str, str)  # user_text, ai_response
    update_live_transcription_signal = pyqtSignal(str, bool)  # text, is_final
    clear_live_transcription_signal = pyqtSignal()
    clear_chat_signal = pyqtSignal()
    clear_text_input_signal = pyqtSignal()
    show_progress_signal = pyqtSignal(str)
    hide_progress_signal = pyqtSignal()
    update_progress_message_signal = pyqtSignal(str)
    show_error_signal = pyqtSignal(str, str)  # title, message
    show_warning_signal = pyqtSignal(str, str)  # title, message
    show_info_signal = pyqtSignal(str, str)  # title, message

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.progress_dialog = None
        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_statusbar()
        self.setup_shortcuts()
        self.setup_signals()

    def setup_ui(self):
        """Настройка основного интерфейса"""
        self.setWindowTitle("AI Audio Recorder - ChatYandexGPT")
        self.setGeometry(100, 100, 900, 800)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(2)

        # Статус модели Vosk
        self.vosk_status = VoskStatusWidget()
        main_layout.addWidget(self.vosk_status)

        # Статус промпта и модели
        self.prompt_status = PromptStatusWidget()
        main_layout.addWidget(self.prompt_status)

        # Кнопки записи
        self.recording_widget = RecordingButtonsWidget()
        self.recording_widget.recording_started.connect(self.app.start_recording)
        self.recording_widget.recording_stopped.connect(self.app.stop_recording)
        self.recording_widget.recording_cancelled.connect(self.app.cancel_recording)
        main_layout.addWidget(self.recording_widget)

        # Live-транскрипция
        self.live_transcription = LiveTranscriptionWidget()
        main_layout.addWidget(self.live_transcription)

        # Разделитель
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)

        # Чат (увеличиваем область)
        self.chat_widget = ChatWidget()
        splitter.addWidget(self.chat_widget)

        # Ввод текста
        self.text_input = TextInputWidget()
        self.text_input.text_sent.connect(self.app.send_text_to_ai)
        self.text_input.clear_button_clicked.connect(self.app.clear_conversation)
        splitter.addWidget(self.text_input)

        # Устанавливаем пропорции: больше места для чата
        splitter.setSizes([600, 120])  # Увеличиваем чат с 500 до 600

        # Применяем темную тему
        self.apply_dark_theme()

    def apply_dark_theme(self):
        """Применение темной темы"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2C2C2C;
                color: #E0E0E0;
            }
            QWidget {
                background-color: #2C2C2C;
                color: #E0E0E0;
            }
            QMenuBar {
                background-color: #3C3C3C;
                color: #E0E0E0;
                border: none;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #6C6C6C;
            }
            QMenu {
                background-color: #3C3C3C;
                color: #E0E0E0;
                border: 1px solid #6C6C6C;
            }
            QMenu::item:selected {
                background-color: #6C6C6C;
            }
            QToolBar {
                background-color: #3C3C3C;
                border: none;
                spacing: 5px;
            }
            QStatusBar {
                background-color: #3C3C3C;
                color: #B0B0B0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #6C6C6C;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

    def setup_menu(self):
        """Настройка меню"""
        menubar = self.menuBar()

        # Меню Файл
        file_menu = menubar.addMenu("Файл")

        export_action = QAction("💾 Экспорт диалога", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.app.export_conversation)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Меню Запись
        record_menu = menubar.addMenu("Запись")

        mic_action = QAction("🎤 Запись с микрофона", self)
        mic_action.setShortcut("Ctrl+R")
        mic_action.triggered.connect(lambda: self.recording_widget.recording_started.emit('mic'))
        record_menu.addAction(mic_action)

        computer_action = QAction("💻 Запись с компьютера", self)
        computer_action.setShortcut("Ctrl+C")
        computer_action.triggered.connect(lambda: self.recording_widget.recording_started.emit('computer'))
        record_menu.addAction(computer_action)

        record_menu.addSeparator()

        cancel_action = QAction("❌ Отменить запись", self)
        cancel_action.setShortcut("Escape")
        cancel_action.triggered.connect(self.app.cancel_recording)
        record_menu.addAction(cancel_action)

        # Меню Настройки
        settings_menu = menubar.addMenu("Настройки")

        prompt_action = QAction("🤖 Сменить помощника", self)
        prompt_action.setShortcut("Ctrl+P")
        prompt_action.triggered.connect(self.app.show_prompt_selector)
        settings_menu.addAction(prompt_action)

        # Меню Управление
        manage_menu = menubar.addMenu("Управление")

        clear_action = QAction("🗑️ Очистить чат", self)
        clear_action.setShortcut("Ctrl+L")
        clear_action.triggered.connect(self.app.clear_conversation)
        manage_menu.addAction(clear_action)

        # Меню Справка
        help_menu = menubar.addMenu("Справка")

        about_action = QAction("О программе", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_toolbar(self):
        """Настройка панели инструментов"""
        toolbar = self.addToolBar("Основная панель")
        toolbar.setMovable(False)

        # Кнопка записи с микрофона
        mic_action = QAction("🎤", self)
        mic_action.setToolTip("Запись с микрофона (Ctrl+R)")
        mic_action.triggered.connect(lambda: self.recording_widget.recording_started.emit('mic'))
        toolbar.addAction(mic_action)

        # Кнопка записи с компьютера
        computer_action = QAction("💻", self)
        computer_action.setToolTip("Запись с компьютера (Ctrl+C)")
        computer_action.triggered.connect(lambda: self.recording_widget.recording_started.emit('computer'))
        toolbar.addAction(computer_action)

        toolbar.addSeparator()

        # Кнопка смены промпта
        prompt_action = QAction("🤖", self)
        prompt_action.setToolTip("Сменить помощника (Ctrl+P)")
        prompt_action.triggered.connect(self.app.show_prompt_selector)
        toolbar.addAction(prompt_action)

        # Кнопка очистки
        clear_action = QAction("🗑️", self)
        clear_action.setToolTip("Очистить чат (Ctrl+L)")
        clear_action.triggered.connect(self.app.clear_conversation)
        toolbar.addAction(clear_action)

        toolbar.addSeparator()

        # Кнопка экспорта
        export_action = QAction("💾", self)
        export_action.setToolTip("Экспорт диалога (Ctrl+E)")
        export_action.triggered.connect(self.app.export_conversation)
        toolbar.addAction(export_action)

    def setup_statusbar(self):
        """Настройка строки состояния"""
        self.statusBar().showMessage("Готов к работе")

    def setup_shortcuts(self):
        """Настройка горячих клавиш"""
        # Горячие клавиши уже настроены в меню
        pass

    def setup_signals(self):
        """Настройка сигналов для безопасного обновления GUI"""
        # Подключаем сигналы к слотам
        self.update_chat_signal.connect(self._update_chat_slot)
        self.update_live_transcription_signal.connect(self._update_live_transcription_slot)
        self.clear_live_transcription_signal.connect(self._clear_live_transcription_slot)
        self.clear_chat_signal.connect(self._clear_chat_slot)
        self.clear_text_input_signal.connect(self._clear_text_input_slot)
        self.show_progress_signal.connect(self._show_progress_slot)
        self.hide_progress_signal.connect(self._hide_progress_slot)
        self.update_progress_message_signal.connect(self._update_progress_message_slot)
        self.show_error_signal.connect(self._show_error_slot)
        self.show_warning_signal.connect(self._show_warning_slot)
        self.show_info_signal.connect(self._show_info_slot)

    def update_vosk_status(self, status: str, is_loaded: bool = False, details: str = ""):
        """Обновление статуса модели Vosk"""
        self.vosk_status.update_status(status, is_loaded, details)

    def update_prompt_status(self, prompt_name: str, model_name: str):
        """Обновление статуса промпта и модели"""
        self.prompt_status.update_prompt(prompt_name)
        self.prompt_status.update_model(model_name)

    def update_live_transcription(self, text: str, is_final: bool = False):
        """Обновление live-транскрипции"""
        self.live_transcription.update_text(text, is_final)

    def clear_live_transcription(self):
        """Очистка live-транскрипции"""
        self.live_transcription.clear()

    def update_chat(self, user_text: str, ai_response: str):
        """Обновление чата"""
        self.chat_widget.add_message(user_text, ai_response)

    def clear_chat(self):
        """Очистка чата"""
        self.chat_widget.clear()

    def clear_text_input(self):
        """Очистка поля ввода"""
        self.text_input.clear()

    def show_progress(self, message: str):
        """Показать диалог прогресса"""
        if not self.progress_dialog:
            self.progress_dialog = ProgressDialog(self)
        self.progress_dialog.update_message(message)
        self.progress_dialog.show()

    def hide_progress(self):
        """Скрыть диалог прогресса"""
        if self.progress_dialog:
            self.progress_dialog.hide()

    def show_error(self, title: str, message: str):
        """Показать ошибку"""
        QMessageBox.critical(self, title, message)

    def show_warning(self, title: str, message: str):
        """Показать предупреждение"""
        QMessageBox.warning(self, title, message)

    def show_info(self, title: str, message: str):
        """Показать информацию"""
        QMessageBox.information(self, title, message)

    def ask_yes_no(self, title: str, message: str) -> bool:
        """Показать диалог подтверждения"""
        reply = QMessageBox.question(self, title, message,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return reply == QMessageBox.StandardButton.Yes

    def show_about(self):
        """Показать информацию о программе"""
        QMessageBox.about(self, "О программе",
                          "ChatYandexGPT - Голосовой интерфейс для YandexGPT\n\n"
                          "Версия: 2.0.0\n"
                          "Интерфейс: PyQt6\n"
                          "Автор: AI Assistant")

    def _update_chat_slot(self, user_text: str, ai_response: str):
        """Слот для обновления чата"""
        self.update_chat(user_text, ai_response)

    def _update_live_transcription_slot(self, text: str, is_final: bool):
        """Слот для обновления live-транскрипции"""
        self.update_live_transcription(text, is_final)

    def _clear_live_transcription_slot(self):
        """Слот для очистки live-транскрипции"""
        self.clear_live_transcription()

    def _clear_chat_slot(self):
        """Слот для очистки чата"""
        self.clear_chat()

    def _clear_text_input_slot(self):
        """Слот для очистки поля ввода"""
        self.clear_text_input()

    def _show_progress_slot(self, message: str):
        """Слот для показа диалога прогресса"""
        self.show_progress(message)

    def _hide_progress_slot(self):
        """Слот для скрытия диалога прогресса"""
        self.hide_progress()

    def _update_progress_message_slot(self, message: str):
        """Слот для обновления сообщения в диалоге прогресса"""
        self.update_progress_message(message)

    def _show_error_slot(self, title: str, message: str):
        """Слот для показа ошибки"""
        self.show_error(title, message)

    def _show_warning_slot(self, title: str, message: str):
        """Слот для показа предупреждения"""
        self.show_warning(title, message)

    def _show_info_slot(self, title: str, message: str):
        """Слот для показа информации"""
        self.show_info(title, message)

    def update_progress_message(self, message: str):
        """Обновить сообщение в диалоге прогресса"""
        if self.progress_dialog:
            self.progress_dialog.update_message(message)
