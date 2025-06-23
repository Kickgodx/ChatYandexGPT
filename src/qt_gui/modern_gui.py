from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QAction, QPainter, QColor, QPen, QBrush
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QSplitter, QMessageBox

from .chat_widget import ChatWidget
from .live_transcription_widget import LiveTranscriptionWidget
from .progress_dialog import ProgressDialog
from .prompt_status_widget import PromptStatusWidget
from .recording_buttons_widget import RecordingButtonsWidget
from .text_input_widget import TextInputWidget
from .vosk_status_widget import VoskStatusWidget


class ModernGUI(QMainWindow):
    """
    Современный GUI на PyQt6 для ChatYandexGPT
    """
    # Сигналы для безопасного обновления GUI из других потоков
    update_chat_signal = pyqtSignal(str, str)
    update_live_transcription_signal = pyqtSignal(str, bool)
    clear_live_transcription_signal = pyqtSignal()
    clear_chat_signal = pyqtSignal()
    clear_text_input_signal = pyqtSignal()
    show_progress_signal = pyqtSignal(str)
    hide_progress_signal = pyqtSignal()
    update_progress_message_signal = pyqtSignal(str)
    show_error_signal = pyqtSignal(str, str)
    show_warning_signal = pyqtSignal(str, str)
    show_info_signal = pyqtSignal(str, str)

    def __init__(self, app):
        """
        Инициализация основного окна и всех виджетов
        """
        super().__init__()
        self.app = app
        self.progress_dialog = None
        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_statusbar()
        self.setup_shortcuts()
        self.setup_signals()
        self.setup_app_icon()

    def setup_ui(self):
        """
        Настройка основного интерфейса: размещение всех виджетов
        """
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
        # Разделитель между чатом и вводом
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)
        # Чат
        self.chat_widget = ChatWidget()
        splitter.addWidget(self.chat_widget)
        # Ввод текста
        self.text_input = TextInputWidget()
        self.text_input.text_sent.connect(self.app.send_text_to_ai)
        self.text_input.clear_button_clicked.connect(self.app.clear_conversation)
        splitter.addWidget(self.text_input)
        # Пропорции разделителя
        splitter.setSizes([600, 120])
        # Применяем темную тему
        self.apply_dark_theme()
        # Устанавливаем фокус на поле ввода текста при запуске
        self.text_input.text_edit.setFocus()

    def apply_dark_theme(self):
        """
        Применение темной темы оформления для всех элементов
        """
        self.setStyleSheet("""
            QMainWindow { background-color: #2C2C2C; color: #E0E0E0; }
            QWidget { background-color: #2C2C2C; color: #E0E0E0; }
            QMenuBar { background-color: #3C3C3C; color: #E0E0E0; border: none; }
            QMenuBar::item { background-color: transparent; padding: 5px 10px; }
            QMenuBar::item:selected { background-color: #6C6C6C; }
            QMenu { background-color: #3C3C3C; color: #E0E0E0; border: 1px solid #6C6C6C; }
            QMenu::item:selected { background-color: #6C6C6C; }
            QToolBar { background-color: #3C3C3C; border: none; spacing: 5px; }
            QStatusBar { background-color: #3C3C3C; color: #B0B0B0; }
            QGroupBox { font-weight: bold; border: 1px solid #6C6C6C; border-radius: 5px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
        """)

    def setup_menu(self):
        """
        Настройка верхнего меню приложения
        """
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
        """
        Настройка панели инструментов (toolbar)
        """
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
        """
        Настройка строки состояния (status bar)
        """
        self.statusBar().showMessage("Готов к работе")

    def setup_shortcuts(self):
        """
        Настройка горячих клавиш (если потребуется)
        """
        pass

    def setup_signals(self):
        """
        Настройка сигналов для безопасного обновления GUI из других потоков
        """
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
        """
        Обновление статуса модели Vosk
        """
        self.vosk_status.update_status(status, is_loaded, details)

    def update_prompt_status(self, prompt_name: str, model_name: str):
        """
        Обновление статуса промпта и модели
        """
        self.prompt_status.update_prompt(prompt_name)
        self.prompt_status.update_model(model_name)

    def update_live_transcription(self, text: str, is_final: bool = False):
        """
        Обновление live-транскрипции
        """
        self.live_transcription.update_text(text, is_final)

    def clear_live_transcription(self):
        """
        Очистка live-транскрипции
        """
        self.live_transcription.clear()

    def update_chat(self, user_text: str, ai_response: str):
        """
        Обновление чата
        """
        self.chat_widget.add_message(user_text, ai_response)

    def clear_chat(self):
        """
        Очистка чата
        """
        self.chat_widget.clear()

    def clear_text_input(self):
        """
        Очистка поля ввода
        """
        self.text_input.clear()

    def show_progress(self, message: str):
        """
        Показать диалог прогресса
        """
        if not self.progress_dialog:
            self.progress_dialog = ProgressDialog(self)
        self.progress_dialog.update_message(message)
        self.progress_dialog.show()

    def hide_progress(self):
        """
        Скрыть диалог прогресса
        """
        if self.progress_dialog:
            self.progress_dialog.hide()

    def show_error(self, title: str, message: str):
        """
        Показать ошибку
        """
        QMessageBox.critical(self, title, message)

    def show_warning(self, title: str, message: str):
        """
        Показать предупреждение
        """
        QMessageBox.warning(self, title, message)

    def show_info(self, title: str, message: str):
        """
        Показать информацию
        """
        QMessageBox.information(self, title, message)

    def ask_yes_no(self, title: str, message: str) -> bool:
        """
        Показать диалог подтверждения
        """
        reply = QMessageBox.question(self, title, message,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return reply == QMessageBox.StandardButton.Yes

    def show_about(self):
        """
        Показать информацию о программе
        """
        QMessageBox.about(self, "О программе",
                          "ChatYandexGPT - Голосовой интерфейс для YandexGPT\n\n"
                          "Версия: 2.0.0\n"
                          "Интерфейс: PyQt6\n"
                          "Автор: Denis R.")

    def _update_chat_slot(self, user_text: str, ai_response: str):
        """
        Слот для обновления чата
        """
        self.update_chat(user_text, ai_response)

    def _update_live_transcription_slot(self, text: str, is_final: bool):
        """
        Слот для обновления live-транскрипции
        """
        self.update_live_transcription(text, is_final)

    def _clear_live_transcription_slot(self):
        """
        Слот для очистки live-транскрипции
        """
        self.clear_live_transcription()

    def _clear_chat_slot(self):
        """
        Слот для очистки чата
        """
        self.clear_chat()

    def _clear_text_input_slot(self):
        """
        Слот для очистки поля ввода
        """
        self.clear_text_input()

    def _show_progress_slot(self, message: str):
        """
        Слот для показа диалога прогресса
        """
        self.show_progress(message)

    def _hide_progress_slot(self):
        """
        Слот для скрытия диалога прогресса
        """
        self.hide_progress()

    def _update_progress_message_slot(self, message: str):
        """
        Слот для обновления сообщения в диалоге прогресса
        """
        self.update_progress_message(message)

    def _show_error_slot(self, title: str, message: str):
        """
        Слот для показа ошибки
        """
        self.show_error(title, message)

    def _show_warning_slot(self, title: str, message: str):
        """
        Слот для показа предупреждения
        """
        self.show_warning(title, message)

    def _show_info_slot(self, title: str, message: str):
        """
        Слот для показа информации
        """
        self.show_info(title, message)

    def update_progress_message(self, message: str):
        """
        Обновить сообщение в диалоге прогресса
        """
        if self.progress_dialog:
            self.progress_dialog.update_message(message)

    def setup_app_icon(self):
        """
        Установка иконки приложения
        """
        try:
            icon = self.create_app_icon()
            if icon:
                self.setWindowIcon(icon)
        except Exception as e:
            print(f"Ошибка установки иконки: {e}")

    @staticmethod
    def create_app_icon():
        """
        Создание иконки приложения программно
        """
        try:
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            gradient = QBrush(QColor(70, 130, 180))
            painter.fillRect(0, 0, 32, 32, gradient)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(8, 8, 16, 16)
            painter.setPen(QPen(QColor(100, 150, 200), 1))
            painter.setBrush(QBrush(QColor(100, 150, 200)))
            painter.drawEllipse(12, 12, 8, 8)
            painter.end()
            return QIcon(pixmap)
        except Exception as e:
            print(f"Ошибка создания иконки: {e}")
            return None
