from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel


class RecordingButtonsWidget(QWidget):
    """Виджет с кнопками записи"""

    recording_started = pyqtSignal(str)
    recording_stopped = pyqtSignal()
    recording_cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(3)
        button_layout = QHBoxLayout()
        self.mic_button = QPushButton("🎤 Запись с микрофона")
        self.mic_button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.mic_button.setMinimumHeight(40)
        self.mic_button.clicked.connect(lambda: self.recording_started.emit('mic'))
        button_layout.addWidget(self.mic_button)
        self.computer_button = QPushButton("💻 Запись с компьютера")
        self.computer_button.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.computer_button.setMinimumHeight(40)
        self.computer_button.clicked.connect(lambda: self.recording_started.emit('computer'))
        button_layout.addWidget(self.computer_button)
        layout.addLayout(button_layout)
        self.recording_label = QLabel("")
        self.recording_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.recording_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_label.setStyleSheet("color: #8B5A5A;")
        layout.addWidget(self.recording_label)
        self.cancel_button = QPushButton("❌ Отменить запись")
        self.cancel_button.setFont(QFont("Arial", 10))
        self.cancel_button.setMinimumHeight(35)
        self.cancel_button.clicked.connect(self.recording_cancelled.emit)
        self.cancel_button.setEnabled(False)
        layout.addWidget(self.cancel_button)
        self.apply_styles()

    def apply_styles(self):
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
        if is_recording:
            if device_type == 'mic':
                self.mic_button.setText("⏹️ Отправить запись (микрофон)")
                self.mic_button.setStyleSheet("""
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
                self.computer_button.setEnabled(False)
                try:
                    self.mic_button.clicked.disconnect()
                except TypeError:
                    pass
                self.mic_button.clicked.connect(self.recording_stopped.emit)
            else:
                self.computer_button.setText("⏹️ Отправить запись (компьютер)")
                self.computer_button.setStyleSheet("""
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
                self.mic_button.setEnabled(False)
                try:
                    self.computer_button.clicked.disconnect()
                except TypeError:
                    pass
                self.computer_button.clicked.connect(self.recording_stopped.emit)
            self.recording_label.setText("🔴 ЗАПИСЬ...")
            self.cancel_button.setEnabled(True)
        else:
            self.mic_button.setText("🎤 Запись с микрофона")
            self.computer_button.setText("💻 Запись с компьютера")
            self.mic_button.setEnabled(True)
            self.computer_button.setEnabled(True)
            try:
                self.mic_button.clicked.disconnect()
            except TypeError:
                pass
            self.mic_button.clicked.connect(lambda: self.recording_started.emit('mic'))
            try:
                self.computer_button.clicked.disconnect()
            except TypeError:
                pass
            self.computer_button.clicked.connect(lambda: self.recording_started.emit('computer'))
            self.apply_styles()
            self.recording_label.setText("")
            self.cancel_button.setEnabled(False)
