from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton


class TextInputWidget(QWidget):
    """Виджет для ввода текста"""

    text_sent = pyqtSignal(str)
    clear_button_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.text_edit.setFocus()

    def setup_ui(self):
        layout = QVBoxLayout(self)
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
        button_layout = QHBoxLayout()
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
        text = self.text_edit.toPlainText().strip()
        if text:
            self.text_sent.emit(text)
            self.text_edit.clear()

    def clear(self):
        self.text_edit.clear()
