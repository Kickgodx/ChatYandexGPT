from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit


class LiveTranscriptionWidget(QWidget):
    """Виджет для live-транскрипции"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.text_edit = None
        self.title_label = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(10, 5, 10, 5)
        self.title_label = QLabel("🎯 Распознанный текст:")
        self.title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.title_label)
        self.text_edit = QTextEdit()
        self.text_edit.setMaximumHeight(60)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Arial", 11))
        layout.addWidget(self.text_edit)

    def update_text(self, text: str, is_final: bool = False):
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
        self.text_edit.clear()
        self.title_label.setText("🎯 Распознанный текст:")
        self.text_edit.setStyleSheet("color: #B0B0B0;")
