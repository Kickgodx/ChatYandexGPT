from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel


class PromptStatusWidget(QWidget):
    """Виджет для отображения статуса промпта и модели"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)
        self.prompt_label = QLabel("Помощник: Программист-помощник")
        self.prompt_label.setFont(QFont("Arial", 10))
        self.prompt_label.setStyleSheet("color: #B0B0B0; font-style: italic;")
        layout.addWidget(self.prompt_label)
        separator = QLabel("|")
        separator.setStyleSheet("color: #6C6C6C; margin: 0 10px;")
        layout.addWidget(separator)
        self.model_label = QLabel("Модель: YandexGPT Lite")
        self.model_label.setFont(QFont("Arial", 10))
        self.model_label.setStyleSheet("color: #B0B0B0; font-style: italic;")
        layout.addWidget(self.model_label)
        layout.addStretch()

    def update_prompt(self, prompt_name: str):
        self.prompt_label.setText(f"Помощник: {prompt_name}")

    def update_model(self, model_name: str):
        self.model_label.setText(f"Модель: {model_name}")
