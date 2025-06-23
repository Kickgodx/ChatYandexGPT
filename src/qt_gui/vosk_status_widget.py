from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel


class VoskStatusWidget(QWidget):
    """Виджет для отображения статуса модели Vosk"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)
        self.status_icon = QLabel("🔄")
        self.status_icon.setFont(QFont("Arial", 12))
        layout.addWidget(self.status_icon)
        self.status_label = QLabel("Модель Vosk: Загрузка...")
        self.status_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.status_label)
        self.details_label = QLabel("")
        self.details_label.setFont(QFont("Arial", 9))
        self.details_label.setStyleSheet("color: #B0B0B0;")
        layout.addWidget(self.details_label)
        layout.addStretch()

    def update_status(self, status: str, is_loaded: bool = False, details: str = ""):
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
