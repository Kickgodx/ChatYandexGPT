from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar


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
        self.message_label = QLabel("Обработка...")
        self.message_label.setFont(QFont("Arial", 12))
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)

    def update_message(self, message: str):
        self.message_label.setText(message)
