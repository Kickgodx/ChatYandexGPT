"""
Менеджер промптов для PyQt6
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTextEdit, QPushButton, QMessageBox, QGroupBox,
    QFormLayout
)


class PromptSelectorDialog(QDialog):
    """Диалог выбора промпта для PyQt6"""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.setup_ui()

    def setup_ui(self):
        """Настройка интерфейса"""
        self.setWindowTitle("Выбор помощника")
        self.setGeometry(200, 200, 700, 600)
        self.setModal(True)

        # Основной layout
        layout = QVBoxLayout(self)

        # Заголовок
        title_label = QLabel("Выберите тип помощника")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Группа выбора модели
        model_group = QGroupBox("Модель YandexGPT")
        model_layout = QFormLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.setFont(QFont("Arial", 11))

        # Заполняем модели
        available_models = self.app.bot.get_available_models()
        self.model_options = []
        for key, value in available_models.items():
            option_text = f"{value['name']} - {value['description']}"
            self.model_options.append((key, option_text))
            self.model_combo.addItem(option_text)

        # Устанавливаем текущую модель
        current_model = self.app.bot.get_current_model_info()
        current_model_option = f"{current_model['name']} - {current_model['description']}"
        index = self.model_combo.findText(current_model_option)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

        model_layout.addRow("Модель:", self.model_combo)
        layout.addWidget(model_group)

        # Группа выбора промпта
        prompt_group = QGroupBox("Тип помощника")
        prompt_layout = QFormLayout(prompt_group)

        self.prompt_combo = QComboBox()
        self.prompt_combo.setFont(QFont("Arial", 11))

        # Заполняем промпты
        available_prompts = self.app.bot.get_available_prompts()
        self.prompt_options = []
        for key, value in available_prompts.items():
            option_text = f"{value['name']} - {value['description']}"
            self.prompt_options.append((key, option_text))
            self.prompt_combo.addItem(option_text)

        # Устанавливаем текущий промпт
        current_prompt = self.app.bot.get_current_prompt_info()
        current_prompt_option = f"{current_prompt['name']} - {current_prompt['description']}"
        index = self.prompt_combo.findText(current_prompt_option)
        if index >= 0:
            self.prompt_combo.setCurrentIndex(index)

        prompt_layout.addRow("Помощник:", self.prompt_combo)
        layout.addWidget(prompt_group)

        # Описание
        description_group = QGroupBox("Описание выбранного помощника")
        description_layout = QVBoxLayout(description_group)

        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        self.description_text.setFont(QFont("Consolas", 10))
        self.description_text.setMaximumHeight(300)
        description_layout.addWidget(self.description_text)

        layout.addWidget(description_group)

        # Кнопки
        button_layout = QHBoxLayout()

        self.apply_button = QPushButton("✅ Применить")
        self.apply_button.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.apply_button.clicked.connect(self.apply_settings)
        button_layout.addWidget(self.apply_button)

        self.reset_button = QPushButton("🔄 Сбросить диалог")
        self.reset_button.setFont(QFont("Arial", 11))
        self.reset_button.clicked.connect(self.reset_conversation)
        button_layout.addWidget(self.reset_button)

        self.cancel_button = QPushButton("❌ Отмена")
        self.cancel_button.setFont(QFont("Arial", 11))
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        # Подключаем сигнал ПОСЛЕ создания всех виджетов
        self.prompt_combo.currentTextChanged.connect(self.on_prompt_changed)

        # Показываем описание текущего промпта
        self.show_current_prompt_description()

        # Применяем стили
        self.apply_styles()

    def apply_styles(self):
        """Применение стилей"""
        self.setStyleSheet("""
            QDialog {
                background-color: #2C2C2C;
                color: #E0E0E0;
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
            QComboBox {
                background-color: #3C3C3C;
                color: #E0E0E0;
                border: 1px solid #6C6C6C;
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #E0E0E0;
            }
            QTextEdit {
                background-color: #3C3C3C;
                color: #E0E0E0;
                border: 1px solid #6C6C6C;
                border-radius: 3px;
            }
            QPushButton {
                background-color: #4C4C4C;
                color: #E0E0E0;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #6C6C6C;
            }
            QPushButton:pressed {
                background-color: #2C2C2C;
            }
        """)

        # Специальные стили для кнопок
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: #4A7C59;
                color: #E0E0E0;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #5A8C69;
            }
            QPushButton:pressed {
                background-color: #3A6C49;
            }
        """)

        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #8B7355;
                color: #E0E0E0;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #9B8365;
            }
            QPushButton:pressed {
                background-color: #7B6345;
            }
        """)

        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #8B5A5A;
                color: #E0E0E0;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #9B6A6A;
            }
            QPushButton:pressed {
                background-color: #7B4A4A;
            }
        """)

    def on_prompt_changed(self, selected_text):
        """Обработчик изменения выбранного промпта"""
        # Проверяем, что виджет существует
        if not hasattr(self, 'description_text') or self.description_text is None:
            return

        if selected_text:
            # Найти ключ промпта по названию
            for key, option_text in self.prompt_options:
                if option_text == selected_text:
                    self.show_prompt_description(key)
                    break

    def show_current_prompt_description(self):
        """Показать описание текущего промпта"""
        current_prompt = self.app.bot.get_current_prompt_info()
        self.show_prompt_description_by_info(current_prompt)

    def show_prompt_description(self, prompt_key):
        """Показать описание промпта по ключу"""
        from yandexchat_bot import PromptCollection
        prompt_info = PromptCollection.get_prompt(prompt_key)
        self.show_prompt_description_by_info(prompt_info)

    def show_prompt_description_by_info(self, prompt_info):
        """Показать описание промпта по информации"""
        # Проверяем, что виджет существует
        if not hasattr(self, 'description_text') or self.description_text is None:
            return

        content = prompt_info['content']
        # Оптимизируем отображение текста
        content = content.replace('\n\n', '\n').replace('\n\n\n', '\n\n')

        self.description_text.setPlainText(content)

    def apply_settings(self):
        """Применить выбранные настройки"""
        selected_prompt = self.prompt_combo.currentText()
        selected_model = self.model_combo.currentText()

        if not selected_prompt:
            QMessageBox.warning(self, "Предупреждение", "Выберите тип промпта")
            return

        if not selected_model:
            QMessageBox.warning(self, "Предупреждение", "Выберите модель")
            return

        # Применяем модель
        for key, option_text in self.model_options:
            if option_text == selected_model:
                try:
                    self.app.bot.set_model(key)
                    model_name = self.app.bot.get_available_models()[key]['name']
                    break
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Не удалось применить модель: {str(e)}")
                    return

        # Применяем промпт
        for key, option_text in self.prompt_options:
            if option_text == selected_prompt:
                try:
                    self.app.bot.set_prompt(key)
                    prompt_name = self.app.bot.get_available_prompts()[key]['name']

                    # Обновляем статус в главном окне
                    self.app.gui.update_prompt_status(prompt_name, model_name)

                    QMessageBox.information(self, "Успех",
                                            f"Настройки изменены:\nМодель: {model_name}\nПомощник: {prompt_name}")
                    self.accept()
                    return
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Не удалось применить промпт: {str(e)}")
                    return

        QMessageBox.critical(self, "Ошибка", "Не удалось применить выбранные настройки")

    def reset_conversation(self):
        """Сбросить диалог"""
        reply = QMessageBox.question(self, "Подтверждение",
                                     "Сбросить историю диалога? Это действие нельзя отменить.",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.app.bot.reset_conversation()
                self.app.gui.clear_chat()
                QMessageBox.information(self, "Успех", "История диалога сброшена")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сбросить диалог: {str(e)}")


class PromptManagerQt:
    """Менеджер промптов для PyQt6"""

    def __init__(self, app):
        self.app = app

    def show_prompt_selector(self):
        """Показать диалог выбора промпта"""
        dialog = PromptSelectorDialog(self.app, self.app.gui)
        dialog.exec()
