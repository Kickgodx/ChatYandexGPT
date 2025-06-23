import tkinter as tk
from tkinter import ttk, messagebox


class PromptManager:
    """Менеджер для выбора и управления промптами в GUI"""

    def __init__(self, app):
        self.app = app
        self.prompt_window = None
        self.prompt_var = None
        self.prompt_combobox = None
        self.model_var = None
        self.model_combobox = None

        # Цветовая схема в серых тонах (такая же как в GUI)
        self.colors = {
            'bg_primary': '#2C2C2C',  # Темно-серый фон
            'bg_secondary': '#3C3C3C',  # Средне-серый для элементов
            'bg_tertiary': '#4C4C4C',  # Светло-серый для кнопок
            'text_primary': '#E0E0E0',  # Светло-серый текст
            'text_secondary': '#B0B0B0',  # Серый текст
            'accent': '#6C6C6C',  # Акцентный серый
            'success': '#4A7C59',  # Темно-зеленый для успеха
            'warning': '#8B7355',  # Темно-оранжевый для предупреждений
            'error': '#8B5A5A',  # Темно-красный для ошибок
            'code_bg': '#1E1E1E',  # Темный фон для кода
            'code_text': '#D4D4D4',  # Светлый текст для кода
            'code_keyword': '#569CD6',  # Синий для ключевых слов
            'code_string': '#CE9178',  # Оранжевый для строк
            'code_comment': '#6A9955',  # Зеленый для комментариев
        }

    def show_prompt_selector(self):
        """Показать окно выбора промпта"""
        if self.prompt_window:
            self.prompt_window.destroy()

        self.prompt_window = tk.Toplevel(self.app.gui.root)
        self.prompt_window.title("Выбор промпта")
        self.prompt_window.geometry("700x600")  # Уменьшили высоту с 600 до 550
        self.prompt_window.minsize(600, 500)  # Минимальный размер окна
        self.prompt_window.transient(self.app.gui.root)
        self.prompt_window.grab_set()
        self.prompt_window.configure(bg=self.colors['bg_primary'])

        # Центрирование окна
        self.prompt_window.update_idletasks()
        x = (self.prompt_window.winfo_screenwidth() // 2) - (700 // 2)
        y = (self.prompt_window.winfo_screenheight() // 2) - (550 // 2)  # Обновили координату Y
        self.prompt_window.geometry(f"700x550+{x}+{y}")  # Обновили размер

        self._create_prompt_selector_ui()

    def _create_prompt_selector_ui(self):
        """Создание интерфейса выбора промпта"""
        # Заголовок
        title_label = tk.Label(self.prompt_window, text="Выберите тип помощника",
                               font=("Arial", 16, "bold"),
                               bg=self.colors['bg_primary'], fg=self.colors['text_primary'])
        title_label.pack(pady=15)

        # Фрейм для выбора модели
        model_frame = tk.Frame(self.prompt_window, bg=self.colors['bg_primary'])
        model_frame.pack(pady=5, padx=20, fill=tk.X)

        # Комбобокс для выбора модели
        tk.Label(model_frame, text="Модель YandexGPT:",
                 font=("Arial", 12),
                 bg=self.colors['bg_primary'], fg=self.colors['text_primary']).pack(anchor=tk.W)

        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(model_frame, textvariable=self.model_var,
                                           state="readonly", font=("Arial", 11))
        self.model_combobox.pack(fill=tk.X, pady=5)

        # Заполнение комбобокса моделей
        available_models = self.app.bot.get_available_models()
        model_options = []
        for key, value in available_models.items():
            model_options.append(f"{value['name']} - {value['description']}")

        self.model_combobox['values'] = model_options

        # Установка текущего значения модели
        current_model = self.app.bot.get_current_model_info()
        current_model_option = f"{current_model['name']} - {current_model['description']}"
        if current_model_option in model_options:
            self.model_combobox.set(current_model_option)

        # Фрейм для выбора промпта
        selection_frame = tk.Frame(self.prompt_window, bg=self.colors['bg_primary'])
        selection_frame.pack(pady=10, padx=20, fill=tk.X)

        # Комбобокс для выбора промпта
        tk.Label(selection_frame, text="Тип помощника:",
                 font=("Arial", 12),
                 bg=self.colors['bg_primary'], fg=self.colors['text_primary']).pack(anchor=tk.W)

        self.prompt_var = tk.StringVar()
        self.prompt_combobox = ttk.Combobox(selection_frame, textvariable=self.prompt_var,
                                            state="readonly", font=("Arial", 11))
        self.prompt_combobox.pack(fill=tk.X, pady=5)

        # Заполнение комбобокса
        available_prompts = self.app.bot.get_available_prompts()
        prompt_options = []
        for key, value in available_prompts.items():
            prompt_options.append(f"{value['name']} - {value['description']}")

        self.prompt_combobox['values'] = prompt_options

        # Установка текущего значения
        current_prompt = self.app.bot.get_current_prompt_info()
        current_option = f"{current_prompt['name']} - {current_prompt['description']}"
        if current_option in prompt_options:
            self.prompt_combobox.set(current_option)

        # Привязка события изменения
        self.prompt_combobox.bind('<<ComboboxSelected>>', self._on_prompt_selected)

        # Фрейм для описания (уменьшаем высоту)
        description_frame = tk.Frame(self.prompt_window, bg=self.colors['bg_primary'])
        description_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(description_frame, text="Описание выбранного помощника:",
                 font=("Arial", 12, "bold"),
                 bg=self.colors['bg_primary'], fg=self.colors['text_primary']).pack(anchor=tk.W)

        # Текстовое поле для описания (уменьшаем высоту)
        self.description_text = tk.Text(description_frame, wrap=tk.WORD, height=10,  # Уменьшили с 15 до 10
                                        state=tk.DISABLED, font=("Consolas", 10),
                                        bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                                        insertbackground=self.colors['text_primary'],
                                        selectbackground=self.colors['accent'],
                                        selectforeground=self.colors['text_primary'],
                                        relief=tk.FLAT)
        self.description_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Убираем скроллбар - он не нужен для фиксированной высоты
        # scrollbar = tk.Scrollbar(description_frame, orient=tk.VERTICAL,
        #                          command=self.description_text.yview,
        #                          bg=self.colors['bg_secondary'])
        # scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # self.description_text.config(yscrollcommand=scrollbar.set)

        # Кнопки (увеличиваем отступ сверху для лучшей видимости)
        button_frame = tk.Frame(self.prompt_window, bg=self.colors['bg_primary'])
        button_frame.pack(pady=15, padx=20, fill=tk.X)

        # Кнопка применения (зеленая, самая заметная)
        apply_button = tk.Button(button_frame, text="✅ Применить",
                                 command=self._apply_prompt,
                                 bg=self.colors['success'], fg=self.colors['text_primary'],
                                 font=("Arial", 12, "bold"), width=15,
                                 relief=tk.FLAT, activebackground=self.colors['accent'],
                                 activeforeground=self.colors['text_primary'])
        apply_button.pack(side=tk.LEFT, padx=10)

        # Кнопка сброса диалога
        reset_button = tk.Button(button_frame, text="🔄 Сбросить диалог",
                                 command=self._reset_conversation,
                                 bg=self.colors['warning'], fg=self.colors['text_primary'],
                                 font=("Arial", 11), width=15,
                                 relief=tk.FLAT, activebackground=self.colors['accent'],
                                 activeforeground=self.colors['text_primary'])
        reset_button.pack(side=tk.LEFT, padx=10)

        # Кнопка отмены
        cancel_button = tk.Button(button_frame, text="❌ Отмена",
                                  command=self._cancel,
                                  bg=self.colors['error'], fg=self.colors['text_primary'],
                                  font=("Arial", 11), width=15,
                                  relief=tk.FLAT, activebackground=self.colors['accent'],
                                  activeforeground=self.colors['text_primary'])
        cancel_button.pack(side=tk.RIGHT, padx=10)

        # Показать описание текущего промпта
        self._show_current_prompt_description()

    def _on_prompt_selected(self, event=None):
        """Обработчик выбора промпта"""
        selected = self.prompt_var.get()
        if selected:
            # Найти ключ промпта по названию
            available_prompts = self.app.bot.get_available_prompts()
            for key, value in available_prompts.items():
                if f"{value['name']} - {value['description']}" == selected:
                    self._show_prompt_description(key)
                    break

    def _show_current_prompt_description(self):
        """Показать описание текущего промпта"""
        current_prompt = self.app.bot.get_current_prompt_info()
        self._show_prompt_description_by_info(current_prompt)

    def _show_prompt_description(self, prompt_key):
        """Показать описание промпта по ключу"""
        from yandexchat_bot import PromptCollection
        prompt_info = PromptCollection.get_prompt(prompt_key)
        self._show_prompt_description_by_info(prompt_info)

    def _show_prompt_description_by_info(self, prompt_info):
        """Показать описание промпта по информации"""
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete(1.0, tk.END)

        # Форматированное отображение описания
        content = prompt_info['content']
        # Убираем лишние переносы строк для лучшего отображения
        content = content.replace('\n\n', '\n').replace('\n\n\n', '\n\n')

        self.description_text.insert(tk.END, content)
        self.description_text.config(state=tk.DISABLED)

    def _apply_prompt(self):
        """Применить выбранный промпт и модель"""
        selected_prompt = self.prompt_var.get()
        selected_model = self.model_var.get()

        if not selected_prompt:
            messagebox.showwarning("Предупреждение", "Выберите тип промпта")
            return

        if not selected_model:
            messagebox.showwarning("Предупреждение", "Выберите модель")
            return

        # Применяем модель
        available_models = self.app.bot.get_available_models()
        for key, value in available_models.items():
            if f"{value['name']} - {value['description']}" == selected_model:
                try:
                    self.app.bot.set_model(key)
                    model_name = value['name']
                    break
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось применить модель: {str(e)}")
                    return

        # Применяем промпт
        available_prompts = self.app.bot.get_available_prompts()
        for key, value in available_prompts.items():
            if f"{value['name']} - {value['description']}" == selected_prompt:
                # Применить промпт
                self.app.bot.set_prompt(key)

                # Обновить статус в главном окне
                self._update_main_window_status(key, value['name'], model_name)

                messagebox.showinfo("Успех", f"Настройки изменены:\nМодель: {model_name}\nПомощник: {value['name']}")
                self.prompt_window.destroy()
                return

        messagebox.showerror("Ошибка", "Не удалось применить выбранные настройки")

    def _reset_conversation(self):
        """Сбросить диалог"""
        if messagebox.askyesno("Подтверждение",
                               "Сбросить историю диалога? Это действие нельзя отменить."):
            self.app.bot.reset_conversation()
            self.app.gui.clear_text_widget()
            messagebox.showinfo("Успех", "История диалога сброшена")

    def _cancel(self):
        """Отмена выбора"""
        self.prompt_window.destroy()

    def _update_main_window_status(self, prompt_key, prompt_name, model_name):
        """Обновить статус в главном окне"""
        # Обновляем заголовок окна
        self.app.gui.root.title(f"AI Audio Recorder - {prompt_name} ({model_name})")

        # Обновляем индикатор текущего промпта в интерфейсе
        if hasattr(self.app.gui, 'prompt_status_label'):
            self.app.gui.prompt_status_label.config(text=f"Помощник: {prompt_name}")

        # Обновляем индикатор текущей модели в интерфейсе
        if hasattr(self.app.gui, 'model_status_label'):
            self.app.gui.model_status_label.config(text=f"Модель: {model_name}")

    def add_prompt_status_to_gui(self):
        """Добавить индикатор текущего промпта в главное окно"""
        if not hasattr(self.app.gui, 'prompt_status_label'):
            # Создаем фрейм для статуса промпта
            status_frame = tk.Frame(self.app.gui.root, bg=self.colors['bg_primary'])
            status_frame.pack(pady=5, padx=10, fill=tk.X)

            # Метка статуса промпта
            current_prompt = self.app.bot.get_current_prompt_info()
            self.app.gui.prompt_status_label = tk.Label(
                status_frame,
                text=f"Помощник: {current_prompt['name']}",
                font=("Arial", 10, "italic"),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg_primary']
            )
            self.app.gui.prompt_status_label.pack(side=tk.LEFT)

            # Метка статуса модели
            current_model = self.app.bot.get_current_model_info()
            self.app.gui.model_status_label = tk.Label(
                status_frame,
                text=f"Модель: {current_model['name']}",
                font=("Arial", 10, "italic"),
                fg=self.colors['text_secondary'],
                bg=self.colors['bg_primary']
            )
            self.app.gui.model_status_label.pack(side=tk.LEFT, padx=(20, 0))
