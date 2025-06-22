import tkinter as tk
from tkinter import ttk, messagebox


class PromptManager:
    """Менеджер для выбора и управления промптами в GUI"""

    def __init__(self, app):
        self.app = app
        self.prompt_window = None
        self.prompt_var = None
        self.prompt_combobox = None

    def show_prompt_selector(self):
        """Показать окно выбора промпта"""
        if self.prompt_window:
            self.prompt_window.destroy()

        self.prompt_window = tk.Toplevel(self.app.gui.root)
        self.prompt_window.title("Выбор промпта")
        self.prompt_window.geometry("700x600")
        self.prompt_window.transient(self.app.gui.root)
        self.prompt_window.grab_set()

        # Центрирование окна
        self.prompt_window.update_idletasks()
        x = (self.prompt_window.winfo_screenwidth() // 2) - (700 // 2)
        y = (self.prompt_window.winfo_screenheight() // 2) - (600 // 2)
        self.prompt_window.geometry(f"700x600+{x}+{y}")

        self._create_prompt_selector_ui()

    def _create_prompt_selector_ui(self):
        """Создание интерфейса выбора промпта"""
        # Заголовок
        title_label = tk.Label(self.prompt_window, text="Выберите тип Промпта",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=20)

        # Фрейм для выбора
        selection_frame = tk.Frame(self.prompt_window)
        selection_frame.pack(pady=20, padx=20, fill=tk.X)

        # Комбобокс для выбора промпта
        tk.Label(selection_frame, text="Тип помощника:", font=("Arial", 12)).pack(anchor=tk.W)

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

        # Фрейм для описания
        description_frame = tk.Frame(self.prompt_window)
        description_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(description_frame, text="Описание выбранного промпта:",
                 font=("Arial", 12, "bold")).pack(anchor=tk.W)

        # Текстовое поле для описания
        self.description_text = tk.Text(description_frame, wrap=tk.WORD, height=15,
                                        state=tk.DISABLED, font=("Arial", 10))
        self.description_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Скроллбар для текста
        scrollbar = tk.Scrollbar(description_frame, orient=tk.VERTICAL,
                                 command=self.description_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.description_text.config(yscrollcommand=scrollbar.set)

        # Кнопки
        button_frame = tk.Frame(self.prompt_window)
        button_frame.pack(pady=20, padx=20, fill=tk.X)

        # Кнопка применения (зеленая, самая заметная)
        apply_button = tk.Button(button_frame, text="✅ Применить",
                                 command=self._apply_prompt, bg="#4CAF50", fg="white",
                                 font=("Arial", 12, "bold"), height=2, width=15)
        apply_button.pack(side=tk.LEFT, padx=10)

        # Кнопка сброса диалога
        reset_button = tk.Button(button_frame, text="🔄 Сбросить диалог",
                                 command=self._reset_conversation, bg="#FF9800", fg="white",
                                 font=("Arial", 11), height=2, width=15)
        reset_button.pack(side=tk.LEFT, padx=10)

        # Кнопка отмены
        cancel_button = tk.Button(button_frame, text="❌ Отмена",
                                  command=self._cancel, bg="#f44336", fg="white",
                                  font=("Arial", 11), height=2, width=15)
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
        """Применить выбранный промпт"""
        selected = self.prompt_var.get()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите тип промпта")
            return

        # Найти ключ промпта
        available_prompts = self.app.bot.get_available_prompts()
        for key, value in available_prompts.items():
            if f"{value['name']} - {value['description']}" == selected:
                # Применить промпт
                self.app.bot.set_prompt(key)

                # Обновить статус в главном окне
                self._update_main_window_status(key, value['name'])

                messagebox.showinfo("Успех", f"Промпт изменен на: {value['name']}")
                self.prompt_window.destroy()
                return

        messagebox.showerror("Ошибка", "Не удалось применить выбранный промпт")

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

    def _update_main_window_status(self, prompt_key, prompt_name):
        """Обновить статус в главном окне"""
        # Обновляем заголовок окна
        self.app.gui.root.title(f"AI Audio Recorder - {prompt_name}")

        # Обновляем индикатор текущего промпта в интерфейсе
        if hasattr(self.app.gui, 'prompt_status_label'):
            self.app.gui.prompt_status_label.config(text=f"Текущий помощник: {prompt_name}")

    def add_prompt_status_to_gui(self):
        """Добавить индикатор текущего промпта в главное окно"""
        if not hasattr(self.app.gui, 'prompt_status_label'):
            # Создаем фрейм для статуса промпта
            status_frame = tk.Frame(self.app.gui.root)
            status_frame.pack(pady=5, padx=10, fill=tk.X)

            # Метка статуса
            current_prompt = self.app.bot.get_current_prompt_info()
            self.app.gui.prompt_status_label = tk.Label(
                status_frame,
                text=f"Текущий промпт: {current_prompt['name']}",
                font=("Arial", 10, "italic"),
                fg="#666666"
            )
            self.app.gui.prompt_status_label.pack(side=tk.LEFT)
