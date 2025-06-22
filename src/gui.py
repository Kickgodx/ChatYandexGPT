import re
import tkinter as tk
from tkinter import scrolledtext, messagebox


class GUI:
    """Класс для создания графического интерфейса"""

    def __init__(self, app):
        self.app = app
        self.root = None
        self.mic_button = None
        self.computer_button = None
        self.recording_label = None
        self.cancel_button = None
        self.text_widget = None
        self.text_entry = None
        self.send_button = None
        self.export_button = None
        self.clear_button = None
        self.prompt_button = None
        self.progress_window = None
        self.progress_label = None

        # Цветовая схема в серых тонах
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

    def create_gui(self):
        """Создание графического интерфейса"""
        self.root = tk.Tk()
        self.root.title("AI Audio Recorder")
        self.root.geometry("800x900")
        self.root.configure(bg=self.colors['bg_primary'])

        # Кнопки записи
        self.mic_button = tk.Button(self.root, text="🎤 Запись с микрофона",
                                    command=self.app.start_mic_recording,
                                    bg=self.colors['bg_tertiary'], fg=self.colors['text_primary'],
                                    font=("Arial", 11, "bold"), relief=tk.FLAT,
                                    activebackground=self.colors['accent'],
                                    activeforeground=self.colors['text_primary'])
        self.mic_button.pack(pady=10, padx=20, fill=tk.X)

        self.computer_button = tk.Button(self.root, text="💻 Запись с компьютера",
                                         command=self.app.start_computer_recording,
                                         bg=self.colors['bg_tertiary'], fg=self.colors['text_primary'],
                                         font=("Arial", 11, "bold"), relief=tk.FLAT,
                                         activebackground=self.colors['accent'],
                                         activeforeground=self.colors['text_primary'])
        self.computer_button.pack(pady=5, padx=20, fill=tk.X)

        # Индикатор записи
        self.recording_label = tk.Label(self.root, text="", fg=self.colors['error'],
                                        font=("Arial", 12, "bold"), bg=self.colors['bg_primary'])
        self.recording_label.pack(pady=5)

        # Кнопка отмены записи
        self.cancel_button = tk.Button(self.root, text="❌ Отменить запись",
                                       command=self.app.cancel_recording,
                                       bg=self.colors['error'], fg=self.colors['text_primary'],
                                       font=("Arial", 10), relief=tk.FLAT,
                                       activebackground=self.colors['accent'],
                                       activeforeground=self.colors['text_primary'])
        self.cancel_button.pack(pady=5, padx=20, fill=tk.X)
        self.cancel_button.config(state=tk.DISABLED)

        # Текстовое поле для отображения диалога с подсветкой кода
        self.text_widget = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state=tk.DISABLED,
            width=80, height=25, font=("Consolas", 10),
            bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
            insertbackground=self.colors['text_primary'],
            selectbackground=self.colors['accent'],
            selectforeground=self.colors['text_primary']
        )
        self.text_widget.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        # Поле для ввода текста
        self.text_entry = tk.Text(self.root, width=80, height=5, font=("Arial", 11),
                                  bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                                  insertbackground=self.colors['text_primary'],
                                  selectbackground=self.colors['accent'],
                                  selectforeground=self.colors['text_primary'],
                                  relief=tk.FLAT)
        self.text_entry.pack(pady=10, padx=20, fill=tk.X)

        # Кнопки управления
        button_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        button_frame.pack(pady=10, padx=20, fill=tk.X)

        self.send_button = tk.Button(button_frame, text="📤 Отправить текст",
                                     command=self.app.send_text_to_ai,
                                     bg=self.colors['success'], fg=self.colors['text_primary'],
                                     font=("Arial", 11, "bold"), relief=tk.FLAT,
                                     activebackground=self.colors['accent'],
                                     activeforeground=self.colors['text_primary'])
        self.send_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.export_button = tk.Button(button_frame, text="💾 Экспорт",
                                       command=self.app.export_conversation,
                                       bg=self.colors['bg_tertiary'], fg=self.colors['text_primary'],
                                       font=("Arial", 11), relief=tk.FLAT,
                                       activebackground=self.colors['accent'],
                                       activeforeground=self.colors['text_primary'])
        self.export_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.clear_button = tk.Button(button_frame, text="🗑️ Очистить",
                                      command=self.app.clear_conversation,
                                      bg=self.colors['warning'], fg=self.colors['text_primary'],
                                      font=("Arial", 11), relief=tk.FLAT,
                                      activebackground=self.colors['accent'],
                                      activeforeground=self.colors['text_primary'])
        self.clear_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Кнопка смены промпта
        self.prompt_button = tk.Button(button_frame, text="🤖 Сменить помощника",
                                       command=self.app.show_prompt_selector,
                                       bg=self.colors['bg_tertiary'], fg=self.colors['text_primary'],
                                       font=("Arial", 11), relief=tk.FLAT,
                                       activebackground=self.colors['accent'],
                                       activeforeground=self.colors['text_primary'])
        self.prompt_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def setup_hotkeys(self):
        """Настройка горячих клавиш"""
        if self.app.settings.settings.get("hotkeys_enabled", True):
            self.root.bind('<Control-r>', lambda e: self.app.start_mic_recording())
            self.root.bind('<Control-c>', lambda e: self.app.start_computer_recording())
            self.root.bind('<Control-s>', lambda e: self.app.send_text_to_ai())
            self.root.bind('<Escape>', lambda e: self.app.cancel_recording())
            self.root.bind('<Control-e>', lambda e: self.app.export_conversation())
            self.root.bind('<Control-p>', lambda e: self.app.show_prompt_selector())

    def update_recording_status(self, is_recording, device_type=None):
        """Обновление статуса записи"""
        if is_recording:
            self.recording_label.config(text="🔴 ЗАПИСЬ...", fg=self.colors['error'])
            self.cancel_button.config(state=tk.NORMAL)
            
            # Делаем активной только ту кнопку, которая была нажата
            if device_type == 'mic':
                self.mic_button.config(
                    text="⏹️ Остановить запись (микрофон)",
                    bg=self.colors['error'],  # Красный цвет для привлечения внимания
                    fg=self.colors['text_primary'],
                    state=tk.NORMAL  # Оставляем активной для остановки записи
                )
                # Другая кнопка остается неактивной
                self.computer_button.config(
                    text="💻 Запись с компьютера",
                    bg=self.colors['bg_tertiary'],
                    fg=self.colors['text_secondary'],
                    state=tk.DISABLED
                )
            elif device_type == 'computer':
                self.computer_button.config(
                    text="⏹️ Остановить запись (компьютер)",
                    bg=self.colors['error'],  # Красный цвет для привлечения внимания
                    fg=self.colors['text_primary'],
                    state=tk.NORMAL  # Оставляем активной для остановки записи
                )
                # Другая кнопка остается неактивной
                self.mic_button.config(
                    text="🎤 Запись с микрофона",
                    bg=self.colors['bg_tertiary'],
                    fg=self.colors['text_secondary'],
                    state=tk.DISABLED
                )
        else:
            self.recording_label.config(text="", fg=self.colors['text_primary'])
            self.cancel_button.config(state=tk.DISABLED)
            
            # Возвращаем обе кнопки в исходное состояние
            self.mic_button.config(
                text="🎤 Запись с микрофона",
                bg=self.colors['bg_tertiary'],
                fg=self.colors['text_primary'],
                state=tk.NORMAL
            )
            self.computer_button.config(
                text="💻 Запись с компьютера",
                bg=self.colors['bg_tertiary'],
                fg=self.colors['text_primary'],
                state=tk.NORMAL
            )

    def show_progress(self, message):
        """Показать окно прогресса"""
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Обработка")
        self.progress_window.geometry("300x100")
        self.progress_window.transient(self.root)
        self.progress_window.grab_set()
        self.progress_window.configure(bg=self.colors['bg_primary'])

        self.progress_label = tk.Label(self.progress_window, text=message,
                                       font=("Arial", 12),
                                       bg=self.colors['bg_primary'],
                                       fg=self.colors['text_primary'])
        self.progress_label.pack(expand=True)

        # Центрирование окна
        self.progress_window.update_idletasks()
        x = (self.progress_window.winfo_screenwidth() // 2) - (300 // 2)
        y = (self.progress_window.winfo_screenheight() // 2) - (100 // 2)
        self.progress_window.geometry(f"300x100+{x}+{y}")

    def hide_progress(self):
        """Скрыть окно прогресса"""
        if hasattr(self, 'progress_window') and self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None

    def update_progress_message(self, message):
        """Обновить сообщение в окне прогресса"""
        if self.progress_label:
            self.progress_label.config(text=message)

    def highlight_code(self, text):
        """Подсветка кода в тексте"""
        # Паттерны для подсветки
        patterns = {
            'keyword': r'\b(def|class|import|from|as|if|else|elif|for|while|try|except|finally|with|return|True|False|None)\b',
            'string': r'"[^"]*"|\'[^\']*\'',
            'comment': r'#.*$',
            'function': r'\b\w+(?=\()',
            'number': r'\b\d+\.?\d*\b',
        }

        # Создаем теги для подсветки
        for tag_name, color in [
            ('keyword', self.colors['code_keyword']),
            ('string', self.colors['code_string']),
            ('comment', self.colors['code_comment']),
            ('function', self.colors['code_text']),
            ('number', self.colors['code_text']),
        ]:
            self.text_widget.tag_configure(tag_name, foreground=color)

        # Применяем подсветку
        for pattern_name, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.MULTILINE):
                start = f"1.0+{match.start()}c"
                end = f"1.0+{match.end()}c"
                self.text_widget.tag_add(pattern_name, start, end)

    def update_text_widget(self, user_text, response):
        """Обновить текстовое поле с диалогом и подсветкой кода"""
        self.text_widget.config(state=tk.NORMAL)

        # Добавляем сообщение пользователя
        self.text_widget.insert(tk.END, f"👤 User: {user_text}\n")

        # Добавляем ответ AI с подсветкой кода
        self.text_widget.insert(tk.END, f"🤖 AI: {response}\n\n")

        # Применяем подсветку кода к последнему добавленному тексту
        # Получаем позицию начала ответа AI
        start_pos = self.text_widget.index("end-2l linestart")
        end_pos = self.text_widget.index("end-1l")

        # Извлекаем текст ответа для подсветки
        response_text = self.text_widget.get(start_pos, end_pos)
        if "🤖 AI: " in response_text:
            response_text = response_text.replace("🤖 AI: ", "")

        # Очищаем теги в области ответа
        self.text_widget.tag_remove("keyword", start_pos, end_pos)
        self.text_widget.tag_remove("string", start_pos, end_pos)
        self.text_widget.tag_remove("comment", start_pos, end_pos)
        self.text_widget.tag_remove("function", start_pos, end_pos)
        self.text_widget.tag_remove("number", start_pos, end_pos)

        # Применяем подсветку кода
        self.highlight_code_in_range(start_pos, end_pos, response_text)

        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def highlight_code_in_range(self, start_pos, end_pos, text):
        """Подсветка кода в определенном диапазоне"""
        patterns = {
            'keyword': r'\b(def|class|import|from|as|if|else|elif|for|while|try|except|finally|with|return|True|False|None)\b',
            'string': r'"[^"]*"|\'[^\']*\'',
            'comment': r'#.*$',
            'function': r'\b\w+(?=\()',
            'number': r'\b\d+\.?\d*\b',
        }

        # Создаем теги для подсветки
        for tag_name, color in [
            ('keyword', self.colors['code_keyword']),
            ('string', self.colors['code_string']),
            ('comment', self.colors['code_comment']),
            ('function', self.colors['code_text']),
            ('number', self.colors['code_text']),
        ]:
            self.text_widget.tag_configure(tag_name, foreground=color)

        # Применяем подсветку
        for pattern_name, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.MULTILINE):
                match_start = f"{start_pos}+{match.start()}c"
                match_end = f"{start_pos}+{match.end()}c"
                self.text_widget.tag_add(pattern_name, match_start, match_end)

    def clear_text_entry(self):
        """Очистить поле ввода текста"""
        self.text_entry.delete("1.0", tk.END)

    def clear_text_widget(self):
        """Очистить текстовое поле диалога"""
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.config(state=tk.DISABLED)

    @staticmethod
    def show_error(title, message):
        """Показать окно ошибки"""
        messagebox.showerror(title, message)

    @staticmethod
    def show_warning(title, message):
        """Показать окно предупреждения"""
        messagebox.showwarning(title, message)

    @staticmethod
    def show_info(title, message):
        """Показать информационное окно"""
        messagebox.showinfo(title, message)

    @staticmethod
    def ask_yes_no(title, message):
        """Показать диалог подтверждения"""
        return messagebox.askyesno(title, message)

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
