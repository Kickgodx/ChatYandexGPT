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

    def create_gui(self):
        """Создание графического интерфейса"""
        self.root = tk.Tk()
        self.root.title("AI Audio Recorder")
        self.root.geometry("700x800")

        # Кнопки записи
        self.mic_button = tk.Button(self.root, text="🎤 Record from Microphone",
                                    command=self.app.start_mic_recording, bg="#4CAF50", fg="white")
        self.mic_button.pack(pady=10)

        self.computer_button = tk.Button(self.root, text="💻 Record from Computer",
                                         command=self.app.start_computer_recording, bg="#2196F3", fg="white")
        self.computer_button.pack(pady=10)

        # Индикатор записи
        self.recording_label = tk.Label(self.root, text="", fg="red", font=("Arial", 12, "bold"))
        self.recording_label.pack(pady=5)

        # Кнопка отмены записи
        self.cancel_button = tk.Button(self.root, text="❌ Cancel Recording",
                                       command=self.app.cancel_recording, bg="#f44336", fg="white")
        self.cancel_button.pack(pady=5)
        self.cancel_button.config(state=tk.DISABLED)

        # Текстовое поле для отображения диалога
        self.text_widget = scrolledtext.ScrolledText(self.root, wrap=tk.WORD,
                                                     state=tk.DISABLED, width=80, height=25)
        self.text_widget.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Поле для ввода текста
        self.text_entry = tk.Text(self.root, width=80, height=5)
        self.text_entry.pack(pady=10, padx=10)

        # Кнопки управления
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.send_button = tk.Button(button_frame, text="📤 Send Text to AI",
                                     command=self.app.send_text_to_ai, bg="#FF9800", fg="white")
        self.send_button.pack(side=tk.LEFT, padx=5)

        self.export_button = tk.Button(button_frame, text="💾 Export Conversation",
                                       command=self.app.export_conversation, bg="#9C27B0", fg="white")
        self.export_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(button_frame, text="🗑️ Clear",
                                      command=self.app.clear_conversation, bg="#607D8B", fg="white")
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Кнопка смены промпта
        self.prompt_button = tk.Button(button_frame, text="🤖 Change Assistant",
                                       command=self.app.show_prompt_selector, bg="#E91E63", fg="white")
        self.prompt_button.pack(side=tk.LEFT, padx=5)

    def setup_hotkeys(self):
        """Настройка горячих клавиш"""
        if self.app.settings.settings.get("hotkeys_enabled", True):
            self.root.bind('<Control-r>', lambda e: self.app.start_mic_recording())
            self.root.bind('<Control-c>', lambda e: self.app.start_computer_recording())
            self.root.bind('<Control-s>', lambda e: self.app.send_text_to_ai())
            self.root.bind('<Escape>', lambda e: self.app.cancel_recording())
            self.root.bind('<Control-e>', lambda e: self.app.export_conversation())
            self.root.bind('<Control-p>', lambda e: self.app.show_prompt_selector())

    def update_recording_status(self, is_recording):
        """Обновление статуса записи"""
        if is_recording:
            self.recording_label.config(text="🔴 ЗАПИСЬ...", fg="red")
            self.cancel_button.config(state=tk.NORMAL)
            self.mic_button.config(state=tk.DISABLED)
            self.computer_button.config(state=tk.DISABLED)
        else:
            self.recording_label.config(text="", fg="black")
            self.cancel_button.config(state=tk.DISABLED)
            self.mic_button.config(state=tk.NORMAL)
            self.computer_button.config(state=tk.NORMAL)

    def show_progress(self, message):
        """Показать окно прогресса"""
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Обработка")
        self.progress_window.geometry("300x100")
        self.progress_window.transient(self.root)
        self.progress_window.grab_set()

        self.progress_label = tk.Label(self.progress_window, text=message, font=("Arial", 12))
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

    def update_text_widget(self, user_text, response):
        """Обновить текстовое поле с диалогом"""
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, f"👤 User: {user_text}\n")
        self.text_widget.insert(tk.END, f"🤖 AI: {response}\n\n")
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

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
