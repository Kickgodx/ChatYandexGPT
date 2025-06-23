import logging
import threading
import time
import warnings

# Подавляем предупреждение Pydantic
warnings.filterwarnings("ignore", message=".*protected_namespaces.*")

from src.audio_processor import AudioProcessor
from src.audio_recorder import AudioRecorder
from src.config import Config
from src.conversation_manager import ConversationManager
from src.env_loader import get_env_var, get_required_env_var
from src.gui import GUI
from src.prompt_manager import PromptManager
from src.settings import Settings
from src.speech_recognizer import get_global_recognizer, preload_vosk_model
from src.utils import setup_logging, ensure_output_directory
from yandexchat_bot import ChatYandexGPTBot


class AIAudioRecorderApp:
    """Основной класс приложения AI Audio Recorder"""

    def __init__(self):
        setup_logging()

        # Загрузка credentials из переменных окружения
        try:
            folder_id = get_required_env_var("YANDEX_FOLDER_ID")
            iam_token = get_env_var("YANDEX_IAM_TOKEN", None)
            api_key = get_env_var("YANDEX_API_KEY", None)
            vosk_model_path = get_env_var("VOSK_MODEL_PATH", "./vosk-model-ru-0.42")
        except ValueError as e:
            logging.error(f"Ошибка загрузки credentials: {e}")
            raise

        # Инициализация компонентов
        self.config = Config()
        self.settings = Settings()
        self.audio_recorder = AudioRecorder(self.config, self.settings)
        self.audio_processor = AudioProcessor(self.config)

        # Создание GUI (должно быть до инициализации распознавателя)
        self.gui = GUI(self)
        self.gui.create_gui()
        self.gui.setup_hotkeys()

        # Оптимизированная инициализация распознавателя речи с глобальным кэшированием
        print("🚀 Инициализация компонентов...")
        self.gui.update_vosk_status("Инициализация...", False, "Подготовка к загрузке модели Vosk")

        # Используем глобальный синглтон для предотвращения повторной загрузки
        try:
            # Получаем глобальный распознаватель (создается если не существует)
            self.speech_recognizer = get_global_recognizer(vosk_model_path, preload=False)

            # Если модель еще не загружена, запускаем загрузку в фоне
            if not self.speech_recognizer.is_model_loaded():
                self.gui.update_vosk_status("Загрузка в фоне...", False, "Модель загружается в фоновом режиме")

                # Запускаем предварительную загрузку в фоне
                preload_vosk_model(vosk_model_path, show_progress=False)

                # Запускаем мониторинг загрузки модели
                self._start_model_monitoring(vosk_model_path)
            else:
                # Модель уже загружена
                model_info = self.speech_recognizer.get_model_info()
                size_mb = model_info.get('size_mb', 0)
                details = f"(Размер: {size_mb:.1f} МБ)"
                self.gui.update_vosk_status("Готово", True, details)
                logging.info("Vosk model already loaded")

        except Exception as e:
            error_msg = f"Ошибка инициализации модели: {str(e)}"
            self.gui.update_vosk_status("Ошибка", False, error_msg)
            logging.error(f"Error initializing speech recognizer: {e}")

        # Инициализация бота с промптом по умолчанию
        print("🤖 Инициализация YandexGPT бота...")
        if iam_token is not None and iam_token != "":
            self.bot = ChatYandexGPTBot(iam_token=iam_token, folder_id=folder_id, model_name="yandexgpt-lite")
        else:
            self.bot = ChatYandexGPTBot(api_key=api_key, folder_id=folder_id, model_name="yandexgpt-lite")

        self.conversation_manager = ConversationManager(self.config, self.bot)

        # Инициализация менеджера промптов
        self.prompt_manager = PromptManager(self)
        self.prompt_manager.add_prompt_status_to_gui()

        logging.info("Application initialized successfully")

    def _start_model_monitoring(self, model_path: str):
        """Запуск мониторинга загрузки модели Vosk"""

        def monitor_loading():
            max_wait_time = 180  # Максимальное время ожидания в секундах
            check_interval = 0.5  # Интервал проверки в секундах
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                try:
                    # Получаем актуальный экземпляр распознавателя
                    current_recognizer = get_global_recognizer(model_path, preload=False)

                    if current_recognizer.is_model_loaded():
                        # Модель загружена успешно
                        model_info = current_recognizer.get_model_info()
                        size_mb = model_info.get('size_mb', 0)
                        details = f"(Размер: {size_mb:.1f} МБ)"
                        self.gui.update_vosk_status("Готово", True, details)
                        logging.info("Vosk model loaded successfully")
                        return
                    else:
                        # Модель еще загружается
                        elapsed = time.time() - start_time
                        self.gui.update_vosk_status(
                            "Загрузка...",
                            False,
                            f"({elapsed:.1f} сек)"
                        )

                except Exception as e:
                    error_msg = f"Ошибка мониторинга: {str(e)}"
                    self.gui.update_vosk_status("Ошибка", False, error_msg)
                    logging.error(f"Error monitoring model loading: {e}")
                    return

                time.sleep(check_interval)

            # Таймаут
            self.gui.update_vosk_status("Таймаут", False, "Превышено время ожидания загрузки модели")
            logging.warning("Model loading timeout")

        # Запускаем мониторинг в отдельном потоке
        monitor_thread = threading.Thread(target=monitor_loading, daemon=True)
        monitor_thread.start()

    def start_mic_recording(self):
        """Начало записи с микрофона"""
        if not self.audio_recorder.is_recording:
            try:
                self.audio_recorder.start_recording('mic')
                self.gui.update_recording_status(True, 'mic')
            except Exception as e:
                self.gui.show_error("Ошибка", f"Не удалось начать запись: {str(e)}")
                logging.error(f"Error starting mic recording: {e}")
        else:
            self.stop_recording()

    def start_computer_recording(self):
        """Начало записи с компьютера"""
        if not self.audio_recorder.is_recording:
            try:
                self.audio_recorder.start_recording('computer')
                self.gui.update_recording_status(True, 'computer')
            except Exception as e:
                self.gui.show_error("Ошибка", f"Не удалось начать запись: {str(e)}")
                logging.error(f"Error starting computer recording: {e}")
        else:
            self.stop_recording()

    def stop_recording(self):
        """Остановка записи"""
        if self.audio_recorder.is_recording:
            frames = self.audio_recorder.stop_recording()
            self.gui.update_recording_status(False)

            if frames:
                self.process_audio(frames)

    def cancel_recording(self):
        """Отмена записи"""
        if self.audio_recorder.is_recording:
            self.audio_recorder.stop_recording()
            self.gui.update_recording_status(False)
            self.gui.show_info("Информация", "Запись отменена")

    def process_audio(self, frames):
        """Обработка записанного аудио с live-транскрипцией"""

        def process():
            try:
                # Получаем актуальный экземпляр распознавателя
                current_recognizer = get_global_recognizer(self.speech_recognizer.model_path, preload=False)

                # Проверяем, загружена ли модель перед обработкой
                if not current_recognizer.is_model_loaded():
                    self.gui.show_error("Ошибка", "Модель распознавания речи еще не загружена. Подождите немного.")
                    return

                self.gui.show_progress("Сохранение аудио...")
                self.audio_processor.save_audio(frames, self.config.WAVE_OUTPUT_FILENAME)

                self.gui.update_progress_message("Нормализация аудио...")
                self.audio_processor.normalize_audio(self.config.WAVE_OUTPUT_FILENAME)

                # Проверка качества аудио
                audio_data = b''.join(frames)
                quality_ok, quality_msg = self.audio_recorder.check_audio_quality(audio_data)

                # Показываем предупреждение только если качество действительно плохое
                if not quality_ok:
                    self.gui.show_warning("Предупреждение", quality_msg)

                # Начинаем live-транскрипцию
                self.gui.root.after(0, self.gui.start_live_transcription)
                self.gui.update_progress_message("Распознавание речи...")

                # Используем live-транскрипцию с callback (модель уже загружена)
                transcribed_text = current_recognizer.transcribe_audio(
                    self.config.WAVE_OUTPUT_FILENAME,
                    live_callback=self.live_transcription_callback
                )

                if not transcribed_text.strip():
                    self.gui.show_warning("Предупреждение", "Речь не распознана. Попробуйте говорить четче.")
                    self.gui.hide_progress()
                    return

                self.gui.update_progress_message("Получение ответа от AI...")
                response = self.bot.get_response(transcribed_text)

                # Добавление в историю
                self.conversation_manager.add_message(transcribed_text, response)
                self.conversation_manager.save_conversation()

                # Обновление интерфейса
                self.gui.root.after(0, lambda: self.gui.update_text_widget(transcribed_text, response))

                self.gui.hide_progress()
                logging.info(f"Processed audio: '{transcribed_text}' -> '{response[:50]}...'")

            except Exception as e:
                self.gui.hide_progress()
                self.gui.show_error("Ошибка", f"Ошибка обработки аудио: {str(e)}")
                logging.error(f"Error processing audio: {e}")

        threading.Thread(target=process, daemon=True).start()

    def live_transcription_callback(self, text, is_final):
        """
        Callback для live-транскрипции

        Args:
            text: Распознанный текст
            is_final: True если это финальный результат
        """
        # Обновляем GUI в главном потоке
        self.gui.root.after(0, lambda: self.gui.update_live_transcription(text, is_final))

    def send_text_to_ai(self):
        """Отправка текста в AI"""
        user_text = self.gui.text_entry.get("1.0", "end-1c").strip()
        if user_text:
            def send():
                try:
                    self.gui.show_progress("Получение ответа от AI...")
                    response = self.bot.get_response(user_text)

                    # Добавление в историю
                    self.conversation_manager.add_message(user_text, response)
                    self.conversation_manager.save_conversation()

                    # Обновление интерфейса
                    self.gui.root.after(0, lambda: self.gui.update_text_widget(user_text, response))
                    self.gui.root.after(0, lambda: self.gui.clear_text_entry())

                    self.gui.hide_progress()
                    logging.info(f"Text sent: '{user_text}' -> '{response[:50]}...'")

                except Exception as e:
                    self.gui.hide_progress()
                    self.gui.show_error("Ошибка", f"Ошибка получения ответа: {str(e)}")
                    logging.error(f"Error sending text: {e}")

            threading.Thread(target=send, daemon=True).start()

    def export_conversation(self):
        """Экспорт диалога"""
        try:
            filename = self.conversation_manager.export_conversation('txt')
            self.gui.show_info("Экспорт", f"Диалог экспортирован в файл: {filename}")
        except Exception as e:
            self.gui.show_error("Ошибка", f"Ошибка экспорта: {str(e)}")

    def clear_conversation(self):
        """Очистка истории диалога"""
        if self.gui.ask_yes_no("Подтверждение", "Очистить историю диалога?"):
            self.gui.clear_text_widget()
            self.conversation_manager.conversation_history.clear()

    def show_prompt_selector(self):
        """Показать окно выбора промпта"""
        self.prompt_manager.show_prompt_selector()

    def get_conversation_summary(self):
        """Получить краткое резюме диалога"""
        return self.bot.get_conversation_summary()

    def run(self):
        """Запуск приложения"""
        ensure_output_directory()
        logging.info("Starting AI Audio Recorder application")
        self.gui.root.mainloop()


if __name__ == "__main__":
    app = AIAudioRecorderApp()
    app.run()
