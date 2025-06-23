#!/usr/bin/env python3
"""
ChatYandexGPT с современным PyQt6 интерфейсом
"""

import logging
import sys
import threading
import time
import warnings

# Подавляем предупреждение Pydantic
warnings.filterwarnings("ignore", message=".*protected_namespaces.*")

from PyQt6.QtWidgets import QApplication

from src.audio_processor import AudioProcessor
from src.audio_recorder import AudioRecorder
from src.config import Config
from src.conversation_manager import ConversationManager
from src.env_loader import get_env_var, get_required_env_var
from src.qt_gui.modern_gui import ModernGUI
from src.qt_gui.prompt_manager_qt import PromptManagerQt
from src.settings import Settings
from src.speech_recognizer import get_global_recognizer, preload_vosk_model
from src.utils import setup_logging, ensure_output_directory
from yandexchat_bot import ChatYandexGPTBot


class AIAudioRecorderApp:
    """Основной класс приложения AI Audio Recorder с PyQt6"""

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
        self.gui = ModernGUI(self)

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
        self.prompt_manager = PromptManagerQt(self)
        # Статус промпта уже отображается в PromptStatusWidget в PyQt6 версии

        # Обновляем статус промпта в GUI
        current_prompt = self.bot.get_current_prompt_info()
        current_model = self.bot.get_current_model_info()
        self.gui.update_prompt_status(current_prompt['name'], current_model['name'])

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

    def start_recording(self, device_type: str):
        """Начало записи с указанного устройства"""
        if not self.audio_recorder.is_recording:
            try:
                self.audio_recorder.start_recording(device_type)
                # Обновляем GUI через виджет
                self.gui.recording_widget.update_recording_status(True, device_type)
            except Exception as e:
                self.gui.show_error("Ошибка", f"Не удалось начать запись: {str(e)}")
                logging.error(f"Error starting {device_type} recording: {e}")
        else:
            self.stop_recording()

    def stop_recording(self):
        """Остановка записи"""
        if self.audio_recorder.is_recording:
            frames = self.audio_recorder.stop_recording()
            # Обновляем GUI через виджет
            self.gui.recording_widget.update_recording_status(False)

            logging.info(
                f"Stop recording called, frames received: {frames is not None}, frames count: {len(frames) if frames else 0}")

            if frames:
                logging.info(f"Starting audio processing with {len(frames)} frames")
                self.process_audio(frames)
            else:
                logging.warning("No frames received from audio recorder")
        else:
            logging.info("Stop recording called but not currently recording")

    def cancel_recording(self):
        """Отмена записи"""
        if self.audio_recorder.is_recording:
            self.audio_recorder.stop_recording()
            # Обновляем GUI через виджет
            self.gui.recording_widget.update_recording_status(False)
            self.gui.show_info("Информация", "Запись отменена")

    def process_audio(self, frames):
        """Обработка записанного аудио с live-транскрипцией"""
        logging.info(f"Process audio started with {len(frames)} frames")

        def process():
            try:
                logging.info("Audio processing thread started")

                # Получаем актуальный экземпляр распознавателя
                current_recognizer = get_global_recognizer(self.speech_recognizer.model_path, preload=False)
                logging.info(f"Got recognizer, model loaded: {current_recognizer.is_model_loaded()}")

                # Проверяем, загружена ли модель перед обработкой
                if not current_recognizer.is_model_loaded():
                    logging.error("Model not loaded, cannot process audio")
                    self.gui.show_error_signal.emit("Ошибка",
                                                    "Модель распознавания речи еще не загружена. Подождите немного.")
                    return

                logging.info("Saving audio...")
                self.gui.show_progress_signal.emit("Сохранение аудио...")
                self.audio_processor.save_audio(frames, self.config.WAVE_OUTPUT_FILENAME)

                logging.info("Normalizing audio...")
                self.gui.update_progress_message_signal.emit("Нормализация аудио...")
                self.audio_processor.normalize_audio(self.config.WAVE_OUTPUT_FILENAME)

                # Проверка качества аудио
                audio_data = b''.join(frames)
                quality_ok, quality_msg = self.audio_recorder.check_audio_quality(audio_data)
                logging.info(f"Audio quality check: {quality_ok}, message: {quality_msg}")

                # Показываем предупреждение только если качество действительно плохое
                if not quality_ok:
                    self.gui.show_warning_signal.emit("Предупреждение", quality_msg)

                # Начинаем live-транскрипцию
                logging.info("Starting transcription...")
                self.gui.clear_live_transcription_signal.emit()
                self.gui.update_progress_message_signal.emit("Распознавание речи...")

                # Используем live-транскрипцию с callback (модель уже загружена)
                transcribed_text = current_recognizer.transcribe_audio(
                    self.config.WAVE_OUTPUT_FILENAME,
                    live_callback=self.live_transcription_callback
                )

                logging.info(f"Transcription result: '{transcribed_text}'")

                if not transcribed_text.strip():
                    logging.warning("No text transcribed from audio")
                    self.gui.show_warning_signal.emit("Предупреждение",
                                                      "Речь не распознана. Попробуйте говорить четче.")
                    self.gui.hide_progress_signal.emit()
                    return

                logging.info("Getting AI response...")
                self.gui.update_progress_message_signal.emit("Получение ответа от AI...")
                response = self.bot.get_response(transcribed_text)

                # Добавление в историю
                self.conversation_manager.add_message(transcribed_text, response)
                self.conversation_manager.save_conversation()

                # Обновление интерфейса через сигналы
                self.gui.update_chat_signal.emit(transcribed_text, response)

                self.gui.hide_progress_signal.emit()
                logging.info(f"Processed audio: '{transcribed_text}' -> '{response[:50]}...'")

            except Exception as e:
                logging.error(f"Error in audio processing: {e}", exc_info=True)
                self.gui.hide_progress_signal.emit()
                self.gui.show_error_signal.emit("Ошибка", f"Ошибка обработки аудио: {str(e)}")

        threading.Thread(target=process, daemon=True).start()

    def live_transcription_callback(self, text, is_final):
        """
        Callback для live-транскрипции

        Args:
            text: Распознанный текст
            is_final: True если это финальный результат
        """
        # Обновляем GUI через сигналы
        self.gui.update_live_transcription_signal.emit(text, is_final)

    def send_text_to_ai(self, user_text: str):
        """Отправка текста в AI"""
        if user_text.strip():
            def send():
                try:
                    self.gui.show_progress_signal.emit("Получение ответа от AI...")
                    response = self.bot.get_response(user_text)

                    # Добавление в историю
                    self.conversation_manager.add_message(user_text, response)
                    self.conversation_manager.save_conversation()

                    # Обновление интерфейса через сигналы
                    self.gui.update_chat_signal.emit(user_text, response)
                    self.gui.clear_text_input_signal.emit()

                    self.gui.hide_progress_signal.emit()
                    logging.info(f"Text sent: '{user_text}' -> '{response[:50]}...'")

                except Exception as e:
                    self.gui.hide_progress_signal.emit()
                    self.gui.show_error_signal.emit("Ошибка", f"Ошибка получения ответа: {str(e)}")
                    logging.error(f"Error sending text: {e}")

            threading.Thread(target=send, daemon=True).start()

    def export_conversation(self):
        """Экспорт диалога"""
        try:
            filename = self.conversation_manager.export_conversation('txt')
            self.gui.show_info_signal.emit("Экспорт", f"Диалог экспортирован в файл: {filename}")
        except Exception as e:
            self.gui.show_error_signal.emit("Ошибка", f"Ошибка экспорта: {str(e)}")

    def clear_conversation(self):
        """Очистка истории диалога"""
        if self.gui.ask_yes_no("Подтверждение", "Очистить историю диалога?"):
            self.gui.clear_chat_signal.emit()
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
        logging.info("Starting AI Audio Recorder application with PyQt6")
        self.gui.show()


def main():
    """Основная функция"""
    # Создаем QApplication
    app = QApplication(sys.argv)

    # Создаем и запускаем наше приложение
    ai_app = AIAudioRecorderApp()
    ai_app.run()

    # Запускаем главный цикл событий
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
