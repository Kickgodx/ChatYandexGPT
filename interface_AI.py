import logging
import threading

from yandex_creds import iam_token, folder_id, path_to_vosk_model, api_key
from yandexchat_bot import ChatYandexGPTBot

from src.utils import setup_logging, ensure_output_directory
from src.config import Config
from src.settings import Settings
from src.audio_recorder import AudioRecorder
from src.audio_processor import AudioProcessor
from src.speech_recognizer import SpeechRecognizer
from src.conversation_manager import ConversationManager
from src.gui import GUI
from src.prompt_manager import PromptManager


class AIAudioRecorderApp:
    """Основной класс приложения AI Audio Recorder"""
    
    def __init__(self):
        setup_logging()

        # Инициализация компонентов
        self.config = Config()
        self.settings = Settings()
        self.audio_recorder = AudioRecorder(self.config, self.settings)
        self.audio_processor = AudioProcessor(self.config)
        self.speech_recognizer = SpeechRecognizer(path_to_vosk_model)

        # Инициализация бота с промптом по умолчанию
        if iam_token != "":
            self.bot = ChatYandexGPTBot(iam_token=iam_token, folder_id=folder_id, model_name="yandexgpt-lite")
        else:
            self.bot = ChatYandexGPTBot(api_key=api_key, folder_id=folder_id, model_name="yandexgpt-lite")

        self.conversation_manager = ConversationManager(self.config, self.bot)

        # Создание GUI
        self.gui = GUI(self)
        self.gui.create_gui()
        self.gui.setup_hotkeys()
        
        # Инициализация менеджера промптов
        self.prompt_manager = PromptManager(self)
        self.prompt_manager.add_prompt_status_to_gui()

        logging.info("Application initialized successfully")

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
        """Обработка записанного аудио"""
        def process():
            try:
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

                self.gui.update_progress_message("Распознавание речи...")
                transcribed_text = self.speech_recognizer.transcribe_audio(self.config.WAVE_OUTPUT_FILENAME)

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
