import json
import logging
import os
import threading
import time
import wave
from typing import Optional, Callable

from vosk import Model, KaldiRecognizer


class SpeechRecognizer:
    """Класс для распознавания речи с оптимизированной загрузкой модели"""

    def __init__(self, model_path: str, preload_model: bool = True, show_progress: bool = True):
        """
        Инициализация распознавателя речи

        Args:
            model_path: Путь к модели Vosk
            preload_model: Загружать ли модель сразу при инициализации
            show_progress: Показывать ли прогресс загрузки
        """
        self.model_path = model_path
        self.model = None
        self._model_loaded = False
        self._loading_lock = threading.Lock()

        if preload_model:
            self._load_model(show_progress)

    def _load_model(self, show_progress: bool = True) -> None:
        """Загрузка модели Vosk с оптимизациями"""
        if self._model_loaded:
            return

        with self._loading_lock:
            if self._model_loaded:  # Двойная проверка
                return

            try:
                if show_progress:
                    print("🔄 Загрузка модели Vosk...")
                    start_time = time.time()

                # Проверяем существование модели
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Модель Vosk не найдена: {self.model_path}")

                # Загружаем модель
                self.model = Model(self.model_path)
                self._model_loaded = True

                if show_progress:
                    load_time = time.time() - start_time
                    print(f"✅ Модель Vosk загружена за {load_time:.2f} секунд")

                logging.info(f"Vosk model loaded from {self.model_path}")

            except Exception as e:
                logging.error(f"Error loading Vosk model: {e}")
                raise

    def _ensure_model_loaded(self) -> None:
        """Убедиться, что модель загружена"""
        if not self._model_loaded:
            self._load_model()

    def preload_model(self, show_progress: bool = True) -> None:
        """
        Предварительная загрузка модели в отдельном потоке

        Args:
            show_progress: Показывать ли прогресс загрузки
        """
        if self._model_loaded:
            return

        def load_in_background():
            try:
                self._load_model(show_progress)
            except Exception as e:
                logging.error(f"Background model loading failed: {e}")

        thread = threading.Thread(target=load_in_background, daemon=True)
        thread.start()

    def is_model_loaded(self) -> bool:
        """Проверить, загружена ли модель"""
        return self._model_loaded

    def get_model_info(self) -> dict:
        """Получить информацию о модели"""
        if not self._model_loaded:
            return {"status": "not_loaded", "path": self.model_path}

        return {
            "status": "loaded",
            "path": self.model_path,
            "size_mb": self._get_model_size()
        }

    def _get_model_size(self) -> float:
        """Получить размер модели в МБ"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.model_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size / (1024 * 1024)  # Конвертируем в МБ
        except:
            return 0.0

    def transcribe_audio(self, file_path: str, live_callback: Optional[Callable[[str, bool], None]] = None) -> str:
        """
        Транскрипция аудио в текст с поддержкой live-отображения

        Args:
            file_path: Путь к аудиофайлу
            live_callback: Функция для обновления GUI в реальном времени
                          Принимает параметр: (text, is_final)
        """
        # Убеждаемся, что модель загружена
        self._ensure_model_loaded()

        with wave.open(file_path, "rb") as wf:
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)

            logging.info("Starting transcription...")
            result_text = ""
            last_partial = ""

            while True:
                data = wf.readframes(1024)
                if len(data) == 0:
                    break

                rec.AcceptWaveform(data)

                # Получаем промежуточный результат для live-отображения
                if live_callback:
                    partial_result = rec.PartialResult()
                    partial_dict = json.loads(partial_result)
                    partial_text = partial_dict.get('partial', '')

                    # Обновляем GUI только если текст изменился
                    if partial_text != last_partial:
                        live_callback(partial_text, False)  # False = не финальный результат
                        last_partial = partial_text

            # Финальный результат
            result = rec.FinalResult()
            result_dict = json.loads(result)
            result_text = result_dict.get('text', '')

            # Отправляем финальный результат
            if live_callback:
                live_callback(result_text, True)  # True = финальный результат

        return result_text

    def transcribe_audio_live(self, file_path: str, update_callback: Callable[[str], None],
                              final_callback: Callable[[str], None]) -> threading.Thread:
        """
        Транскрипция аудио с отдельными callback для промежуточных и финальных результатов

        Args:
            file_path: Путь к аудиофайлу
            update_callback: Функция для обновления промежуточных результатов
            final_callback: Функция для обработки финального результата
        """
        # Убеждаемся, что модель загружена
        self._ensure_model_loaded()

        def process_audio():
            try:
                with wave.open(file_path, "rb") as wf:
                    rec = KaldiRecognizer(self.model, wf.getframerate())
                    rec.SetWords(True)

                    logging.info("Starting live transcription...")
                    last_partial = ""

                    while True:
                        data = wf.readframes(1024)
                        if len(data) == 0:
                            break

                        rec.AcceptWaveform(data)

                        # Получаем промежуточный результат
                        partial_result = rec.PartialResult()
                        partial_dict = json.loads(partial_result)
                        partial_text = partial_dict.get('partial', '')

                        # Обновляем GUI только если текст изменился
                        if partial_text != last_partial:
                            update_callback(partial_text)
                            last_partial = partial_text

                    # Финальный результат
                    result = rec.FinalResult()
                    result_dict = json.loads(result)
                    final_text = result_dict.get('text', '')

                    final_callback(final_text)

            except Exception as e:
                logging.error(f"Error in live transcription: {e}")
                final_callback("")  # Пустой результат в случае ошибки

        # Запускаем в отдельном потоке
        thread = threading.Thread(target=process_audio, daemon=True)
        thread.start()
        return thread


# Глобальный экземпляр для кэширования модели
_global_recognizer = None


def get_global_recognizer(model_path: str, preload: bool = True) -> SpeechRecognizer:
    """
    Получить глобальный экземпляр распознавателя (синглтон)

    Args:
        model_path: Путь к модели Vosk
        preload: Загружать ли модель сразу

    Returns:
        SpeechRecognizer: Глобальный экземпляр распознавателя
    """
    global _global_recognizer

    if _global_recognizer is None or _global_recognizer.model_path != model_path:
        _global_recognizer = SpeechRecognizer(model_path, preload_model=preload)

    return _global_recognizer


def preload_vosk_model(model_path: str, show_progress: bool = True) -> None:
    """
    Предварительная загрузка модели Vosk в фоновом режиме

    Args:
        model_path: Путь к модели Vosk
        show_progress: Показывать ли прогресс загрузки
    """

    def background_load():
        try:
            # Получаем глобальный экземпляр и загружаем модель
            recognizer = get_global_recognizer(model_path, preload=False)
            recognizer.preload_model(show_progress)
        except Exception as e:
            logging.error(f"Background model preloading failed: {e}")

    thread = threading.Thread(target=background_load, daemon=True)
    thread.start()
