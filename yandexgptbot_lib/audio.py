"""
Модуль для работы с аудио: запись, обработка и распознавание речи
"""

import json
import logging
import os
import threading
import time
import wave
from typing import List, Optional, Tuple, Any, Callable

import numpy as np
import pyaudio
from vosk import Model, KaldiRecognizer


class AudioRecorder:
    """Класс для записи аудио с микрофона и компьютера"""

    def __init__(self, config, settings):
        self.config = config
        self.settings = settings
        self.is_recording = False
        self.stream = None
        self.frames: List[bytes] = []
        self.lock = threading.Lock()
        self.audio = None
        self.device_type = None

    def start_recording(self, device_type: str = 'mic') -> None:
        """Начало записи аудио"""
        with self.lock:
            if not self.is_recording:
                self.device_type = device_type
                self.audio = pyaudio.PyAudio()

                if device_type == 'mic':
                    self.stream = self.audio.open(
                        format=self.config.FORMAT,
                        channels=self.config.CHANNELS,
                        rate=self.config.RATE,
                        input=True,
                        frames_per_buffer=self.config.CHUNK,
                        stream_callback=self.callback
                    )
                elif device_type == 'computer':
                    input_device_index = self.find_stereo_mix()
                    if input_device_index is None:
                        raise ValueError("Stereo Mix device not found")

                    self.stream = self.audio.open(
                        format=self.config.FORMAT,
                        channels=self.config.CHANNELS,
                        rate=self.config.RATE,
                        input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=self.config.CHUNK,
                        stream_callback=self.callback
                    )

                self.frames = []
                self.stream.start_stream()
                self.is_recording = True
                logging.info(f"Recording started from {device_type}")

    def stop_recording(self) -> Optional[List[bytes]]:
        """Остановка записи аудио"""
        with self.lock:
            if self.is_recording:
                self.stream.stop_stream()
                self.stream.close()
                self.audio.terminate()
                self.is_recording = False
                logging.info("Recording stopped")
                return self.frames
        return None

    def callback(self, in_data: bytes, frame_count: int, time_info: Any, status: Any) -> Tuple[bytes, int]:
        """Callback для записи аудио"""
        self.frames.append(in_data)
        return in_data, pyaudio.paContinue

    def find_stereo_mix(self) -> Optional[int]:
        """Поиск устройства Stereo Mix"""
        for i in range(self.audio.get_device_count()):
            dev = self.audio.get_device_info_by_index(i)
            try:
                dev_name = dev['name'].encode('windows-1251').decode('utf-8')
            except UnicodeDecodeError:
                dev_name = dev['name']
            if dev_name.lower().startswith("стерео") or dev_name.startswith("Стерео"):
                return i
        return None

    def check_audio_quality(self, audio_data: bytes) -> Tuple[bool, str]:
        """Проверка качества аудио"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_array ** 2))

        # Порог для определения слишком тихого аудио
        threshold = self.settings.settings.get("audio_quality_threshold", 15)

        # Возвращаем предупреждение только если аудио действительно слишком тихое
        if rms < threshold:
            return False, f"Аудио слишком тихое (RMS: {rms:.2f}, порог: {threshold})"

        # Если аудио нормальное, не показываем никаких сообщений
        return True, ""


class AudioProcessor:
    """Класс для обработки аудио"""

    def __init__(self, config):
        self.config = config

    @staticmethod
    def normalize_audio(file_path: str) -> None:
        """Нормализация уровня звука в аудиофайле"""
        with wave.open(file_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)

        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array * (32767 / max_val)

        normalized_audio_data = audio_array.astype(np.int16).tobytes()

        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(normalized_audio_data)

    def save_audio(self, frames: List[bytes], file_path: str) -> None:
        """Сохранение аудио в файл"""
        audio = pyaudio.PyAudio()
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(self.config.CHANNELS)
            wf.setsampwidth(audio.get_sample_size(self.config.FORMAT))
            wf.setframerate(self.config.RATE)
            wf.writeframes(b''.join(frames))
        audio.terminate()


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
                          Принимает параметры: (text, is_final)
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

        Returns:
            Thread: Поток выполнения транскрипции
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
_global_recognizer_lock = threading.Lock()


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

    with _global_recognizer_lock:
        # Проверяем, есть ли уже загруженный распознаватель с той же моделью
        if _global_recognizer is not None and _global_recognizer.model_path == model_path:
            # Если модель уже загружена, возвращаем существующий экземпляр
            if _global_recognizer.is_model_loaded():
                return _global_recognizer
            # Если модель еще загружается, ждем завершения
            elif not preload:
                return _global_recognizer

        # Создаем новый экземпляр только если нужно
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
