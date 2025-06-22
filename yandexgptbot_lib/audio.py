"""
Модуль для работы с аудио: запись, обработка и распознавание речи
"""

import json
import logging
import threading
import wave
from typing import List, Optional, Tuple, Any

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
    """Класс для распознавания речи"""

    def __init__(self, model_path: str):
        self.model = Model(model_path)

    def transcribe_audio(self, file_path: str) -> str:
        """Транскрипция аудио в текст"""
        with wave.open(file_path, "rb") as wf:
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)

            logging.info("Starting transcription...")
            result_text = ""

            while True:
                data = wf.readframes(1024)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)

            result = rec.FinalResult()
            result_dict = json.loads(result)
            result_text = result_dict.get('text', '')
        return result_text
