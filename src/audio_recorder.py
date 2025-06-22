import logging
import threading

import numpy as np
import pyaudio


class AudioRecorder:
    """Класс для записи аудио с микрофона и компьютера"""

    def __init__(self, config, settings):
        self.config = config
        self.settings = settings
        self.is_recording = False
        self.stream = None
        self.frames = []
        self.lock = threading.Lock()
        self.audio = None
        self.device_type = None

    def start_recording(self, device_type='mic'):
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

    def stop_recording(self):
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

    def callback(self, in_data, frame_count, time_info, status):
        """Callback для записи аудио"""
        self.frames.append(in_data)
        return in_data, pyaudio.paContinue

    def find_stereo_mix(self):
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

    def check_audio_quality(self, audio_data):
        """Проверка качества аудио"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_array ** 2))
        threshold = self.settings.settings.get("audio_quality_threshold", 100)
        if rms < threshold:
            return False, f"Аудио слишком тихое (RMS: {rms:.2f})"
        return True, f"Качество аудио OK (RMS: {rms:.2f})"
