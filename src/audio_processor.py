import wave

import numpy as np
import pyaudio


class AudioProcessor:
    """Класс для обработки аудио"""

    def __init__(self, config):
        self.config = config

    @staticmethod
    def normalize_audio(file_path):
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

    def save_audio(self, frames, file_path):
        """Сохранение аудио в файл"""
        audio = pyaudio.PyAudio()
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(self.config.CHANNELS)
            wf.setsampwidth(audio.get_sample_size(self.config.FORMAT))
            wf.setframerate(self.config.RATE)
            wf.writeframes(b''.join(frames))
        audio.terminate()
