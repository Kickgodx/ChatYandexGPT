import pyaudio


class Config:
    """Конфигурация приложения"""
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    WAVE_OUTPUT_FILENAME = "./outputs/question.wav"
    RESPONSE_OUTPUT_FILENAME = "./outputs/responses.txt"
