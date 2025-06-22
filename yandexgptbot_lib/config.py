"""
Конфигурация приложения
"""

import os

import pyaudio


class Config:
    """Конфигурация приложения"""
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    WAVE_OUTPUT_FILENAME = "./outputs/question.wav"
    RESPONSE_OUTPUT_FILENAME = "./outputs/responses.txt"

    @classmethod
    def get_vosk_model_path(cls) -> str:
        """Получить путь к модели Vosk из переменных окружения"""
        return os.getenv("VOSK_MODEL_PATH", "./vosk-model-ru-0.42")

    @classmethod
    def get_yandex_credentials(cls) -> dict:
        """Получить credentials Yandex из переменных окружения"""
        return {
            "folder_id": os.getenv("YANDEX_FOLDER_ID"),
            "iam_token": os.getenv("YANDEX_IAM_TOKEN"),
            "api_key": os.getenv("YANDEX_API_KEY"),
            "model_name": os.getenv("YANDEX_MODEL_NAME", "yandexgpt-lite"),
            "prompt_type": os.getenv("YANDEX_PROMPT_TYPE", "programmer")
        }
