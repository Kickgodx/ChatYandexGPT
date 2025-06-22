"""
YandexGPTBot - Библиотека для работы с YandexGPT через голосовой и текстовый интерфейс

Основные компоненты:
- ChatYandexGPTBot: Основной класс для работы с YandexGPT
- AudioRecorder: Запись аудио с микрофона и компьютера
- SpeechRecognizer: Распознавание речи с помощью Vosk
- PromptCollection: Коллекция предустановленных промптов
- ConversationManager: Управление историей диалогов
"""

from .audio import AudioRecorder, AudioProcessor, SpeechRecognizer
from .bot import ChatYandexGPTBot
from .config import Config
from .conversation import ConversationManager
from .prompts import PromptCollection
from .settings import Settings

__version__ = "1.0.0"
__author__ = "Den"
__email__ = "wheelman4000@gmail.com"

__all__ = [
    "ChatYandexGPTBot",
    "AudioRecorder",
    "AudioProcessor",
    "SpeechRecognizer",
    "PromptCollection",
    "ConversationManager",
    "Config",
    "Settings"
]
