"""
YandexGPTBot Library - Библиотека для работы с YandexGPT и распознаванием речи

Версия: 1.1.2
Дата: 2025-06-23

Исправления в версии 1.1.2:
- Исправлена критическая ошибка с синхронизацией загрузки модели Vosk
- Улучшена работа глобального синглтона для предотвращения дублирования загрузки
- Исправлен мониторинг загрузки модели в UI
- Добавлено подавление предупреждения Pydantic
- Улучшена потокобезопасность при конкурентном доступе к модели

Основные возможности:
- Работа с YandexGPT API
- Распознавание речи с помощью Vosk
- Оптимизированная загрузка модели с кэшированием
- Live-транскрипция в реальном времени
- Управление диалогами и промптами
- CLI интерфейс
- Конфигурация и настройки

Автор: AI Assistant
Лицензия: MIT
"""

__version__ = "1.1.2"
__author__ = "AI Assistant"
__license__ = "MIT"

from .audio import AudioRecorder, AudioProcessor, SpeechRecognizer, get_global_recognizer, preload_vosk_model
# Основные импорты
from .bot import ChatYandexGPTBot
from .cli import main as cli_main
from .config import Config
from .conversation import ConversationManager
from .prompts import PromptCollection
from .settings import Settings

# Экспорт основных классов
__all__ = [
    'ChatYandexGPTBot',
    'AudioRecorder',
    'AudioProcessor',
    'SpeechRecognizer',
    'get_global_recognizer',
    'preload_vosk_model',
    'PromptCollection',
    'ConversationManager',
    'Config',
    'Settings',
    'cli_main'
]
