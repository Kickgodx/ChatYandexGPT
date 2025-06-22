"""
Управление настройками приложения
"""

import json
from typing import Dict, Any


class Settings:
    """Управление настройками приложения"""

    def __init__(self, filename: str = "settings.json"):
        self.filename = filename
        self.settings = self.load_settings()

    def load_settings(self) -> Dict[str, Any]:
        """Загрузка настроек из файла"""
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_settings()

    @staticmethod
    def get_default_settings() -> Dict[str, Any]:
        """Получение настроек по умолчанию"""
        return {
            "audio_quality_threshold": 15,
            "max_conversation_history": 50,
            "auto_save_interval": 10,
            "hotkeys_enabled": True
        }

    def save_settings(self) -> None:
        """Сохранение настроек в файл"""
        with open(self.filename, 'w') as f:
            json.dump(self.settings, f)

    def get(self, key: str, default: Any = None) -> Any:
        """Получить значение настройки"""
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Установить значение настройки"""
        self.settings[key] = value
        self.save_settings()
