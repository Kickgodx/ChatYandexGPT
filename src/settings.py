import json


class Settings:
    """Управление настройками приложения"""

    def __init__(self, filename="settings.json"):
        self.filename = filename
        self.settings = self.load_settings()

    def load_settings(self):
        """Загрузка настроек из файла"""
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_settings()

    @staticmethod
    def get_default_settings():
        """Получение настроек по умолчанию"""
        return {
            "audio_quality_threshold": 100,
            "max_conversation_history": 50,
            "auto_save_interval": 10,
            "hotkeys_enabled": True
        }

    def save_settings(self):
        """Сохранение настроек в файл"""
        with open(self.filename, 'w') as f:
            json.dump(self.settings, f)
