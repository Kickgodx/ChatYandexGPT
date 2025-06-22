"""
Утилита для загрузки переменных окружения из .env файла
"""

import os
from typing import Optional


def load_env_variables() -> None:
    """Загрузка переменных окружения из .env файла"""
    env_file = ".env"

    if not os.path.exists(env_file):
        print(f"⚠️ Файл {env_file} не найден. Создайте его на основе env.example")
        return

    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Убираем кавычки если есть
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    # Устанавливаем переменную окружения только если её нет
                    if key and not os.getenv(key):
                        os.environ[key] = value

    except Exception as e:
        print(f"❌ Ошибка загрузки {env_file}: {e}")


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Получить переменную окружения"""
    return os.getenv(key, default)


def get_required_env_var(key: str) -> str:
    """Получить обязательную переменную окружения"""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Переменная окружения {key} не установлена. Проверьте файл .env")
    return value


# Автоматическая загрузка при импорте модуля
load_env_variables()
