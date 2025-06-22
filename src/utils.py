import logging
import os


def setup_logging():
    """Настройка логирования приложения"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler('app.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def ensure_output_directory():
    """Создание папки outputs если её нет"""
    os.makedirs("outputs", exist_ok=True)
