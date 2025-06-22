#!/usr/bin/env python3
"""
Скрипт для инициализации проекта ChatYandexGPT
"""

import os
import shutil
import sys


def check_python_version():
    """Проверка версии Python"""
    if sys.version_info < (3, 7):
        print("❌ Требуется Python 3.7 или выше")
        print(f"   Текущая версия: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} - OK")
    return True


def create_env_file():
    """Создание файла .env из env.example"""
    if os.path.exists(".env"):
        print("✅ Файл .env уже существует")
        return True
    
    if not os.path.exists("env.example"):
        print("❌ Файл env.example не найден")
        return False
    
    try:
        shutil.copy("env.example", ".env")
        print("✅ Файл .env создан из env.example")
        print("   ⚠️ Заполните credentials в файле .env")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания .env: {e}")
        return False


def create_outputs_directory():
    """Создание папки outputs"""
    try:
        os.makedirs("outputs", exist_ok=True)
        print("✅ Папка outputs создана")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания папки outputs: {e}")
        return False


def check_vosk_model():
    """Проверка наличия модели Vosk"""
    vosk_path = os.getenv("VOSK_MODEL_PATH", "./vosk-model-ru-0.42")
    
    if os.path.exists(vosk_path):
        print(f"✅ Модель Vosk найдена: {vosk_path}")
        return True
    else:
        print(f"⚠️ Модель Vosk не найдена: {vosk_path}")
        print("   Скачайте модель с https://alphacephei.com/vosk/models")
        print("   И распакуйте в папку vosk-model-ru-0.42")
        return False


def check_requirements():
    """Проверка файла requirements.txt"""
    if os.path.exists("requirements.txt"):
        print("✅ Файл requirements.txt найден")
        return True
    else:
        print("❌ Файл requirements.txt не найден")
        return False


def show_next_steps():
    """Показать следующие шаги"""
    print("\n" + "="*60)
    print("🎉 Инициализация проекта завершена!")
    print("="*60)
    print("\n📋 Следующие шаги:")
    print("1. Активируйте виртуальное окружение:")
    print("   python -m venv venv")
    print("   # Windows:")
    print("   .\\venv\\Scripts\\activate")
    print("   # Linux/Mac:")
    print("   source venv/bin/activate")
    print()
    print("2. Установите зависимости:")
    print("   pip install -r requirements.txt")
    print()
    print("3. Заполните credentials в файле .env:")
    print("   - YANDEX_FOLDER_ID (обязательно)")
    print("   - YANDEX_IAM_TOKEN или YANDEX_API_KEY")
    print()
    print("4. Скачайте модель Vosk (если не скачана)")
    print()
    print("5. Запустите приложение:")
    print("   python interface_AI.py")
    print()
    print("📚 Дополнительная информация:")
    print("   - README.md - полная документация")
    print("   - examples/ - примеры использования")
    print("   - yandexgptbot_lib/ - библиотечная версия")


def main():
    """Основная функция"""
    print("🚀 Инициализация проекта ChatYandexGPT")
    print("="*60)
    
    success = True
    
    # Проверки
    if not check_python_version():
        success = False
    
    if not check_requirements():
        success = False
    
    if not create_env_file():
        success = False
    
    if not create_outputs_directory():
        success = False
    
    check_vosk_model()  # Предупреждение, не критично
    
    if success:
        show_next_steps()
    else:
        print("\n❌ Инициализация не завершена. Исправьте ошибки и попробуйте снова.")


if __name__ == "__main__":
    main() 