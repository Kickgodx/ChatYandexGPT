#!/usr/bin/env python3
"""
Скрипт для сборки пакета YandexGPTBot
"""

import os
import shutil
import subprocess
import sys


def clean_previous_builds():
    """Очистка предыдущих сборок (не затрагивает папку build)"""
    print("🧹 Очистка предыдущих сборок...")

    # Очищаем только файлы сборки, не папку build
    dirs_to_clean = ['dist', 'build', 'yandexgptbot_lib.egg-info']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  Удалена папка: {dir_name}")


def build_package():
    """Сборка пакета"""
    print("🔨 Сборка пакета...")

    try:
        # Сборка дистрибутива
        subprocess.run([sys.executable, 'setup.py', 'sdist', 'bdist_wheel'], check=True)
        print("  ✅ Сборка завершена успешно")

        # Показать созданные файлы
        if os.path.exists('dist'):
            print("  📦 Созданные файлы:")
            for file in os.listdir('dist'):
                print(f"    - {file}")

    except subprocess.CalledProcessError as e:
        print(f"  ❌ Ошибка сборки: {e}")
        return False

    return True


def install_package():
    """Установка пакета в режиме разработки"""
    print("📦 Установка пакета в режиме разработки...")

    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=True)
        print("  ✅ Установка завершена успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Ошибка установки: {e}")
        return False


def test_import():
    """Тест импорта пакета"""
    print("🧪 Тест импорта пакета...")

    try:
        import yandexgptbot_lib
        print(f"  ✅ Пакет импортирован успешно (версия: {yandexgptbot_lib.__version__})")

        # Тест основных компонентов
        from yandexgptbot_lib import ChatYandexGPTBot, PromptCollection
        print("  ✅ Основные компоненты доступны")

        # Тест CLI
        from yandexgptbot_lib.cli import main
        print("  ✅ CLI интерфейс доступен")

        return True
    except ImportError as e:
        print(f"  ❌ Ошибка импорта: {e}")
        return False


def main():
    """Основная функция"""
    print("🚀 Сборка пакета YandexGPTBot")
    print("=" * 50)

    # Проверка наличия setup.py
    if not os.path.exists('setup.py'):
        print("❌ Файл setup.py не найден")
        return

    # Очистка предыдущих сборок
    clean_previous_builds()

    # Сборка
    if not build_package():
        print("❌ Сборка не удалась")
        return

    # Установка
    if not install_package():
        print("❌ Установка не удалась")
        return

    # Тест
    if not test_import():
        print("❌ Тест импорта не прошел")
        return

    print("\n🎉 Пакет успешно собран и установлен!")
    print("\n📋 Следующие шаги:")
    print("  1. Обновите credentials в ваших примерах")
    print("  2. Запустите примеры: python examples/basic_usage.py")
    print("  3. Используйте CLI: yandexgptbot --help")
    print("  4. Для публикации: python -m twine upload dist/*")


if __name__ == "__main__":
    main()
