#!/usr/bin/env python3
"""
Тестовый файл для демонстрации работы с промптами
"""

import os
import sys

# Добавляем корневую папку проекта в sys.path для корректного импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yandexchat_bot import ChatYandexGPTBot
from src.prompt_collection import PromptCollection
from yandex_creds import iam_token, folder_id, api_key


def test_prompt_collection():
    """Тестирование коллекции промптов"""
    print("=== Тестирование коллекции промптов ===\n")

    # Получение списка доступных промптов
    available_prompts = PromptCollection.get_available_prompts()

    print("Доступные промпты:")
    for key, value in available_prompts.items():
        print(f"  {key}: {value['name']} - {value['description']}")

    print("\n" + "=" * 50 + "\n")

    # Тестирование получения промпта
    for prompt_type in available_prompts.keys():
        prompt_info = PromptCollection.get_prompt(prompt_type)
        print(f"Промпт '{prompt_type}':")
        print(f"  Название: {prompt_info['name']}")
        print(f"  Описание: {prompt_info['description']}")
        print(f"  Содержимое: {prompt_info['content'][:100]}...")
        print()


def test_bot_with_different_prompts():
    """Тестирование бота с разными промптами"""
    print("=== Тестирование бота с разными промптами ===\n")

    # Инициализация бота
    if iam_token:
        bot = ChatYandexGPTBot(iam_token=iam_token, folder_id=folder_id)
    else:
        bot = ChatYandexGPTBot(api_key=api_key, folder_id=folder_id)

    # Тестовые вопросы для разных типов промптов
    test_questions = {
        "programmer": "Как написать функцию для сортировки массива в Python?",
        "teacher": "Объясни, что такое фотосинтез простыми словами",
        "writer": "Помоги написать вступление к эссе о важности чтения",
        "analyst": "Какие методы анализа данных ты знаешь?",
        "creative": "Придумай идею для мобильного приложения",
        "general": "Расскажи интересный факт о космосе"
    }

    # Тестирование каждого типа промпта
    for prompt_type, question in test_questions.items():
        print(f"--- Тестирование промпта: {prompt_type} ---")

        # Установка промпта
        bot.set_prompt(prompt_type)
        current_prompt = bot.get_current_prompt_info()
        print(f"Текущий помощник: {current_prompt['name']}")

        # Отправка вопроса
        print(f"Вопрос: {question}")
        try:
            response = bot.get_response(question)
            print(f"Ответ: {response[:200]}...")
        except Exception as e:
            print(f"Ошибка: {e}")

        print("\n" + "-" * 50 + "\n")


def test_bot_methods():
    """Тестирование методов бота"""
    print("=== Тестирование методов бота ===\n")

    # Инициализация бота
    if iam_token:
        bot = ChatYandexGPTBot(iam_token=iam_token, folder_id=folder_id)
    else:
        bot = ChatYandexGPTBot(api_key=api_key, folder_id=folder_id)

    # Тестирование получения информации о текущем промпте
    current_prompt = bot.get_current_prompt_info()
    print(f"Текущий промпт: {current_prompt['name']}")

    # Тестирование получения истории диалога
    history = bot.get_conversation_history()
    print(f"Количество сообщений в истории: {len(history)}")

    # Тестирование добавления контекста
    bot.add_context("Пользователь работает над проектом на Python")
    print("Контекст добавлен")

    # Тестирование получения резюме диалога
    summary = bot.get_conversation_summary()
    print(f"Резюме диалога: {summary}")

    # Тестирование сброса диалога
    bot.reset_conversation()
    print("Диалог сброшен")

    summary_after_reset = bot.get_conversation_summary()
    print(f"Резюме после сброса: {summary_after_reset}")


def main():
    """Основная функция тестирования"""
    print("🚀 Тестирование системы промптов ChatYandexGPT\n")

    try:
        # Тест 1: Коллекция промптов
        test_prompt_collection()

        # Тест 2: Методы бота
        test_bot_methods()

        # Тест 3: Бот с разными промптами (только если есть доступ к API)
        print("Хотите протестировать бота с разными промптами? (y/n): ", end="")
        choice = input().lower().strip()

        if choice == 'y':
            test_bot_with_different_prompts()
        else:
            print("Пропускаем тест с API")

    except Exception as e:
        print(f"Ошибка при тестировании: {e}")

    print("\n✅ Тестирование завершено!")


if __name__ == "__main__":
    main()
