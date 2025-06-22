#!/usr/bin/env python3
"""
Пример управления промптами в библиотеке YandexGPTBot
"""

import os
import sys

# Добавляем корневую папку проекта в sys.path для корректного импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yandexgptbot_lib import ChatYandexGPTBot, PromptCollection


def main():
    """Основная функция примера"""
    print("🤖 Пример управления промптами YandexGPTBot")
    print("=" * 50)

    # Получение credentials из переменных окружения
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    iam_token = os.getenv("YANDEX_IAM_TOKEN")
    api_key = os.getenv("YANDEX_API_KEY")

    try:
        # Инициализация бота (если есть credentials)
        bot = None
        if folder_id and (iam_token or api_key):
            if iam_token:
                bot = ChatYandexGPTBot(
                    folder_id=folder_id,
                    iam_token=iam_token,
                    prompt_type="programmer",
                    model_name="yandexgpt-lite"
                )
            else:
                bot = ChatYandexGPTBot(
                    folder_id=folder_id,
                    api_key=api_key,
                    prompt_type="programmer",
                    model_name="yandexgpt-lite"
                )

        # 1. Демонстрация доступных промптов
        print("1️⃣ Доступные промпты:")
        available_prompts = PromptCollection.get_available_prompts()
        
        for key, prompt_info in available_prompts.items():
            print(f"   {key}: {prompt_info['name']}")
            print(f"      {prompt_info['description']}")
        print()

        # 2. Демонстрация получения промпта
        print("2️⃣ Демонстрация получения промпта:")
        for prompt_type in ["programmer", "teacher", "writer"]:
            prompt_info = PromptCollection.get_prompt(prompt_type)
            print(f"   {prompt_type}: {prompt_info['name']}")
            print(f"      Описание: {prompt_info['description']}")
            print(f"      Содержимое: {prompt_info['content'][:100]}...")
            print()
        print()

        # 3. Демонстрация работы с ботом (если доступен)
        if bot:
            print("3️⃣ Демонстрация работы с ботом:")
            
            # Показываем текущие настройки
            current_prompt = bot.get_current_prompt_info()
            current_model = bot.get_current_model_info()
            
            print(f"   Текущий помощник: {current_prompt['name']}")
            print(f"   Текущая модель: {current_model['name']}")
            print()

            # Тестируем разные промпты
            test_prompts = [
                ("programmer", "Как написать функцию для сортировки списка?"),
                ("teacher", "Объясни, что такое фотосинтез"),
                ("writer", "Помоги написать вступление к эссе"),
                ("analyst", "Какие методы анализа данных ты знаешь?"),
                ("creative", "Придумай идею для мобильного приложения"),
                ("general", "Расскажи интересный факт о космосе")
            ]

            for prompt_type, question in test_prompts:
                print(f"   🔄 Смена на промпт: {prompt_type}")
                bot.set_prompt(prompt_type)
                
                current_prompt = bot.get_current_prompt_info()
                print(f"   🤖 Помощник: {current_prompt['name']}")
                print(f"   ❓ Вопрос: {question}")
                
                try:
                    response = bot.get_response(question)
                    print(f"   💬 Ответ: {response[:200]}...")
                except Exception as e:
                    print(f"   ❌ Ошибка: {e}")
                
                print("-" * 50)

            # 4. Демонстрация управления диалогом
            print("\n4️⃣ Управление диалогом:")
            
            # Добавление контекста
            bot.add_context("Пользователь работает над проектом на Python")
            print("   📝 Контекст добавлен")
            
            # Получение резюме
            summary = bot.get_conversation_summary()
            print(f"   📊 Резюме: {summary}")
            
            # Сброс диалога
            bot.reset_conversation()
            print("   🔄 Диалог сброшен")
            
            summary_after_reset = bot.get_conversation_summary()
            print(f"   📊 Резюме после сброса: {summary_after_reset}")

        else:
            print("3️⃣ Демонстрация работы с ботом:")
            print("   ⚠️ Credentials не указаны, пропускаем тесты с API")
            print("   Создайте файл .env на основе env.example и заполните credentials:")
            print("   - YANDEX_FOLDER_ID")
            print("   - YANDEX_IAM_TOKEN или YANDEX_API_KEY")

        # 5. Демонстрация получения содержимого промпта
        print("\n5️⃣ Содержимое промптов:")
        for prompt_type in ["programmer", "teacher"]:
            content = PromptCollection.get_prompt_content(prompt_type)
            print(f"   {prompt_type}:")
            print(f"      {content[:200]}...")
            print()

        print("✅ Пример управления промптами завершен!")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main() 