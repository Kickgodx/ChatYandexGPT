#!/usr/bin/env python3
"""
Пример базового использования библиотеки YandexGPTBot
"""

import os
import sys

# Добавляем корневую папку проекта в sys.path для корректного импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yandexgptbot_lib import ChatYandexGPTBot


def main():
    """Основная функция примера"""
    print("🚀 Пример базового использования YandexGPTBot")
    print("=" * 50)

    # Получение credentials из переменных окружения
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    iam_token = os.getenv("YANDEX_IAM_TOKEN")
    api_key = os.getenv("YANDEX_API_KEY")

    if not folder_id:
        print("❌ Необходимо указать YANDEX_FOLDER_ID в переменных окружения")
        print("   Создайте файл .env на основе env.example и заполните credentials")
        return

    if not iam_token and not api_key:
        print("❌ Необходимо указать YANDEX_IAM_TOKEN или YANDEX_API_KEY в переменных окружения")
        print("   Создайте файл .env на основе env.example и заполните credentials")
        return

    try:
        # Инициализация бота
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

        # Получение информации о текущих настройках
        current_prompt = bot.get_current_prompt_info()
        current_model = bot.get_current_model_info()

        print(f"🤖 Текущий помощник: {current_prompt['name']}")
        print(f"🚀 Текущая модель: {current_model['name']}")
        print()

        # Примеры вопросов для программиста
        questions = [
            "Как написать функцию для сортировки списка в Python?",
            "Объясни, что такое декораторы в Python",
            "Как создать простой веб-сервер на Flask?",
            "В чем разница между list и tuple в Python?"
        ]

        for i, question in enumerate(questions, 1):
            print(f"❓ Вопрос {i}: {question}")
            print("🤖 Ответ:")

            try:
                response = bot.get_response(question)
                print(response)
            except Exception as e:
                print(f"❌ Ошибка: {e}")

            print("-" * 50)

        # Получение резюме диалога
        summary = bot.get_conversation_summary()
        print(f"📊 {summary}")

        # Демонстрация смены промпта
        print("\n🔄 Смена промпта на 'teacher'...")
        bot.set_prompt("teacher")

        teacher_question = "Объясни, что такое фотосинтез простыми словами"
        print(f"❓ Вопрос: {teacher_question}")
        print("🤖 Ответ:")

        try:
            response = bot.get_response(teacher_question)
            print(response)
        except Exception as e:
            print(f"❌ Ошибка: {e}")

        print("\n✅ Пример завершен успешно!")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
