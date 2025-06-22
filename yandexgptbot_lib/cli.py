"""
CLI интерфейс для YandexGPTBot
"""

import argparse
import logging
import os
import sys

from .bot import ChatYandexGPTBot


def setup_logging(verbose: bool = False) -> None:
    """Настройка логирования"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s][%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler('yandexgptbot.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def ensure_output_directory() -> None:
    """Создание папки outputs если её нет"""
    os.makedirs("outputs", exist_ok=True)


def chat_mode(bot: ChatYandexGPTBot) -> None:
    """Интерактивный режим чата"""
    print("🤖 YandexGPTBot CLI - Интерактивный режим")
    print("Введите 'quit' или 'exit' для выхода")
    print("Введите 'help' для справки")
    print("-" * 50)

    while True:
        try:
            user_input = input("👤 Вы: ").strip()

            if user_input.lower() in ['quit', 'exit', 'выход']:
                print("👋 До свидания!")
                break
            elif user_input.lower() in ['help', 'помощь']:
                print_help()
                continue
            elif user_input.lower() in ['status', 'статус']:
                print_status(bot)
                continue
            elif user_input.lower() in ['reset', 'сброс']:
                bot.reset_conversation()
                print("🔄 История диалога сброшена")
                continue
            elif not user_input:
                continue

            print("🤖 AI: ", end="", flush=True)
            response = bot.get_response(user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


def print_help() -> None:
    """Вывод справки"""
    print("\n📖 Справка по командам:")
    print("  help/помощь    - Показать эту справку")
    print("  status/статус  - Показать статус бота")
    print("  reset/сброс    - Сбросить историю диалога")
    print("  quit/exit/выход - Выйти из программы")
    print()


def print_status(bot: ChatYandexGPTBot) -> None:
    """Вывод статуса бота"""
    current_prompt = bot.get_current_prompt_info()
    current_model = bot.get_current_model_info()
    summary = bot.get_conversation_summary()

    print("\n📊 Статус бота:")
    print(f"  Модель: {current_model['name']}")
    print(f"  Помощник: {current_prompt['name']}")
    print(f"  {summary}")
    print()


def main() -> None:
    """Основная функция CLI"""
    parser = argparse.ArgumentParser(
        description="YandexGPTBot - CLI интерфейс для работы с YandexGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  yandexgptbot --iam-token YOUR_TOKEN --folder-id YOUR_FOLDER_ID
  yandexgptbot --api-key YOUR_API_KEY --folder-id YOUR_FOLDER_ID --model yandexgpt
  yandexgptbot --prompt teacher --verbose
        """
    )

    # Параметры авторизации
    auth_group = parser.add_mutually_exclusive_group(required=True)
    auth_group.add_argument('--iam-token', help='IAM токен для авторизации')
    auth_group.add_argument('--api-key', help='API ключ для авторизации')

    # Обязательные параметры
    parser.add_argument('--folder-id', required=True, help='ID папки в Yandex Cloud')

    # Опциональные параметры
    parser.add_argument('--model', default='yandexgpt-lite',
                        choices=['yandexgpt-lite', 'yandexgpt', 'yandexgpt-plus'],
                        help='Модель YandexGPT (по умолчанию: yandexgpt-lite)')
    parser.add_argument('--prompt', default='programmer',
                        choices=['programmer', 'teacher', 'writer', 'analyst', 'creative', 'general'],
                        help='Тип помощника (по умолчанию: programmer)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')
    parser.add_argument('--list-models', action='store_true', help='Показать доступные модели')
    parser.add_argument('--list-prompts', action='store_true', help='Показать доступные промпты')

    args = parser.parse_args()

    # Настройка логирования
    setup_logging(args.verbose)

    # Показать доступные модели
    if args.list_models:
        print("🚀 Доступные модели YandexGPT:")
        models = ChatYandexGPTBot.get_available_models()
        for key, model in models.items():
            print(f"  {key}: {model['name']} - {model['description']}")
        return

    # Показать доступные промпты
    if args.list_prompts:
        print("🤖 Доступные типы помощников:")
        prompts = ChatYandexGPTBot.get_available_prompts()
        for key, prompt in prompts.items():
            print(f"  {key}: {prompt['name']} - {prompt['description']}")
        return

    try:
        # Создание папки outputs
        ensure_output_directory()

        # Инициализация бота
        if args.iam_token:
            bot = ChatYandexGPTBot(
                folder_id=args.folder_id,
                iam_token=args.iam_token,
                prompt_type=args.prompt,
                model_name=args.model
            )
        else:
            bot = ChatYandexGPTBot(
                folder_id=args.folder_id,
                api_key=args.api_key,
                prompt_type=args.prompt,
                model_name=args.model
            )

        # Запуск интерактивного режима
        chat_mode(bot)

    except Exception as e:
        logging.error(f"Ошибка: {e}")
        print(f"❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
