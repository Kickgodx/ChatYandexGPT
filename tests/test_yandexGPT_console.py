"""Пример работы с чатом через gigachain"""
import logging

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.yandex import ChatYandexGPT

from src.env_loader import get_required_env_var, get_env_var, load_env_variables

load_env_variables()
# Загрузка credentials из переменных окружения
try:
    folder_id = get_required_env_var("YANDEX_FOLDER_ID")
    iam_token = get_env_var("YANDEX_IAM_TOKEN", None)
    api_key = get_env_var("YANDEX_API_KEY", None)
    vosk_model_path = get_env_var("VOSK_MODEL_PATH", "./vosk-model-ru-0.42")
except ValueError as e:
    logging.error(f"Ошибка загрузки credentials: {e}")
    raise e

chat = ChatYandexGPT(
    api_key=api_key,
    folder_id=folder_id)

messages = [
    SystemMessage(
        content="Ты бот-программист, который помогает пользователю решить его задачи, а так же хорошо разбираешься во всех аспектах программирования и тестирования."
    )
]

while True:
    # Ввод пользователя
    user_input = input("User: ")
    messages.append(HumanMessage(content=user_input))
    res = chat(messages)
    messages.append(res)
    # Ответ модели
    print("Bot: ", res.content)
