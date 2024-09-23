"""Пример работы с чатом через gigachain"""
from langchain.schema import HumanMessage, SystemMessage
# from langchain.chat_models.yandex import ChatYandexGPT
from langchain_community.chat_models.yandex import ChatYandexGPT
from yandex_creds import iam_token, folder_id

chat = ChatYandexGPT(
    iam_token=iam_token,
    folder_id=folder_id)

messages = [
    SystemMessage(
        content="Ты бот-программист, который помогает пользователю решить его задачи, а так же хорошо разбираешься во всех аспектах программирования и тестирования."
    )
]

while(True):
    # Ввод пользователя
    user_input = input("User: ")
    messages.append(HumanMessage(content=user_input))
    res = chat(messages)
    messages.append(res)
    # Ответ модели
    print("Bot: ", res.content)