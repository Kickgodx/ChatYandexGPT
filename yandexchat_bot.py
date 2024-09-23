"""Пример работы с чатом через gigachain"""
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.yandex import ChatYandexGPT

class ChatYandexGPTBot:
    def __init__(self, iam_token, folder_id):
        # Авторизация в сервисе GigaChat
        self.chat = ChatYandexGPT(iam_token=iam_token, folder_id=folder_id)
        self.messages = [
            SystemMessage(
                content="Ты бот-программист, который помогает пользователю решить его задачи, а так же хорошо разбираешься во всех аспектах программирования и тестирования. Так же ты можешь помочь с теоретическими вопросами"
            )
        ]

    def get_response(self, user_input):
        # Ввод пользователя
        self.messages.append(HumanMessage(content=user_input))
        res = self.chat.invoke(self.messages)
        self.messages.append(res)
        # Ответ модели
        return res.content