"""Пример работы с чатом через gigachain"""
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.yandex import ChatYandexGPT

class ChatYandexGPTBot:
    def __init__(self, folder_id, iam_token=None, api_key=None):
        # Авторизация в сервисе ChatYandexGPTBot
        if api_key:
            self.chat = ChatYandexGPT(api_key=api_key, folder_id=folder_id)
        if iam_token:
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