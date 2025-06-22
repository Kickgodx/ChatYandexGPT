"""Улучшенный класс для работы с YandexGPT через LangChain"""
import logging

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.yandex import ChatYandexGPT

from src.prompt_collection import PromptCollection


class ChatYandexGPTBot:
    """Улучшенный класс для работы с YandexGPT"""

    def __init__(self, folder_id, iam_token=None, api_key=None, prompt_type="programmer"):
        """
        Инициализация бота
        
        Args:
            folder_id (str): ID папки в Yandex Cloud
            iam_token (str, optional): IAM токен для авторизации
            api_key (str, optional): API ключ для авторизации
            prompt_type (str): Тип промпта из PromptCollection
        """
        # Авторизация в сервисе
        if api_key:
            self.chat = ChatYandexGPT(api_key=api_key, folder_id=folder_id)
        elif iam_token:
            self.chat = ChatYandexGPT(iam_token=iam_token, folder_id=folder_id)
        else:
            raise ValueError("Необходимо указать либо iam_token, либо api_key")

        # Инициализация истории сообщений
        self.messages = []
        
        # Установка промпта
        self.prompt_type = prompt_type
        self.set_prompt(prompt_type)

        logging.info(f"ChatYandexGPTBot initialized with prompt type: {prompt_type}")

    def set_prompt(self, prompt_type):
        """Установить новый промпт"""
        self.prompt_type = prompt_type
        prompt_content = PromptCollection.get_prompt_content(prompt_type)

        # Обновляем системное сообщение
        system_message = SystemMessage(content=prompt_content)

        # Если есть история сообщений, заменяем первое (системное) сообщение
        if self.messages:
            self.messages[0] = system_message
        else:
            self.messages = [system_message]

        logging.info(f"Prompt changed to: {prompt_type}")

    def reset_conversation(self):
        """Сбросить историю диалога"""
        prompt_content = PromptCollection.get_prompt_content(self.prompt_type)
        self.messages = [SystemMessage(content=prompt_content)]
        logging.info("Conversation history reset")

    def get_response(self, user_input):
        """
        Получить ответ от AI
        
        Args:
            user_input (str): Ввод пользователя
            
        Returns:
            str: Ответ от AI
        """
        try:
            # Добавляем сообщение пользователя
            self.messages.append(HumanMessage(content=user_input))

            # Получаем ответ от модели
            response = self.chat.invoke(self.messages)

            # Добавляем ответ в историю
            self.messages.append(response)

            logging.info(f"Response generated for prompt type: {self.prompt_type}")
            return response.content

        except Exception as e:
            logging.error(f"Error getting response: {e}")
            return f"Произошла ошибка при получении ответа: {str(e)}"

    def get_conversation_history(self):
        """Получить историю диалога"""
        return self.messages

    def get_current_prompt_info(self):
        """Получить информацию о текущем промпте"""
        return PromptCollection.get_prompt(self.prompt_type)

    @staticmethod
    def get_available_prompts():
        """Получить список доступных промптов"""
        return PromptCollection.get_available_prompts()

    def add_context(self, context):
        """
        Добавить контекстную информацию в диалог
        
        Args:
            context (str): Контекстная информация
        """
        context_message = SystemMessage(content=f"Дополнительный контекст: {context}")
        self.messages.append(context_message)
        logging.info("Context added to conversation")

    def get_conversation_summary(self):
        """Получить краткое резюме диалога"""
        if len(self.messages) <= 1:  # Только системное сообщение
            return "Диалог еще не начался"

        user_messages = [msg.content for msg in self.messages if isinstance(msg, HumanMessage)]
        ai_messages = [msg.content for msg in self.messages if
                       hasattr(msg, 'content') and not isinstance(msg, HumanMessage) and not isinstance(msg, SystemMessage)]

        summary = f"Диалог содержит {len(user_messages)} сообщений пользователя и {len(ai_messages)} ответов AI"
        return summary
