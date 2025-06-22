"""
Модуль для работы с YandexGPT через LangChain
"""

import logging
from typing import Optional, Dict, Any, List

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models.yandex import ChatYandexGPT

from .prompts import PromptCollection


class ChatYandexGPTBot:
    """Улучшенный класс для работы с YandexGPT"""

    # Доступные модели YandexGPT
    AVAILABLE_MODELS = {
        "yandexgpt-lite": {
            "name": "YandexGPT Lite",
            "description": "Быстрая и экономичная модель для простых задач",
            "max_tokens": 4000
        },
        "yandexgpt": {
            "name": "YandexGPT",
            "description": "Базовая модель для большинства задач",
            "max_tokens": 8000
        },
        "yandexgpt-plus": {
            "name": "YandexGPT Plus",
            "description": "Продвинутая модель с улучшенными возможностями",
            "max_tokens": 8000
        }
    }

    def __init__(self, folder_id: str, iam_token: Optional[str] = None,
                 api_key: Optional[str] = None, prompt_type: str = "programmer",
                 model_name: str = "yandexgpt-lite"):
        """
        Инициализация бота
        
        Args:
            folder_id: ID папки в Yandex Cloud
            iam_token: IAM токен для авторизации
            api_key: API ключ для авторизации
            prompt_type: Тип промпта из PromptCollection
            model_name: Название модели YandexGPT
        """
        # Проверка доступности модели
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Модель '{model_name}' не поддерживается. Доступные модели: {list(self.AVAILABLE_MODELS.keys())}")

        self.model_name = model_name
        self.model_info = self.AVAILABLE_MODELS[model_name]

        # Авторизация в сервисе
        if api_key:
            self.chat = ChatYandexGPT(api_key=api_key, folder_id=folder_id, model_name=model_name)
        elif iam_token:
            self.chat = ChatYandexGPT(iam_token=iam_token, folder_id=folder_id, model_name=model_name)
        else:
            raise ValueError("Необходимо указать либо iam_token, либо api_key")

        # Инициализация истории сообщений
        self.messages: List = []

        # Установка промпта
        self.prompt_type = prompt_type
        self.set_prompt(prompt_type)

        logging.info(f"ChatYandexGPTBot initialized with model: {model_name}, prompt type: {prompt_type}")

    def set_model(self, model_name: str) -> None:
        """Установить новую модель"""
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Модель '{model_name}' не поддерживается")

        self.model_name = model_name
        self.model_info = self.AVAILABLE_MODELS[model_name]

        # Пересоздаем чат с новой моделью
        if hasattr(self, 'chat'):
            # Сохраняем текущие credentials
            if hasattr(self.chat, 'api_key'):
                self.chat = ChatYandexGPT(api_key=self.chat.api_key, folder_id=self.chat.folder_id, model_name=model_name)
            elif hasattr(self.chat, 'iam_token'):
                self.chat = ChatYandexGPT(iam_token=self.chat.iam_token, folder_id=self.chat.folder_id, model_name=model_name)

        logging.info(f"Model changed to: {model_name}")

    def get_current_model_info(self) -> Dict[str, Any]:
        """Получить информацию о текущей модели"""
        return {
            'name': self.model_info['name'],
            'description': self.model_info['description'],
            'max_tokens': self.model_info['max_tokens'],
            'model_name': self.model_name
        }

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Получить список доступных моделей"""
        return cls.AVAILABLE_MODELS

    def set_prompt(self, prompt_type: str) -> None:
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

    def reset_conversation(self) -> None:
        """Сбросить историю диалога"""
        prompt_content = PromptCollection.get_prompt_content(self.prompt_type)
        self.messages = [SystemMessage(content=prompt_content)]
        logging.info("Conversation history reset")

    def get_response(self, user_input: str) -> str:
        """
        Получить ответ от AI
        
        Args:
            user_input: Ввод пользователя
            
        Returns:
            Ответ от AI
        """
        try:
            # Добавляем сообщение пользователя
            self.messages.append(HumanMessage(content=user_input))

            # Получаем ответ от модели
            response = self.chat.invoke(self.messages)

            # Добавляем ответ в историю
            self.messages.append(response)

            logging.info(f"Response generated by {self.model_name} for prompt type: {self.prompt_type}")
            return response.content

        except Exception as e:
            logging.error(f"Error getting response from {self.model_name}: {e}")
            return f"Произошла ошибка при получении ответа от {self.model_info['name']}: {str(e)}"

    def get_conversation_history(self) -> List:
        """Получить историю диалога"""
        return self.messages

    def get_current_prompt_info(self) -> Dict[str, str]:
        """Получить информацию о текущем промпте"""
        return PromptCollection.get_prompt(self.prompt_type)

    @staticmethod
    def get_available_prompts() -> Dict[str, Dict[str, str]]:
        """Получить список доступных промптов"""
        return PromptCollection.get_available_prompts()

    def add_context(self, context: str) -> None:
        """
        Добавить контекстную информацию в диалог
        
        Args:
            context: Контекстная информация
        """
        context_message = SystemMessage(content=f"Дополнительный контекст: {context}")
        self.messages.append(context_message)
        logging.info("Context added to conversation")

    def get_conversation_summary(self) -> str:
        """Получить краткое резюме диалога"""
        if len(self.messages) <= 1:  # Только системное сообщение
            return "Диалог еще не начался"

        user_messages = [msg.content for msg in self.messages if isinstance(msg, HumanMessage)]
        ai_messages = [msg.content for msg in self.messages if
                       hasattr(msg, 'content') and not isinstance(msg, HumanMessage) and not isinstance(msg, SystemMessage)]

        summary = f"Диалог содержит {len(user_messages)} сообщений пользователя и {len(ai_messages)} ответов AI (модель: {self.model_info['name']})"
        return summary

    def get_system_info(self) -> Dict[str, Any]:
        """Получить полную информацию о системе"""
        current_prompt = self.get_current_prompt_info()
        current_model = self.get_current_model_info()

        return {
            'model': current_model,
            'prompt': current_prompt,
            'conversation_length': len(self.messages),
            'max_tokens': current_model['max_tokens']
        }
