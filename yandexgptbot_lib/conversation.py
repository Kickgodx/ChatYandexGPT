"""
Модуль для управления историей диалогов
"""

import time
from typing import List, Dict, Any


class ConversationManager:
    """Класс для управления историей диалогов"""

    def __init__(self, config, bot):
        self.config = config
        self.bot = bot
        self.conversation_history: List[Dict[str, Any]] = []

    def add_message(self, user_message: str, ai_response: str) -> None:
        """Добавление сообщения в историю"""
        self.conversation_history.append({
            'user': user_message,
            'ai': ai_response,
            'timestamp': time.time()
        })

        # Ограничение истории
        max_history = 50
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

    def save_conversation(self) -> None:
        """Сохранение диалога в файл"""
        with open(self.config.RESPONSE_OUTPUT_FILENAME, 'a', encoding='utf-8') as f:
            for message in self.conversation_history[-2:]:  # Сохраняем только последние 2 сообщения
                f.write(f"User: {message['user']}\n")
                f.write(f"AI: {message['ai']}\n\n")

    def export_conversation(self, format: str = 'txt') -> str:
        """Экспорт диалога в файл"""
        if format == 'txt':
            filename = f"conversation_{int(time.time())}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                for message in self.conversation_history:
                    f.write(f"User: {message['user']}\n")
                    f.write(f"AI: {message['ai']}\n\n")
        return filename

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Получить историю диалога"""
        return self.conversation_history

    def clear_history(self) -> None:
        """Очистить историю диалога"""
        self.conversation_history.clear()

    def get_conversation_summary(self) -> str:
        """Получить краткое резюме диалога"""
        if not self.conversation_history:
            return "История диалога пуста"

        user_count = len(self.conversation_history)
        ai_count = len(self.conversation_history)

        return f"Диалог содержит {user_count} сообщений пользователя и {ai_count} ответов AI"
