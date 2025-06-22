import time


class ConversationManager:
    """Класс для управления историей диалогов"""

    def __init__(self, config, bot):
        self.config = config
        self.bot = bot
        self.conversation_history = []

    def add_message(self, user_message, ai_response):
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

    def save_conversation(self):
        """Сохранение диалога в файл"""
        with open(self.config.RESPONSE_OUTPUT_FILENAME, 'a', encoding='utf-8') as f:
            for message in self.conversation_history[-2:]:  # Сохраняем только последние 2 сообщения
                f.write(f"User: {message['user']}\n")
                f.write(f"AI: {message['ai']}\n\n")

    def export_conversation(self, format='txt'):
        """Экспорт диалога в файл"""
        if format == 'txt':
            filename = f"conversation_{int(time.time())}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                for message in self.conversation_history:
                    f.write(f"User: {message['user']}\n")
                    f.write(f"AI: {message['ai']}\n\n")
        return filename
