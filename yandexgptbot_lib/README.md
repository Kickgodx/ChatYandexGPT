# YandexGPTBot Library

Библиотека для работы с YandexGPT через голосовой и текстовый интерфейс.

## Установка

```bash
pip install yandexgptbot-lib
```

## Быстрый старт

### Базовое использование

```python
from yandexgptbot_lib import ChatYandexGPTBot

# Инициализация бота
bot = ChatYandexGPTBot(
    folder_id="your_folder_id",
    iam_token="your_iam_token",  # или используйте api_key
    prompt_type="programmer",
    model_name="yandexgpt-lite"
)

# Получение ответа
response = bot.get_response("Как написать функцию для сортировки списка в Python?")
print(response)
```

### CLI интерфейс

```bash
# Показать доступные модели
yandexgptbot --list-models

# Показать доступные промпты
yandexgptbot --list-prompts

# Запуск интерактивного режима
yandexgptbot --iam-token YOUR_TOKEN --folder-id YOUR_FOLDER_ID

# С определенной моделью и промптом
yandexgptbot --api-key YOUR_API_KEY --folder-id YOUR_FOLDER_ID --model yandexgpt --prompt teacher
```

## Основные компоненты

### ChatYandexGPTBot

Основной класс для работы с YandexGPT:

```python
from yandexgptbot_lib import ChatYandexGPTBot

bot = ChatYandexGPTBot(
    folder_id="your_folder_id",
    iam_token="your_iam_token",
    prompt_type="programmer",
    model_name="yandexgpt-lite"
)

# Получение ответа
response = bot.get_response("Ваш вопрос")

# Смена модели
bot.set_model("yandexgpt")

# Смена промпта
bot.set_prompt("teacher")

# Сброс диалога
bot.reset_conversation()

# Получение информации
current_prompt = bot.get_current_prompt_info()
current_model = bot.get_current_model_info()
summary = bot.get_conversation_summary()
```

### Аудио компоненты

```python
from yandexgptbot_lib import AudioRecorder, AudioProcessor, SpeechRecognizer, Config, Settings

config = Config()
settings = Settings()

# Запись аудио
recorder = AudioRecorder(config, settings)
recorder.start_recording('mic')  # или 'computer'
# ... запись ...
frames = recorder.stop_recording()

# Обработка аудио
processor = AudioProcessor(config)
processor.save_audio(frames, "output.wav")
processor.normalize_audio("output.wav")

# Распознавание речи
recognizer = SpeechRecognizer("./vosk-model-ru-0.42")
text = recognizer.transcribe_audio("output.wav")
```

### Управление диалогами

```python
from yandexgptbot_lib import ConversationManager

manager = ConversationManager(config, bot)
manager.add_message("Вопрос пользователя", "Ответ AI")
manager.save_conversation()
filename = manager.export_conversation('txt')
```

## Доступные модели

- **yandexgpt-lite**: Быстрая и экономичная модель (4000 токенов)
- **yandexgpt**: Базовая модель (8000 токенов)
- **yandexgpt-plus**: Продвинутая модель (8000 токенов)

## Доступные промпты

- **programmer**: Помощник по программированию
- **teacher**: Учитель-наставник
- **writer**: Писатель-редактор
- **analyst**: Аналитик-исследователь
- **creative**: Креативный помощник
- **general**: Универсальный помощник

## Примеры

Смотрите папку `examples/` для подробных примеров использования:

- `basic_usage.py` - Базовое использование
- `audio_processing.py` - Работа с аудио
- `prompt_management.py` - Управление промптами

## Требования

- Python 3.7+
- Yandex Cloud аккаунт с доступом к YandexGPT
- Модель Vosk для распознавания речи (опционально)

## Лицензия

MIT License 