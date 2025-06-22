# ChatYandexGPT - Проект и Библиотека

Этот проект содержит как полноценное приложение с GUI, так и библиотеку для использования в других проектах.

## Структура проекта

```
ChatYandexGPT/
├── interface_AI.py              # Основное приложение с GUI
├── yandexchat_bot.py            # Оригинальный класс бота
├── src/                         # Исходный код приложения
├── chatyandexgpt/               # 🆕 Библиотечная версия
│   ├── __init__.py
│   ├── bot.py                   # Адаптированный класс бота
│   ├── audio.py                 # Аудио компоненты
│   ├── prompts.py               # Коллекция промптов
│   ├── conversation.py          # Управление диалогами
│   ├── config.py                # Конфигурация
│   ├── settings.py              # Настройки
│   └── cli.py                   # CLI интерфейс
├── examples/                    # 🆕 Примеры использования библиотеки
│   ├── basic_usage.py
│   ├── audio_processing.py
│   └── prompt_management.py
├── setup.py                     # 🆕 Конфигурация пакета
├── build_library.py             # 🆕 Скрипт сборки
└── README.md                    # Документация приложения
```

## Использование

### 1. Оригинальное приложение (GUI)

Запустите полноценное приложение с графическим интерфейсом:

```bash
python interface_AI.py
```

### 2. Библиотека

#### Установка библиотеки

```bash
# Сборка и установка в режиме разработки
cd yandexgptbot_lib/package
python package_builder.py

# Или установка из PyPI (когда будет опубликована)
pip install chatyandexgpt
```

#### Использование в коде

```python
from chatyandexgpt import ChatYandexGPTBot

bot = ChatYandexGPTBot(
    folder_id="your_folder_id",
    iam_token="your_iam_token",
    prompt_type="programmer",
    model_name="yandexgpt-lite"
)

response = bot.get_response("Как написать функцию сортировки?")
print(response)
```

#### CLI интерфейс

```bash
# Показать доступные модели и промпты
chatyandexgpt --list-models
chatyandexgpt --list-prompts

# Интерактивный режим
chatyandexgpt --iam-token YOUR_TOKEN --folder-id YOUR_FOLDER_ID

# С определенными настройками
chatyandexgpt --api-key YOUR_API_KEY --folder-id YOUR_FOLDER_ID --model yandexgpt --prompt teacher
```

## Примеры использования библиотеки

### Базовое использование

```python
# examples/basic_usage.py
from chatyandexgpt import ChatYandexGPTBot

bot = ChatYandexGPTBot(
    folder_id="your_folder_id",
    iam_token="your_iam_token",
    prompt_type="programmer"
)

response = bot.get_response("Как написать функцию для сортировки списка в Python?")
print(response)
```

### Работа с аудио

```python
# examples/audio_processing.py
from chatyandexgpt import AudioRecorder, SpeechRecognizer, ChatYandexGPTBot

# Запись и распознавание речи
recorder = AudioRecorder(config, settings)
recorder.start_recording('mic')
# ... запись ...
frames = recorder.stop_recording()

recognizer = SpeechRecognizer("./vosk-model-ru-0.42")
text = recognizer.transcribe_audio("audio.wav")

# Отправка в AI
bot = ChatYandexGPTBot(folder_id="...", iam_token="...")
response = bot.get_response(text)
```

### Управление промптами

```python
# examples/prompt_management.py
from chatyandexgpt import ChatYandexGPTBot, PromptCollection

# Получение доступных промптов
prompts = PromptCollection.get_available_prompts()

# Смена промпта
bot = ChatYandexGPTBot(folder_id="...", iam_token="...")
bot.set_prompt("teacher")
response = bot.get_response("Объясни фотосинтез")
```

## Разработка

### Сборка библиотеки

```bash
cd yandexgptbot_lib/package
python package_builder.py
```

Этот скрипт:

1. Очищает предыдущие сборки
2. Собирает библиотеку
3. Устанавливает её в режиме разработки
4. Тестирует импорт

### Публикация на PyPI

```bash
cd yandexgptbot_lib/package
# Сборка дистрибутива
python setup.py sdist bdist_wheel

# Публикация
python -m twine upload dist/*
```

## Основные отличия библиотеки от приложения

| Компонент         | Приложение         | Библиотека               |
|-------------------|--------------------|--------------------------|
| **Интерфейс**     | GUI (tkinter)      | API + CLI                |
| **Аудио**         | Полная интеграция  | Модульные компоненты     |
| **Промпты**       | Встроенные в GUI   | Программное управление   |
| **Настройки**     | JSON файл          | Программная конфигурация |
| **Использование** | Готовое приложение | Встраивание в проекты    |

## Преимущества библиотеки

1. **Модульность** - можно использовать только нужные компоненты
2. **Программируемость** - полный контроль через API
3. **CLI интерфейс** - удобно для автоматизации
4. **Типизация** - поддержка type hints
5. **Документация** - подробные docstrings
6. **Примеры** - готовые примеры использования

## Миграция с приложения на библиотеку

Если вы используете оригинальное приложение и хотите перейти на библиотеку:

1. **Импорты**: Замените импорты из `src/` на `chatyandexgpt`
2. **Инициализация**: Используйте `ChatYandexGPTBot` напрямую
3. **Аудио**: Используйте отдельные классы `AudioRecorder`, `SpeechRecognizer`
4. **Промпты**: Используйте `PromptCollection` для управления промптами

## Поддержка

- **Приложение**: Используйте `interface_AI.py` для GUI приложения
- **Библиотека**: Используйте `chatyandexgpt/` для программного использования
- **Примеры**: Смотрите папку `examples/` для примеров использования библиотеки

## Лицензия

MIT License 
