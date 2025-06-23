# YandexGPTBot Library

Библиотека для работы с YandexGPT через голосовой и текстовый интерфейс с оптимизированной загрузкой модели Vosk.

## 🆕 Новые возможности v1.1.0

- **🚀 Оптимизированная загрузка модели Vosk** - ускоренная инициализация
- **🔄 Live-транскрипция** - отображение распознанного текста в реальном времени
- **💾 Глобальное кэширование** - переиспользование загруженной модели
- **⚡ Ленивая загрузка** - загрузка модели только при необходимости
- **🔄 Фоновая загрузка** - загрузка модели в отдельном потоке

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
    iam_token="your_iam_token",
    prompt_type="programmer",
    model_name="yandexgpt-lite"
)

# Получение ответа
response = bot.get_response("Как написать функцию на Python?")
print(response)
```

### Аудио компоненты с оптимизацией

```python
from yandexgptbot_lib import AudioRecorder, AudioProcessor, SpeechRecognizer, Config, Settings
from yandexgptbot_lib import get_global_recognizer, preload_vosk_model

# Конфигурация
config = Config()
settings = Settings()

# Запись аудио
recorder = AudioRecorder(config, settings)
recorder.start_recording('mic')
# ... запись ...
frames = recorder.stop_recording()

# Обработка аудио
processor = AudioProcessor(config)
processor.save_audio(frames, "output.wav")
processor.normalize_audio("output.wav")

# Распознавание речи (с оптимизацией)
recognizer = get_global_recognizer("./vosk-model-ru-0.42", preload=True)
text = recognizer.transcribe_audio("output.wav")
```

## 🚀 Оптимизация загрузки модели Vosk

### Стратегии загрузки

#### 1. Обычная загрузка (блокирующая)

```python
from yandexgptbot_lib import SpeechRecognizer

# Модель загружается сразу при создании объекта
recognizer = SpeechRecognizer(model_path, preload_model=True, show_progress=True)
```

#### 2. Ленивая загрузка (по требованию)

```python
# Модель загружается только при первом использовании
recognizer = SpeechRecognizer(model_path, preload_model=False)
# Модель загрузится автоматически при вызове transcribe_audio()
```

#### 3. Предварительная загрузка в фоне

```python
from yandexgptbot_lib import preload_vosk_model

# Запускаем загрузку в отдельном потоке
preload_vosk_model(model_path, show_progress=True)

# Создаем распознаватель (модель уже загружается в фоне)
recognizer = SpeechRecognizer(model_path, preload_model=False)

# Проверяем готовность
while not recognizer.is_model_loaded():
    time.sleep(0.1)
```

#### 4. Глобальное кэширование (синглтон)

```python
from yandexgptbot_lib import get_global_recognizer

# Первый вызов загружает модель
recognizer1 = get_global_recognizer(model_path, preload=True)

# Последующие вызовы используют кэш (мгновенно)
recognizer2 = get_global_recognizer(model_path, preload=False)
```

### Сравнение производительности

| Стратегия   | Первый запуск | Повторный запуск | Память           |
|-------------|---------------|------------------|------------------|
| Обычная     | ~2-5 сек      | ~2-5 сек         | Нормальная       |
| Ленивая     | Мгновенно     | ~2-5 сек         | Нормальная       |
| Фоновая     | ~2-5 сек      | Мгновенно        | Нормальная       |
| Кэширование | ~2-5 сек      | Мгновенно        | Оптимизированная |

## 🎯 Live-транскрипция

### Базовое использование

```python
from yandexgptbot_lib import SpeechRecognizer

recognizer = SpeechRecognizer("./vosk-model-ru-0.42")

# Live-транскрипция с callback
def live_callback(text, is_final):
    if is_final:
        print(f"Финальный результат: {text}")
    else:
        print(f"Промежуточный: {text}")

# Использование live-транскрипции
final_text = recognizer.transcribe_audio("audio.wav", live_callback=live_callback)
```

### Расширенное использование

```python
# С отдельными callback для промежуточных и финальных результатов
def update_callback(text):
    print(f"Обновление: {text}")

def final_callback(text):
    print(f"Готово: {text}")

# Запуск в отдельном потоке
thread = recognizer.transcribe_audio_live("audio.wav", update_callback, final_callback)
thread.join()  # Ждем завершения
```

## Управление диалогами

```python
from yandexgptbot_lib import ConversationManager

manager = ConversationManager(config, bot)

# Добавление сообщений
manager.add_message("Привет", "Здравствуйте!")

# Сохранение диалога
manager.save_conversation()

# Экспорт диалога
filename = manager.export_conversation('txt')
print(f"Диалог экспортирован в: {filename}")

# Получение истории
history = manager.get_conversation_history()
print(f"Количество сообщений: {len(history)}")
```

## Управление промптами

```python
from yandexgptbot_lib import PromptCollection

# Получение списка доступных промптов
available_prompts = PromptCollection.get_available_prompts()
for key, info in available_prompts.items():
    print(f"{key}: {info['name']} - {info['description']}")

# Получение конкретного промпта
programmer_prompt = PromptCollection.get_prompt("programmer")
print(f"Промпт программиста: {programmer_prompt['name']}")

# Получение содержимого промпта
content = PromptCollection.get_prompt_content("teacher")
print(f"Содержимое: {content[:100]}...")
```

## Конфигурация

```python
from yandexgptbot_lib import Config, Settings

# Конфигурация приложения
config = Config()
print(f"Частота дискретизации: {config.RATE}")
print(f"Каналы: {config.CHANNELS}")

# Настройки пользователя
settings = Settings()
settings.set("audio_quality_threshold", 20)
settings.save_settings()
```

## CLI интерфейс

```bash
# Запуск в режиме чата
python -m yandexgptbot_lib.cli

# С параметрами
python -m yandexgptbot_lib.cli --model yandexgpt-lite --prompt programmer
```

## Примеры

### Полный пример с аудио

```python
import os
from yandexgptbot_lib import (
    ChatYandexGPTBot, AudioRecorder, AudioProcessor,
    SpeechRecognizer, Config, Settings, get_global_recognizer
)

# Настройка
config = Config()
settings = Settings()

# Инициализация бота
bot = ChatYandexGPTBot(
    folder_id=os.getenv("YANDEX_FOLDER_ID"),
    iam_token=os.getenv("YANDEX_IAM_TOKEN"),
    prompt_type="programmer"
)

# Запись аудио
recorder = AudioRecorder(config, settings)
recorder.start_recording('mic')
# ... запись 5 секунд ...
frames = recorder.stop_recording()

# Обработка аудио
processor = AudioProcessor(config)
processor.save_audio(frames, "question.wav")
processor.normalize_audio("question.wav")

# Распознавание речи (с оптимизацией)
recognizer = get_global_recognizer("./vosk-model-ru-0.42", preload=True)
text = recognizer.transcribe_audio("question.wav")

if text.strip():
    # Получение ответа от AI
    response = bot.get_response(text)
    print(f"Вопрос: {text}")
    print(f"Ответ: {response}")
else:
    print("Речь не распознана")
```

### Пример с live-транскрипцией

```python
from yandexgptbot_lib import SpeechRecognizer, get_global_recognizer

# Получаем глобальный распознаватель
recognizer = get_global_recognizer("./vosk-model-ru-0.42", preload=True)


# Функция для обновления GUI
def update_gui(text, is_final):
    if is_final:
        print(f"✅ Финальный результат: {text}")
        # Здесь можно отправить в AI
    else:
        print(f"🎯 Распознавание...: {text}")


# Live-транскрипция
final_text = recognizer.transcribe_audio("audio.wav", live_callback=update_gui)
```

## Тестирование оптимизации

Для тестирования различных стратегий загрузки используйте:

```bash
python test_vosk_optimization.py
```

Этот скрипт демонстрирует:

- Сравнение времени загрузки разных стратегий
- Тестирование конкурентной загрузки
- Анализ использования памяти

## Требования

- Python 3.7+
- Модель Vosk (скачать с https://alphacephei.com/vosk/models)
- Credentials Yandex Cloud

## Поддержка

- **Документация:** [README.md](../README.md)
- **Примеры:** [examples/](../examples/)
- **Тесты:** [test_vosk_optimization.py](../test_vosk_optimization.py)

## Лицензия

MIT License 
