#!/usr/bin/env python3
"""
Пример работы с аудио компонентами библиотеки YandexGPTBot
Демонстрация оптимизированной загрузки модели Vosk
"""

import os
import sys
import time

# Добавляем корневую папку проекта в sys.path для корректного импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yandexgptbot_lib import AudioRecorder, AudioProcessor, SpeechRecognizer, Config, Settings, ChatYandexGPTBot
from yandexgptbot_lib.audio import preload_vosk_model, get_global_recognizer


def demonstrate_model_loading_strategies():
    """Демонстрация различных стратегий загрузки модели"""
    print("🚀 Демонстрация стратегий загрузки модели Vosk")
    print("=" * 60)

    vosk_model_path = os.getenv("VOSK_MODEL_PATH", "./vosk-model-ru-0.42")

    # Проверка наличия модели Vosk
    if not os.path.exists(vosk_model_path):
        print("❌ Модель Vosk не найдена!")
        print(f"   Скачайте модель с https://alphacephei.com/vosk/models")
        print(f"   И распакуйте в папку: {vosk_model_path}")
        return

    print(f"📁 Модель найдена: {vosk_model_path}")
    print()

    # Стратегия 1: Обычная загрузка (блокирующая)
    print("1️⃣ Стратегия 1: Обычная загрузка (блокирующая)")
    start_time = time.time()

    recognizer1 = SpeechRecognizer(vosk_model_path, preload_model=True, show_progress=True)

    load_time = time.time() - start_time
    print(f"   ⏱️ Время загрузки: {load_time:.2f} секунд")
    print(f"   📊 Статус модели: {recognizer1.get_model_info()}")
    print()

    # Стратегия 2: Ленивая загрузка (по требованию)
    print("2️⃣ Стратегия 2: Ленивая загрузка (по требованию)")
    start_time = time.time()

    recognizer2 = SpeechRecognizer(vosk_model_path, preload_model=False, show_progress=False)
    print(f"   ⚡ Инициализация мгновенная")
    print(f"   📊 Статус модели: {recognizer2.get_model_info()}")

    # Загружаем при первом использовании
    print("   🔄 Первое использование (загрузка модели)...")
    start_load = time.time()
    recognizer2._ensure_model_loaded()
    load_time = time.time() - start_load
    print(f"   ⏱️ Время загрузки при использовании: {load_time:.2f} секунд")
    print()

    # Стратегия 3: Предварительная загрузка в фоне
    print("3️⃣ Стратегия 3: Предварительная загрузка в фоне")
    start_time = time.time()

    # Запускаем предварительную загрузку
    preload_vosk_model(vosk_model_path, show_progress=True)

    # Создаем распознаватель (модель уже загружается в фоне)
    recognizer3 = SpeechRecognizer(vosk_model_path, preload_model=False, show_progress=False)
    print(f"   ⚡ Инициализация мгновенная")

    # Ждем завершения загрузки
    while not recognizer3.is_model_loaded():
        print("   🔄 Ожидание завершения загрузки...")
        time.sleep(1)

    total_time = time.time() - start_time
    print(f"   ⏱️ Общее время: {total_time:.2f} секунд")
    print(f"   📊 Статус модели: {recognizer3.get_model_info()}")
    print()

    # Стратегия 4: Глобальный синглтон (кэширование)
    print("4️⃣ Стратегия 4: Глобальный синглтон (кэширование)")
    start_time = time.time()

    # Первый вызов - загружает модель
    global_recognizer1 = get_global_recognizer(vosk_model_path, preload=True)
    first_load_time = time.time() - start_time
    print(f"   ⏱️ Первый вызов: {first_load_time:.2f} секунд")

    # Второй вызов - использует кэш
    start_time = time.time()
    global_recognizer2 = get_global_recognizer(vosk_model_path, preload=False)
    second_load_time = time.time() - start_time
    print(f"   ⚡ Второй вызов: {second_load_time:.4f} секунд (кэш)")

    print(f"   📊 Статус модели: {global_recognizer1.get_model_info()}")
    print()

    print("✅ Демонстрация стратегий загрузки завершена!")
    print()


def demonstrate_audio_processing():
    """Демонстрация работы с аудио компонентами"""
    print("🎤 Демонстрация работы с аудио компонентами")
    print("=" * 60)

    # Инициализация конфигурации
    config = Config()
    settings = Settings()

    # Получение credentials из переменных окружения
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    iam_token = os.getenv("YANDEX_IAM_TOKEN")
    api_key = os.getenv("YANDEX_API_KEY")
    vosk_model_path = os.getenv("VOSK_MODEL_PATH", "./vosk-model-ru-0.42")

    try:
        # Инициализация компонентов с оптимизированной загрузкой
        print("🔧 Инициализация компонентов...")
        recorder = AudioRecorder(config, settings)
        processor = AudioProcessor(config)

        # Используем глобальный распознаватель для кэширования
        recognizer = get_global_recognizer(vosk_model_path, preload=True)

        # Инициализация бота (если есть credentials)
        bot = None
        if folder_id and (iam_token or api_key):
            if iam_token:
                bot = ChatYandexGPTBot(
                    folder_id=folder_id,
                    iam_token=iam_token,
                    prompt_type="general",
                    model_name="yandexgpt-lite"
                )
            else:
                bot = ChatYandexGPTBot(
                    folder_id=folder_id,
                    api_key=api_key,
                    prompt_type="general",
                    model_name="yandexgpt-lite"
                )

        print("🎤 Компоненты инициализированы успешно!")
        print()

        # Демонстрация записи с микрофона
        print("1️⃣ Демонстрация записи с микрофона")
        print("   Нажмите Enter для начала записи (5 секунд)...")
        input()

        try:
            recorder.start_recording('mic')
            print("   🔴 Запись началась... Говорите!")

            # Запись в течение 5 секунд
            time.sleep(5)

            frames = recorder.stop_recording()
            print("   ⏹️ Запись остановлена")

            if frames:
                # Сохранение аудио
                audio_file = "demo_mic.wav"
                processor.save_audio(frames, audio_file)
                print(f"   💾 Аудио сохранено в: {audio_file}")

                # Проверка качества
                audio_data = b''.join(frames)
                quality_ok, quality_msg = recorder.check_audio_quality(audio_data)
                if not quality_ok:
                    print(f"   ⚠️ {quality_msg}")

                # Нормализация
                processor.normalize_audio(audio_file)
                print("   🔊 Аудио нормализовано")

                # Распознавание речи (модель уже загружена)
                print("   🎯 Распознавание речи...")
                transcribed_text = recognizer.transcribe_audio(audio_file)

                if transcribed_text.strip():
                    print(f"   📝 Распознанный текст: '{transcribed_text}'")

                    # Отправка в AI (если бот доступен)
                    if bot:
                        print("   🤖 Отправка в AI...")
                        response = bot.get_response(transcribed_text)
                        print(f"   💬 Ответ AI: {response}")
                else:
                    print("   ❌ Речь не распознана")
            else:
                print("   ❌ Ошибка записи")

        except Exception as e:
            print(f"   ❌ Ошибка записи: {e}")

        print()

        # Демонстрация live-транскрипции
        print("2️⃣ Демонстрация live-транскрипции")

        # Проверяем, есть ли уже записанный файл
        if os.path.exists("demo_mic.wav"):
            print("   📁 Используем записанный файл: demo_mic.wav")
            audio_file = "demo_mic.wav"
        else:
            print("   📁 Создаем тестовый аудиофайл...")
            # Создаем простой тестовый аудиофайл
            import wave
            import numpy as np

            # Генерируем простой синусоидальный сигнал
            sample_rate = 16000
            duration = 2  # секунды
            frequency = 440  # Hz

            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
            audio_data = (audio_data * 32767).astype(np.int16)

            audio_file = "test_audio.wav"
            with wave.open(audio_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())

            print(f"   💾 Тестовый файл создан: {audio_file}")

        # Live-транскрипция
        try:
            print("   🎯 Live-транскрипция...")

            def live_callback(text, is_final):
                if is_final:
                    print(f"   ✅ Финальный результат: '{text}'")
                else:
                    if text:
                        print(f"   🎯 Промежуточный: '{text}'")

            # Используем live-транскрипцию
            final_text = recognizer.transcribe_audio(audio_file, live_callback=live_callback)

            if final_text.strip():
                print(f"   📝 Итоговый текст: '{final_text}'")
            else:
                print("   📝 Итоговый текст: (пусто - это нормально для тестового файла)")

        except Exception as e:
            print(f"   ❌ Ошибка live-транскрипции: {e}")

        print()

        # Демонстрация работы с существующим аудиофайлом
        print("3️⃣ Демонстрация работы с существующим аудиофайлом")

        # Обработка файла
        try:
            # Нормализация
            processor.normalize_audio(audio_file)
            print("   🔊 Аудио нормализовано")

            # Распознавание
            print("   🎯 Распознавание речи...")
            transcribed_text = recognizer.transcribe_audio(audio_file)

            if transcribed_text.strip():
                print(f"   📝 Распознанный текст: '{transcribed_text}'")
            else:
                print("   📝 Распознанный текст: (пусто - это нормально для тестового файла)")

        except Exception as e:
            print(f"   ❌ Ошибка обработки: {e}")

        print()
        print("✅ Пример работы с аудио завершен!")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


def main():
    """Основная функция примера"""
    print("🎤 Пример работы с аудио компонентами YandexGPTBot")
    print("🚀 С оптимизированной загрузкой модели Vosk")
    print("=" * 80)

    # Демонстрация стратегий загрузки
    demonstrate_model_loading_strategies()

    # Демонстрация работы с аудио
    demonstrate_audio_processing()


if __name__ == "__main__":
    main()
