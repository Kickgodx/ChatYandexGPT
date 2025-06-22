#!/usr/bin/env python3
"""
Пример работы с аудио компонентами библиотеки YandexGPTBot
"""

import os
import sys
import time

# Добавляем корневую папку проекта в sys.path для корректного импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yandexgptbot_lib import AudioRecorder, AudioProcessor, SpeechRecognizer, Config, Settings, ChatYandexGPTBot


def main():
    """Основная функция примера"""
    print("🎤 Пример работы с аудио компонентами YandexGPTBot")
    print("=" * 60)

    # Инициализация конфигурации
    config = Config()
    settings = Settings()

    # Получение credentials из переменных окружения
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    iam_token = os.getenv("YANDEX_IAM_TOKEN")
    api_key = os.getenv("YANDEX_API_KEY")
    vosk_model_path = os.getenv("VOSK_MODEL_PATH", "./vosk-model-ru-0.42")

    # Проверка наличия модели Vosk
    if not os.path.exists(vosk_model_path):
        print("❌ Модель Vosk не найдена!")
        print(f"   Скачайте модель с https://alphacephei.com/vosk/models")
        print(f"   И распакуйте в папку: {vosk_model_path}")
        print(f"   Или укажите правильный путь в переменной VOSK_MODEL_PATH")
        return

    try:
        # Инициализация компонентов
        recorder = AudioRecorder(config, settings)
        processor = AudioProcessor(config)
        recognizer = SpeechRecognizer(vosk_model_path)

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

                # Распознавание речи
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

        # Демонстрация работы с существующим аудиофайлом
        print("2️⃣ Демонстрация работы с существующим аудиофайлом")
        
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


if __name__ == "__main__":
    main() 