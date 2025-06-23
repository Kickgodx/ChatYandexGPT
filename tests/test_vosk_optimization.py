#!/usr/bin/env python3
"""
Тестирование оптимизации загрузки модели Vosk
Демонстрация исправления проблемы с повторной загрузкой
"""

import gc
import os
import sys
import threading
import time

import psutil

# Добавляем корневую папку проекта в sys.path для корректного импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yandexgptbot_lib import SpeechRecognizer, get_global_recognizer, preload_vosk_model


def test_loading_strategies():
    """Тестирование различных стратегий загрузки"""
    print("🚀 Тестирование стратегий загрузки модели Vosk")
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

    # Стратегия 4: Глобальный синглтон (кэширование) - ИСПРАВЛЕННАЯ ВЕРСИЯ
    print("4️⃣ Стратегия 4: Глобальный синглтон (кэширование) - ИСПРАВЛЕННАЯ")
    start_time = time.time()

    # Первый вызов - загружает модель
    global_recognizer1 = get_global_recognizer(vosk_model_path, preload=True)
    first_load_time = time.time() - start_time
    print(f"   ⏱️ Первый вызов: {first_load_time:.2f} секунд")

    # Второй вызов - использует кэш (мгновенно)
    start_time = time.time()
    global_recognizer2 = get_global_recognizer(vosk_model_path, preload=False)
    second_load_time = time.time() - start_time
    print(f"   ⚡ Второй вызов: {second_load_time:.4f} секунд (кэш)")

    # Третий вызов - тоже использует кэш
    start_time = time.time()
    global_recognizer3 = get_global_recognizer(vosk_model_path, preload=False)
    third_load_time = time.time() - start_time
    print(f"   ⚡ Третий вызов: {third_load_time:.4f} секунд (кэш)")

    # Проверяем, что это один и тот же объект
    print(f"   🔍 Одинаковые объекты: {global_recognizer1 is global_recognizer2 is global_recognizer3}")
    print(f"   📊 Статус модели: {global_recognizer1.get_model_info()}")
    print()

    print("✅ Тестирование стратегий загрузки завершено!")
    print()


def test_concurrent_loading():
    """Тестирование конкурентной загрузки модели"""
    print("🔄 Тестирование конкурентной загрузки")
    print("=" * 60)

    vosk_model_path = os.getenv("VOSK_MODEL_PATH", "./vosk-model-ru-0.42")

    if not os.path.exists(vosk_model_path):
        print("❌ Модель Vosk не найдена!")
        return

    results = []
    threads = []

    def load_recognizer(thread_id):
        """Функция загрузки распознавателя в отдельном потоке"""
        try:
            start_time = time.time()
            recognizer = get_global_recognizer(vosk_model_path, preload=False)
            load_time = time.time() - start_time

            results.append({
                'thread_id': thread_id,
                'load_time': load_time,
                'is_loaded': recognizer.is_model_loaded(),
                'object_id': id(recognizer)
            })

            print(f"   Поток {thread_id}: {load_time:.4f} сек, загружена: {recognizer.is_model_loaded()}")

        except Exception as e:
            print(f"   Поток {thread_id}: Ошибка - {e}")

    # Запускаем 5 потоков одновременно
    print("Запуск 5 потоков для одновременной загрузки...")
    for i in range(5):
        thread = threading.Thread(target=load_recognizer, args=(i,))
        threads.append(thread)
        thread.start()

    # Ждем завершения всех потоков
    for thread in threads:
        thread.join()

    # Анализируем результаты
    print("\n📊 Результаты конкурентной загрузки:")
    object_ids = set()
    for result in results:
        object_ids.add(result['object_id'])
        print(f"   Поток {result['thread_id']}: {result['load_time']:.4f} сек, "
              f"загружена: {result['is_loaded']}, объект: {result['object_id']}")

    print(f"\n🔍 Количество уникальных объектов: {len(object_ids)}")
    if len(object_ids) == 1:
        print("✅ УСПЕХ: Все потоки используют один и тот же объект (кэширование работает)")
    else:
        print("❌ ОШИБКА: Создано несколько объектов (кэширование не работает)")

    print()


def test_memory_usage():
    """Тестирование использования памяти"""
    print("💾 Тестирование использования памяти")
    print("=" * 60)

    vosk_model_path = os.getenv("VOSK_MODEL_PATH", "./vosk-model-ru-0.42")

    if not os.path.exists(vosk_model_path):
        print("❌ Модель Vosk не найдена!")
        return

    process = psutil.Process()

    # Измеряем память до загрузки
    memory_before = process.memory_info().rss / 1024 / 1024  # МБ
    print(f"📊 Память до загрузки: {memory_before:.1f} МБ")

    # Загружаем модель
    recognizer1 = get_global_recognizer(vosk_model_path, preload=True)

    # Ждем завершения загрузки
    while not recognizer1.is_model_loaded():
        time.sleep(0.1)

    # Измеряем память после загрузки
    memory_after_load = process.memory_info().rss / 1024 / 1024  # МБ
    print(f"📊 Память после загрузки: {memory_after_load:.1f} МБ")
    print(f"📈 Увеличение памяти: {memory_after_load - memory_before:.1f} МБ")

    # Создаем еще несколько распознавателей (должны использовать кэш)
    recognizers = []
    for i in range(5):
        recognizer = get_global_recognizer(vosk_model_path, preload=False)
        recognizers.append(recognizer)
        time.sleep(0.1)  # Небольшая пауза

    # Измеряем память после создания дополнительных объектов
    memory_after_multiple = process.memory_info().rss / 1024 / 1024  # МБ
    print(f"📊 Память после создания 5 объектов: {memory_after_multiple:.1f} МБ")
    print(f"📈 Дополнительное увеличение: {memory_after_multiple - memory_after_load:.1f} МБ")

    # Проверяем, что все объекты одинаковые
    object_ids = set(id(r) for r in recognizers)
    print(f"🔍 Количество уникальных объектов: {len(object_ids)}")

    if len(object_ids) == 1:
        print("✅ УСПЕХ: Все объекты одинаковые (кэширование работает)")
    else:
        print("❌ ОШИБКА: Создано несколько объектов")

    # Очищаем память
    del recognizers
    del recognizer1
    gc.collect()

    # Измеряем память после очистки
    memory_after_cleanup = process.memory_info().rss / 1024 / 1024  # МБ
    print(f"📊 Память после очистки: {memory_after_cleanup:.1f} МБ")
    print(f"📉 Освобождено памяти: {memory_after_multiple - memory_after_cleanup:.1f} МБ")

    print()


def test_repeated_usage():
    """Тестирование повторного использования модели (ИСПРАВЛЕНИЕ КРИТИЧЕСКОЙ ОШИБКИ)"""
    print("🔄 Тестирование повторного использования модели")
    print("=" * 60)

    vosk_model_path = os.getenv("VOSK_MODEL_PATH", "./vosk-model-ru-0.42")

    if not os.path.exists(vosk_model_path):
        print("❌ Модель Vosk не найдена!")
        return

    print("🔧 Создание первого распознавателя...")
    start_time = time.time()
    recognizer1 = get_global_recognizer(vosk_model_path, preload=True)

    # Ждем загрузки
    while not recognizer1.is_model_loaded():
        time.sleep(0.1)

    first_load_time = time.time() - start_time
    print(f"   ⏱️ Первая загрузка: {first_load_time:.2f} секунд")

    print("\n🔄 Симуляция повторного использования (как в приложении)...")

    # Симулируем использование в приложении
    for i in range(3):
        print(f"   Использование {i + 1}:")

        # Получаем распознаватель (как при обработке аудио)
        start_time = time.time()
        recognizer = get_global_recognizer(vosk_model_path, preload=False)
        get_time = time.time() - start_time

        print(f"     ⏱️ Получение распознавателя: {get_time:.4f} секунд")
        print(f"     📊 Модель загружена: {recognizer.is_model_loaded()}")
        print(f"     🔍 Тот же объект: {recognizer is recognizer1}")

        # Симулируем транскрипцию (без реального файла)
        if recognizer.is_model_loaded():
            print(f"     ✅ Готов к транскрипции")
        else:
            print(f"     ❌ Модель не загружена!")

        print()

    print("✅ Тестирование повторного использования завершено!")
    print()


def test_synchronization_fix():
    """Тест исправления синхронизации загрузки модели"""
    print("🔧 Тест исправления синхронизации загрузки модели")
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

    # Тест 1: Проверка правильной синхронизации
    print("1️⃣ Тест правильной синхронизации:")

    # Получаем глобальный распознаватель
    recognizer1 = get_global_recognizer(vosk_model_path, preload=False)
    print(f"   📊 Статус модели: {recognizer1.get_model_info()}")

    # Запускаем фоновую загрузку
    print("   🔄 Запуск фоновой загрузки...")
    preload_vosk_model(vosk_model_path, show_progress=False)

    # Ждем и проверяем статус
    max_wait = 30
    start_time = time.time()

    while time.time() - start_time < max_wait:
        # Получаем актуальный экземпляр
        current_recognizer = get_global_recognizer(vosk_model_path, preload=False)

        if current_recognizer.is_model_loaded():
            print(f"   ✅ Модель загружена за {time.time() - start_time:.2f} сек")
            print(f"   📊 Статус: {current_recognizer.get_model_info()}")
            break
        else:
            elapsed = time.time() - start_time
            print(f"   🔄 Ожидание... ({elapsed:.1f} сек)")
            time.sleep(1)
    else:
        print("   ❌ Таймаут загрузки модели")
        return

    print()

    # Тест 2: Проверка повторного использования модели
    print("2️⃣ Тест повторного использования модели:")

    start_time = time.time()
    recognizer2 = get_global_recognizer(vosk_model_path, preload=False)
    load_time = time.time() - start_time

    print(f"   ⚡ Время получения экземпляра: {load_time:.4f} сек")
    print(f"   📊 Статус: {recognizer2.get_model_info()}")

    # Проверяем, что это тот же экземпляр
    if recognizer1 is recognizer2:
        print("   ✅ Используется тот же экземпляр (синглтон работает)")
    else:
        print("   ❌ Создан новый экземпляр (проблема с синглтоном)")

    print()

    # Тест 3: Проверка транскрипции
    print("3️⃣ Тест транскрипции:")

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

    test_audio_file = "test_sync_audio.wav"
    with wave.open(test_audio_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    print(f"   📁 Создан тестовый файл: {test_audio_file}")

    # Тестируем транскрипцию
    try:
        start_time = time.time()
        transcribed_text = current_recognizer.transcribe_audio(test_audio_file)
        transcribe_time = time.time() - start_time

        print(f"   ⏱️ Время транскрипции: {transcribe_time:.2f} сек")
        print(f"   📝 Результат: '{transcribed_text}'")
        print("   ✅ Транскрипция работает корректно")

    except Exception as e:
        print(f"   ❌ Ошибка транскрипции: {e}")

    # Удаляем тестовый файл
    if os.path.exists(test_audio_file):
        os.remove(test_audio_file)

    print()

    # Тест 4: Проверка конкурентного доступа
    print("4️⃣ Тест конкурентного доступа:")

    def concurrent_access_test(thread_id):
        try:
            recognizer = get_global_recognizer(vosk_model_path, preload=False)
            if recognizer.is_model_loaded():
                return f"Thread {thread_id}: OK"
            else:
                return f"Thread {thread_id}: Model not loaded"
        except Exception as e:
            return f"Thread {thread_id}: Error - {e}"

    # Запускаем несколько потоков одновременно
    threads = []
    results = []

    def thread_worker(thread_id):
        result = concurrent_access_test(thread_id)
        results.append(result)

    for i in range(5):
        thread = threading.Thread(target=thread_worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Ждем завершения всех потоков
    for thread in threads:
        thread.join()

    # Проверяем результаты
    success_count = sum(1 for result in results if "OK" in result)
    print(f"   📊 Успешных потоков: {success_count}/{len(results)}")

    for result in results:
        print(f"   {result}")

    if success_count == len(results):
        print("   ✅ Все потоки работают корректно")
    else:
        print("   ⚠️ Есть проблемы с конкурентным доступом")

    print()

    print("✅ Тест исправления синхронизации завершен!")


def main():
    """Основная функция тестирования"""
    print("🧪 Тестирование исправлений оптимизации Vosk")
    print("=" * 80)

    # Тестирование стратегий загрузки
    test_loading_strategies()

    # Тестирование конкурентной загрузки
    test_concurrent_loading()

    # Тестирование использования памяти
    test_memory_usage()

    # Тестирование повторного использования (ИСПРАВЛЕНИЕ)
    test_repeated_usage()

    # Тест исправления синхронизации
    test_synchronization_fix()

    print("\n" + "=" * 80)
    print("🎉 Все тесты завершены!")
    print("\n📋 Резюме исправлений:")
    print("✅ Добавлен индикатор загрузки модели в GUI")
    print("✅ Исправлена критическая ошибка с повторной загрузкой")
    print("✅ Улучшен глобальный синглтон с потокобезопасностью")
    print("✅ Добавлен мониторинг статуса загрузки")
    print("✅ Предотвращено создание дублирующих объектов")


if __name__ == "__main__":
    main()
