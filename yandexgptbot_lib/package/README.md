# Сборка библиотеки YandexGPTBot

Эта папка содержит файлы для сборки и публикации библиотеки `yandexgptbot-lib`.

## Структура файлов

- `setup.py` - конфигурация пакета для PyPI
- `package_builder.py` - скрипт для сборки и установки пакета
- `MANIFEST.in` - список файлов для включения в дистрибутив
- `README_LIBRARY.md` - документация библиотеки

## Быстрая сборка

Для сборки и установки библиотеки в режиме разработки:

```bash
cd yandexgptbot_lib/package
python package_builder.py
```

Этот скрипт:

1. Очистит предыдущие сборки (папки `dist` и `yandexgptbot_lib.egg-info`)
2. Соберет пакет (source distribution и wheel)
3. Установит пакет в режиме разработки
4. Протестирует импорт

## Ручная сборка

### Сборка дистрибутива

```bash
cd yandexgptbot_lib/package
python setup.py sdist bdist_wheel
```

### Установка в режиме разработки

```bash
cd yandexgptbot_lib/package
pip install -e .
```

### Установка из собранного пакета

```bash
cd yandexgptbot_lib/package
pip install dist/yandexgptbot_lib-*.whl
```

## Публикация в PyPI

### Подготовка к публикации

1. Обновите версию в `setup.py`
2. Обновите `CHANGELOG.md` (если есть)
3. Соберите пакет: `python setup.py sdist bdist_wheel`

### Тестовая публикация (TestPyPI)

```bash
cd yandexgptbot_lib/package
python -m twine upload --repository testpypi dist/*
```

### Публикация в PyPI

```bash
cd yandexgptbot_lib/package
python -m twine upload dist/*
```

## Структура проекта

```
ChatYandexGPT/
├── yandexgptbot_lib/           # Библиотека и файлы сборки
│   ├── __init__.py             # Инициализация библиотеки
│   ├── bot.py                  # Основной класс бота
│   ├── audio.py                # Аудио компоненты
│   ├── prompts.py              # Коллекция промптов
│   ├── conversation.py         # Управление диалогами
│   ├── config.py               # Конфигурация
│   ├── settings.py             # Настройки
│   ├── cli.py                  # CLI интерфейс
│   ├── README.md               # Документация библиотеки
│   └── package/                # 🆕 Файлы для сборки
│       ├── setup.py            # Конфигурация пакета
│       ├── package_builder.py  # Скрипт сборки
│       ├── MANIFEST.in         # Список файлов для дистрибутива
│       ├── README.md           # Документация по сборке
│       └── README_LIBRARY.md   # Документация библиотеки
├── examples/                   # Примеры использования
└── ...
```

## Примечания

- Скрипт `package_builder.py` очищает только папки `dist` и `yandexgptbot_lib.egg-info`, не затрагивая папку `package`
- Исходный код библиотеки находится в папке `yandexgptbot_lib/` в корне проекта
- После установки библиотека доступна как `yandexgptbot-lib` и CLI команда `yandexgptbot` 
