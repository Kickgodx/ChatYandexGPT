import os

from setuptools import setup, find_packages

# Переходим в корневую папку проекта для чтения файлов
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)

# Чтение README.md
readme_path = os.path.join(project_root, "README.md")
with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Безопасное чтение requirements.txt
requirements_path = os.path.join(project_root, "requirements.txt")
try:
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith('#')]
except UnicodeDecodeError:
    # Если UTF-8 не работает, пробуем другие кодировки
    try:
        with open(requirements_path, "r", encoding="latin-1") as fh:
            requirements = [line.strip() for line in fh if line.strip() and not line.startswith('#')]
    except:
        # Если ничего не работает, используем базовые зависимости
        requirements = [
            "langchain>=0.3.0",
            "langchain-community>=0.3.0",
            "pyaudio>=0.2.14",
            "vosk>=0.3.45",
            "numpy>=1.26.0",
            "requests>=2.32.0",
            "yandexcloud>=0.319.0"
        ]

setup(
    name="yandexgptbot-lib",
    version="1.0.0",
    author="Den",
    author_email="wheelman4000@gmail.com",
    description="Библиотека для работы с YandexGPT через голосовой и текстовый интерфейс",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kickgodx/ChatYandexGPT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "yandexgptbot=yandexgptbot_lib.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "yandexgptbot_lib": ["*.json", "*.txt"],
    },
)
