# 📝 Анализатор тональности русского текста

Этот проект представляет собой веб-приложение на `Streamlit` для анализа тональности текста на русском языке с использованием модели `RuBERT-Tiny`. Также включены тесты на `pytest` и автоматическая CI/CD-проверка через GitHub Actions.

## 🚀 Возможности
- Определение эмоциональной окраски текста (нейтральный, позитивный, негативный)
- Визуализация уверенности модели
- Поддержка GPU (если доступен `CUDA`)
- Автоматические тесты перед пушем в `main`

## 📦 Установка и запуск

### 1. Клонирование репозитория
```sh
git clone https://github.com/your_username/your_project.git
cd your_project
```

### 2. Создание виртуального окружения и установка зависимостей
```sh
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Запуск приложения
```sh
streamlit run app/main.py
```

### 🛠 Тестирование
```sh
pytest --maxfail=1 --disable-warnings -v
```


## ⚙ CI/CD через GitHub Actions
```sh
├── 📂 app/
│   ├── main.py            # Код Streamlit-приложения
│   ├── __init__.py
│   └── ...
├── tests/
│   ├── test_sentiment.py  # Тесты для анализа тональности
│   └── ...
├── .github/workflows/test.yml  # CI/CD конфигурация
├── requirements.txt       # Список зависимостей
├── README.md              # Этот файл
└── ...
```