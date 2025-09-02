# Multilingual Named Entity Recognition with BERT

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.55.4-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Репозиторий содержит реализацию модели для распознавания именованных сущностей (Named Entity Recognition, NER) на нескольких языках, основанную на предобученной многоязычной модели `bert-base-multilingual-cased` от Hugging Face.

Модель обучена распознавать 15 типов сущностей (таких как PER, ORG, LOC, DIS, FOOD и др.) с помощью архитектуры `BertForTokenClassification`.

## Особенности

*   **Многоязычность:** Основана на `bert-base-multilingual-cased`, что позволяет работать с текстами на множестве языков.
*   **Расширенный набор сущностей:** Поддерживает 30 меток (B-I схема для 15 классов), включая специфические типы, такие как `ANIM` (животные), `BIO` (биологические объекты), `DIS` (болезни), `FOOD` (еда) и другие.
*   **Полный пайплайн:** Код включает в себя все этапы: предобработку данных, обучение модели, инференс и оценку качества.
*   **Эффективность:** Использует библиотеки PyTorch и Transformers для быстрого обучения и предсказаний с поддержкой GPU (CUDA).

## Поддерживаемые классы сущностей

Модель обучена предсказывавать следующие типы сущностей:
*   **PER** - Person (Личности)
*   **ORG** - Organization (Организации)
*   **LOC** - Location (Локации)
*   **ANIM** - Animal (Животные)
*   **BIO** - Biological (Биологические объекты)
*   **CEL** - Celestial (Небесные тела)
*   **DIS** - Disease (Болезни)
*   **EVE** - Event (События)
*   **FOOD** - Food (Еда)
*   **INST** - Instrument (Инструменты)
*   **MEDIA** - Media (Медиа)
*   **MYTH** - Mythological (Мифологические объекты)
*   **PLANT** - Plant (Растения)
*   **TIME** - Time (Временные отметки)
*   **VEHI** - Vehicle (Транспортные средства)

Каждый класс использует схему разметки BIO (Begin, Inside, Outside).

## 📁 Структура репозитория

```
Multilingual-BERT-NER/
├── src/                   # Исходный код
│   ├── dataset.py         # Класс Dataset и DataLoader для обработки данных
│   ├── model.py           # Архитектура модели на основе BertForTokenClassification
│   ├── train.py           # Скрипт для обучения модели
│   ├── eval.py            # Скрипт для оценки модели на test set
│   ├── training.ipynb     # Jupyter notebook с примером запуска обучения
│   └── outputs/           # Результаты работы модели
│       └── checkpoints/   # Сохраненные веса после каждой эпохи    
├── data/                  # Данные (не включены в репозиторий)
│   ├── train/             # Тренировочные данные
│   ├── val/               # Валидационные данные
│   └── test.jsonl         # Тестовые данные
├── requirements.txt       # Зависимости проекта
└── README.md              # Этот файл
```

## Установка и настройка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/LevOssadtchi-NLP/Multilingual-BERT-NER.git
    cd Multilingual-BERT-NER
    ```

2.  **Создайте виртуальное окружение и установите зависимости:**
    Рекомендуется использовать Python 3.8 или выше.
    ```bash
    pip install -r requirements.txt
    ```

## Использование

### Обучение модели

Для обучения модели выполните скрипт `train.py`. Перед запуском убедитесь, что ваши данные лежат в директории `data/` в соответствующем формате (JSONL, как в `dataset.py`).

```bash
cd src
python train.py
```
Скрипт автоматически сохранит чекпоинты модели после каждой эпохи в папку `outputs/checkpoints/`.

### Оценка модели

Для запуска инференса и получения метрик (precision, recall, f1-score) на тестовом наборе данных используйте скрипт `eval.py`. Не забудьте указать в коде правильный путь к чекпоинту.

```bash
python eval.py
```

### Формат данных

Данные должны быть в формате `.jsonl`, где каждая строка — JSON-объект с двумя полями:
```json
{"tokens": ["Word1", "Word2", ...], "ner_tags": [0, 1, 2, ...]}
```
*   `tokens`: список слов (токенов) в предложении.
*   `ner_tags`: список числовых идентификаторов меток для каждого токена. Соответствие между метками и ID задается в `dataset.py` в словаре `label2id`.

## Результаты

Модель демонстрирует высокое качество на основном классе сущностей (PER, ORG, LOC), а также неплохие результаты на более специфических классах.

Пример результатов на тестовой выборке (F1-score):
| Entity | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **LOC** | 0.99 | 0.99 | 0.99 |
| **ORG** | 0.97 | 0.97 | 0.97 |
| **PER** | 0.99 | 0.99 | 0.99 |
| **DIS** | 0.80 | 0.80 | 0.80 |
| **FOOD** | 0.67 | 0.62 | 0.64 |
| **MICRO AVG** | 0.92 | 0.93 | 0.93 |

Полный отчет можно получить, запустив `eval.py`.
