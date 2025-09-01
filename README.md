# Multilingual-BERT-NER
Implementation of a multilingual Named Entity Recognition (NER) system using BERT in PyTorch. The project builds a transformer-based model from scratch to identify entities like persons, organizations, locations, and time expressions across multiple languages, highlighting effective sequence labeling and contextual embedding techniques.

```Multilingual-BERT-NER/
├── data/
│   ├── train.jsonl         # тренировочный датасет
│   ├── val.jsonl           # валидационный датасет
│   └── test.jsonl          # тестовый датасет
│
├── src/
│   ├── dataset.py          # загрузка и токенизация данных
│   ├── model.py            # загрузка предобученной BERT и подготовка к NER
│   ├── train.py            # скрипт тренировки / fine-tuning
│   ├── eval.py             # оценка качества (F1, accuracy, per-class)
│   └── utils.py            # вспомогательные функции (метрики, сохранение модели)
│
├── outputs/
│   ├── checkpoints/        # сохраняем модель после обучения
│   └── logs/               # tensorboard / логи обучения
│
├── requirements.txt        # необходимые библиотеки (transformers, datasets, torch, seqeval)
├── README.md               # описание проекта
└── run.sh                  # опциональный скрипт для запуска обучения
```
