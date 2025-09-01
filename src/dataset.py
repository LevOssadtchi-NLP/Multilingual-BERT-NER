import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import torch

# -----------------------------
# Настройки
# -----------------------------
DATA_DIR = Path("data/train")
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LEN = 128  # максимальная длина токенов
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

# Расшифровка меток NER (пример)
id2label = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-BIO": 9,
    "I-BIO": 10,
    "B-CEL": 11,
    "I-CEL": 12,
    "B-DIS": 13,
    "I-DIS": 14,
    "B-EVE": 15,
    "I-EVE": 16,
    "B-FOOD": 17,
    "I-FOOD": 18,
    "B-INST": 19,
    "I-INST": 20,
    "B-MEDIA": 21,
    "I-MEDIA": 22,
    "B-MYTH": 23,
    "I-MYTH": 24,
    "B-PLANT": 25,
    "I-PLANT": 26,
    "B-TIME": 27,
    "I-TIME": 28,
    "B-VEHI": 29,
    "I-VEHI": 30
}

label2id = {v: k for k, v in id2label.items()}

# -----------------------------
# Dataset
# -----------------------------
class NERDataset(Dataset):
    def __init__(self, file_path):
        self.examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.examples.append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        tokens = item["tokens"]
        labels = item["ner_tags"]

        # Токенизация с сохранением выравнивания меток
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        # Выравнивание меток под токены
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # игнорировать при вычислении loss
            else:
                label_ids.append(labels[word_idx])

        encoding["labels"] = torch.tensor(label_ids)  # <- важно
        # Преобразуем тензоры к 1D
        encoding = {k: v.squeeze() if isinstance(v, torch.Tensor) else v for k, v in encoding.items()}
        return encoding


# -----------------------------
# Функция создания DataLoader
# -----------------------------
def get_dataloader(file_path, batch_size=16, shuffle=True):
    dataset = NERDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
# -----------------------------
# Пример использования
# -----------------------------
if __name__ == "__main__":
    train_loader = get_dataloader(DATA_DIR / "train_de.jsonl", batch_size=8)
    batch = next(iter(train_loader))
    print(batch.keys())
    print(batch["input_ids"].shape, batch["labels"].shape)
