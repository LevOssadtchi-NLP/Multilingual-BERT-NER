import torch
from transformers import BertForTokenClassification
from dataset import NERDataset, label2id, id2label
from model import NERModel
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Загружаем модель
# =========================
model = NERModel()
model = BertForTokenClassification.from_pretrained(
    "/content/drive/MyDrive/Multilingual-BERT-NER/src/outputs/checkpoints/epoch3"
)
model.to(DEVICE)
model.eval()

# =========================
# Датасет и DataLoader
# =========================
test_dataset = NERDataset("data/test.jsonl")
test_loader = DataLoader(test_dataset, batch_size=8)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch_gpu = {k: v.to(DEVICE) for k, v in batch.items() if torch.is_tensor(v)}
        outputs = model(**batch_gpu)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        labels = batch["labels"].tolist()

        # Собираем предсказания и настоящие метки, игнорируя -100
        for p_seq, l_seq in zip(preds, labels):
            p_clean = []
            l_clean = []
            for p_tok, l_tok in zip(p_seq, l_seq):
                if l_tok != -100:
                    p_clean.append(id2label[p_tok])
                    l_clean.append(id2label[l_tok])
            all_preds.append(p_clean)
            all_labels.append(l_clean)

# =========================
# Метрики
# =========================
print(classification_report(all_labels, all_preds))

