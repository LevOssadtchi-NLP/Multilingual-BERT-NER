# train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from dataset import NERDataset
from model import NERModel
from tqdm import tqdm

# =========================
# Настройки
# =========================
EPOCHS = 3
BATCH_SIZE = 8
LR = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Датасеты и DataLoader
# =========================
train_dataset = NERDataset("../data/train/train_en.jsonl")
val_dataset   = NERDataset("../data/val/val_en.jsonl")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =========================
# Модель
# =========================
model = NERModel()
model.to(DEVICE)

# =========================
# Оптимизатор и scheduler
# =========================
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps
)

# =========================
# Циклы обучения и валидации
# =========================
def train_epoch(loader):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        optimizer.zero_grad()
        # Перемещаем все тензоры на GPU/CPU
        batch_gpu = {k: v.to(DEVICE) for k, v in batch.items() if torch.is_tensor(v)}
        outputs = model(**batch_gpu)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            batch_gpu = {k: v.to(DEVICE) for k, v in batch.items() if torch.is_tensor(v)}
            outputs = model(**batch_gpu)
            total_loss += outputs.loss.item()
    return total_loss / len(loader)

# =========================
# Тренировочный цикл
# =========================
for epoch in range(EPOCHS):
    train_loss = train_epoch(train_loader)
    val_loss   = eval_epoch(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Сохраняем чекпоинт после каждой эпохи
    model.model.save_pretrained(f"outputs/checkpoints/epoch{epoch+1}")
