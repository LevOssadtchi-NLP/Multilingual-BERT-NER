# model.py
import torch.nn as nn
from transformers import BertForTokenClassification
from dataset import label2id, id2label

class NERModel(nn.Module):
    def __init__(self, model_name="bert-base-multilingual-cased"):
        super().__init__()
        self.model = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )

    def forward(self, **kwargs):
        # Передаём все аргументы в BertForTokenClassification
        return self.model(**kwargs)
