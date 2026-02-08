#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Импорты всякой шляпы, может нужно больше хз
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLIPredictor:
    """
        Универсальный класс predictor-а.
        Подаем любую модель сюда, получаем результат.
    """
    def __init__(self, model_dir, max_len = 256, device=None):
        # Ожидается стандартная структура сохранённой модели HuggingFace
        # (результат model.save_pretrained() + tokenizer.save_pretrained()):
        #
        # MODEL_DIR/
        # ├── config.json                  # обязательный
        # ├── model.safetensors            # или pytorch_model.bin
        # ├── tokenizer_config.json
        # ├── special_tokens_map.json
        # ├── tokenizer.json               # ИЛИ vocab.txt / merges.txt / spiece.model и т.п.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        self.max_len = max_len
        # Либо device получаем напрямую, либо берем самый выгодный доступный, если не указали
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval() # Переводим модель в predict режим

    def predict(self, premise, hypothesis):
        enc = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
            proba = F.softmax(logits, dim=1)[0]

        label = torch.argmax(proba).item()
        return label, proba.cpu().numpy()

    def predict_batch(self, pairs, batch_size=32):
        all_preds = []
        all_proba = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]

            enc = self.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = self.model(**enc)
                logits = outputs.logits
                proba = F.softmax(logits, dim=1)

            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_proba.extend(proba.cpu().numpy())

        return np.array(all_preds), np.array(all_proba)