#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_model_ruwanli.py
Обучение модели RuBERT для задачи NLI на русском (RuWANLI)
"""

from datetime import datetime
import os

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize

# ========== Конфигурация ==========
OUTPUT_DIR = './outputs'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 20
EPOCHS = 12
NUM_LABELS = 3
MAX_LEN = 256
MODEL_NAME = "ai-forever/ruBert-large"  # Можно менять на другую RuBERT-модель

LABEL2ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

# ========== Функция токенизации с добавлением labels ==========
def tokenize_with_labels(df, tokenizer):
    # Конвертируем текстовые метки в числа
    labels = df.label.map(LABEL2ID).values.astype(np.int64)

    enc = tokenizer(
        list(df.premise),
        list(df.hypothesis),
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    )
    enc['labels'] = torch.tensor(labels, dtype=torch.long)
    return enc

# ========== Метрики ==========
def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=1)

    proba_tensor = F.softmax(torch.tensor(pred.predictions), dim=1)
    y_proba_multi = proba_tensor.numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    y_true_bin = label_binarize(y_true, classes=[0,1,2])

    roc_auc_val = roc_auc_score(y_true_bin, y_proba_multi, multi_class='ovr')

    pr_auc_vals = []
    for i in range(3):
        prec_vals, rec_vals, _ = precision_recall_curve(y_true_bin[:, i], y_proba_multi[:, i])
        pr_auc_vals.append(auc(rec_vals, prec_vals))
    pr_auc_val = np.mean(pr_auc_vals)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val
    }

# ========== Основной скрипт ==========
def main():
    # Загружаем RuWANLI
    dataset = load_dataset("deepvk/ru-WANLI")
    train_df = dataset['train'].to_pandas()
    val_df   = dataset['validation'].to_pandas()
    test_df  = dataset['test'].to_pandas()

    # Токенизация
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenize_with_labels(train_df, tokenizer)
    val_enc   = tokenize_with_labels(val_df, tokenizer)
    test_enc  = tokenize_with_labels(test_df, tokenizer)

    # Конвертируем в HF Dataset
    train_dataset = Dataset.from_dict(train_enc)
    val_dataset   = Dataset.from_dict(val_enc)
    test_dataset  = Dataset.from_dict(test_enc)

    # Устанавливаем формат PyTorch
    train_dataset.set_format(type='torch')
    val_dataset.set_format(type='torch')
    test_dataset.set_format(type='torch')

    # Модель
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        hidden_dropout_prob=0.2
    ).to(DEVICE)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=1.5e-5,
        weight_decay=0.05,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to="none",
        fp16=True,
        fp16_opt_level="O1"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Обучение
    trainer.train()

    # Сохраняем модель и токенизатор
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model_name = MODEL_NAME.replace("/", "_")
    save_path = os.path.join(
        OUTPUT_DIR,
        "models",
        f"{safe_model_name}_{timestamp}"
    )
    os.makedirs(save_path, exist_ok=True)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Модель и токенизатор сохранены в {save_path}")

    # Предсказания на тесте
    raw_preds = trainer.predict(test_dataset)
    metrics = compute_metrics(raw_preds)
    print("=== Test metrics ===")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
