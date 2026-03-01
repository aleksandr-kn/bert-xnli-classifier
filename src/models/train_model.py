#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_model.py
Использование трансформеров (BERT, RuBERT) для классификации по датасету XNLI
Попытка решения задачи NLI (определения наличия противоречий в тесте)
"""
from datetime import datetime
import os

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize

# Константы конфигурации
TRAIN_PATH = "data/xnli/ru/train-00000-of-00001.parquet"
TEST_PATH = "data/xnli/ru/test-00000-of-00001.parquet"
VAL_PATH = "data/xnli/ru/validation-00000-of-00001.parquet"
OUTPUT_DIR = './outputs'
DEVICE = torch.device("cuda")  #"cuda", "cpu", "mps"
BATCH_SIZE = 20
EPOCHS = 8
NUM_LABELS = 3
MAX_LEN = 256
MODEL_NAME = "ai-forever/ruBert-large" # 24 слоя, 'LARGE' BERT модель
# MODEL_NAME = "DeepPavlov/rubert-base-cased"
# MODEL_NAME = "DeepPavlov/rubert-base-cased-conversational"
# MODEL_NAME = "DeepPavlov/rubert-base-cased-sentence" # Можно попробовать и эту модель
# MODEL_NAME = "sberbank-ai/sbert_large_nlu_ru" # 24 слоя, 'LARGE' BERT модель

# Функция загрузки данных из датасета xnli
def load_xnli(file_path):
    df = pd.read_parquet(file_path)

    return df

# Функция для токенизации
def tokenize_fn(df, tokenizer):
    return tokenizer(
        list(df.premise),
        list(df.hypothesis),
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    )

# Тут основная логика
def main():
    # Загрузка данных
    train_df = load_xnli(TRAIN_PATH)
    val_df = load_xnli(VAL_PATH)
    test_df = load_xnli(TEST_PATH)

    # Токенизация данных
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_enc = tokenize_fn(train_df, tokenizer)
    val_enc   = tokenize_fn(val_df, tokenizer)
    test_enc  = tokenize_fn(test_df, tokenizer)

    # HuggingFace Datasets
    train_dataset = Dataset.from_dict({**train_enc, "labels": list(train_df.label)})
    val_dataset   = Dataset.from_dict({**val_enc, "labels": list(val_df.label)})
    test_dataset  = Dataset.from_dict({**test_enc, "labels": list(test_df.label)})

    # Модель
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        hidden_dropout_prob = 0.2,
    )

    # Training arguments
    training_args = TrainingArguments(
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=1.5e-5,
        weight_decay=0.05,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        output_dir=OUTPUT_DIR,
        logging_dir='./outputs/logs',
        report_to="none",

        # fp16
        fp16=True,
        fp16_opt_level="O1",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Обучение
    trainer.train()

    # Сохранение модели

    # timestamp вида: 2025-11-29_17-42-10
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(OUTPUT_DIR, 'models')
    save_path = os.path.join(save_path, timestamp)
    os.makedirs(save_path, exist_ok=True)

    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Модель и токенизатор сохранены в {save_path}")

    # Предсказания
    raw_preds = trainer.predict(test_dataset)
    y_test = raw_preds.label_ids
    y_pred = np.argmax(raw_preds.predictions, axis=1)

    # Вероятности классов через softmax
    proba_tensor = F.softmax(torch.tensor(raw_preds.predictions), dim=1)
    y_proba_multi = proba_tensor.numpy()  # shape: [num_samples, num_classes]

    # Метрики для multi-class
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Для ROC AUC и PR AUC делаем one-hot бинаризацию меток
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # shape: [num_samples, 3]

    # ROC AUC
    roc_auc_val = roc_auc_score(y_test_bin, y_proba_multi, multi_class='ovr')

    # PR AUC (среднее по классам)
    pr_auc_vals = []
    for i in range(3):
        prec_vals, rec_vals, _ = precision_recall_curve(y_test_bin[:, i], y_proba_multi[:, i])
        pr_auc_vals.append(auc(rec_vals, prec_vals))
    pr_auc_val = np.mean(pr_auc_vals)

    # Вывод результатов
    print("=== BERT-based model results ===")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("ROC AUC (macro):", roc_auc_val)
    print("PR AUC (macro):", pr_auc_val)

if __name__ == "__main__":
    main()