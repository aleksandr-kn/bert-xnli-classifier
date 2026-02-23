#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate.py
Оценка сохранённой BERT / RuBERT модели на XNLI (ru).
Метрики:
- Accuracy, Precision, Recall, F1
- ROC AUC (macro)
- PR AUC (macro)
- Классификационный отчёт по классам
- Отдельный анализ класса "contradiction" (метка 2)
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve,
    auc, classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize


# === Константы ===
MODEL_DIR = "outputs/models/2025-11-28_05-42-10"
TEST_PATH = "data/xnli/ru/test-00000-of-00001.parquet"
MAX_LEN = 256
NUM_LABELS = 3

# === Функция: загрузка XNLI из Parquet ===
def load_xnli(file_path):
    return pd.read_parquet(file_path)

# === Функция: токенизация (как в train_model.py) ===
def tokenize_fn(df, tokenizer):
    return tokenizer(
        list(df.premise),
        list(df.hypothesis),
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    )

def main():
    print("=== Загрузка модели и токенизатора ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    # === загрузка тестового датасета ===
    print("=== Загрузка тестовых данных ===")
    test_df = load_xnli(TEST_PATH)
    test_enc = tokenize_fn(test_df, tokenizer)

    test_dataset = Dataset.from_dict({**test_enc, "labels": list(test_df.label)})

    # === Trainer в режиме предсказаний ===
    trainer = Trainer(model=model)

    print("=== Предсказание ===")
    raw_preds = trainer.predict(test_dataset)

    y_test = raw_preds.label_ids
    y_pred = np.argmax(raw_preds.predictions, axis=1)

    # Вероятности
    proba_tensor = F.softmax(torch.tensor(raw_preds.predictions), dim=1)
    y_proba_multi = proba_tensor.numpy()

    # === Общие метрики ===
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    roc_auc_val = roc_auc_score(y_test_bin, y_proba_multi, multi_class='ovr')

    pr_auc_vals = []
    for i in range(NUM_LABELS):
        p_curve, r_curve, _ = precision_recall_curve(y_test_bin[:, i], y_proba_multi[:, i])
        pr_auc_vals.append(auc(r_curve, p_curve))
    pr_auc_val = np.mean(pr_auc_vals)

    # === Вывод результатов ===
    print("\n=== Общие метрики ===")
    print("Accuracy:", acc)
    print("Precision (macro):", prec)
    print("Recall (macro):", rec)
    print("F1 (macro):", f1)
    print("ROC AUC (macro):", roc_auc_val)
    print("PR AUC (macro):", pr_auc_val)

    # === Метрики по классам ===
    print("\n=== По классам ===")
    target_names = ["entailment (0)", "neutral (1)", "contradiction (2)"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # === Детальный анализ contradiction ===
    print("\n=== Подробный анализ класса 2 (contradiction) ===")
    contr_labels_true = (y_test == 2).astype(int)
    contr_labels_pred = (y_pred == 2).astype(int)

    contr_prec = precision_score(contr_labels_true, contr_labels_pred)
    contr_rec = recall_score(contr_labels_true, contr_labels_pred)
    contr_f1 = f1_score(contr_labels_true, contr_labels_pred)
    contr_support = contr_labels_true.sum()

    print(f"Precision: {contr_prec:.4f}")
    print(f"Recall:    {contr_rec:.4f}")
    print(f"F1:        {contr_f1:.4f}")
    print(f"Support:   {contr_support}")

    # === Confusion matrix для contradiction vs остальные ===
    cm = confusion_matrix(contr_labels_true, contr_labels_pred)
    print("\nConfusion Matrix (contradiction vs not-contradiction):")
    print(cm)

    # === Визуализация confusion matrix ===
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['not-contradiction', 'contradiction'],
                yticklabels=['not-contradiction', 'contradiction'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: Contradiction vs Not-Contradiction')
    plt.show()

    # === ROC и PR кривая для contradiction ===
    fpr, tpr, thresholds_roc = roc_auc_score(contr_labels_true, y_proba_multi[:, 2]), None, None
    prec_curve, rec_curve, _ = precision_recall_curve(contr_labels_true, y_proba_multi[:, 2])
    pr_auc_contr = auc(rec_curve, prec_curve)

    # Визуализация PR-кривой
    plt.figure()
    plt.plot(rec_curve, prec_curve, label=f'PR curve (AUC={pr_auc_contr:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR-кривая для contradiction (class 2)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
