#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Общие метрики для оценки NLI-моделей.
Используется в train_model.py, train_model_ruwanli.py, evaluate.py и каскаде.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize


def compute_metrics(pred, num_labels=3):
    """
    Метрики для HuggingFace Trainer (вызывается каждую эпоху на eval_dataset).

    Args:
        pred: EvalPrediction с полями predictions и label_ids
        num_labels: количество классов (по умолчанию 3)

    Returns:
        dict: accuracy, precision, recall, f1, roc_auc, pr_auc
    """
    y_true = pred.label_ids
    y_pred = np.argmax(pred.predictions, axis=1)

    proba_tensor = F.softmax(torch.tensor(pred.predictions), dim=1)
    y_proba_multi = proba_tensor.numpy()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    y_true_bin = label_binarize(y_true, classes=list(range(num_labels)))
    roc_auc_val = roc_auc_score(y_true_bin, y_proba_multi, multi_class='ovr')

    pr_auc_vals = []
    for i in range(num_labels):
        prec_vals, rec_vals, _ = precision_recall_curve(y_true_bin[:, i], y_proba_multi[:, i])
        pr_auc_vals.append(auc(rec_vals, prec_vals))
    pr_auc_val = np.mean(pr_auc_vals)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
    }
