#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Дообучение BERT на расширенном датасете (оригинальный XNLI + hard negatives).

Повторяет паттерн обучения из train_model.py, но добавляет к тренировочному
набору собранные hard negatives от LLM.
"""

from datetime import datetime
import os

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from src.utils.metrics import compute_metrics
from src.cascade.collector import load_hard_negatives

# Константы (повторяют train_model.py)
BASE_MODEL_NAME = "ai-forever/ruBert-large"
TRAIN_PATH = "data/xnli/ru/train-00000-of-00001.parquet"
VAL_PATH = "data/xnli/ru/validation-00000-of-00001.parquet"
TEST_PATH = "data/xnli/ru/test-00000-of-00001.parquet"
MAX_LEN = 256
NUM_LABELS = 3
BATCH_SIZE = 20
EPOCHS = 8
OUTPUT_DIR = "./outputs"


def build_augmented_dataset(original_train_path, hard_negative_paths):
    """
    Объединяет оригинальный XNLI train с hard negatives.

    Из HN-файлов берет только premise, hypothesis, label (=llm_label).
    Результат: pd.concat + shuffle с фиксированным seed.

    Args:
        original_train_path: путь к оригинальному train parquet
        hard_negative_paths: список путей к parquet-файлам с hard negatives

    Returns:
        pd.DataFrame с колонками premise, hypothesis, label
    """
    original_df = pd.read_parquet(original_train_path)
    original_size = len(original_df)

    # Берем только нужные колонки из оригинала
    train_df = original_df[["premise", "hypothesis", "label"]].copy()

    # Добавляем hard negatives
    hn_total = 0
    for hn_path in hard_negative_paths:
        if not os.path.exists(hn_path):
            print(f"  Файл не найден, пропускаю: {hn_path}")
            continue
        hn_df = load_hard_negatives(hn_path)
        # label в HN-файле - это llm_label (ground truth)
        hn_subset = hn_df[["premise", "hypothesis", "label"]].copy()
        train_df = pd.concat([train_df, hn_subset], ignore_index=True)
        hn_total += len(hn_subset)

    # Shuffle с фиксированным seed для воспроизводимости
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Оригинальный датасет: {original_size}")
    print(f"  Добавлено hard negatives: {hn_total}")
    print(f"  Итого: {len(train_df)}")
    print(f"  Распределение меток: {dict(train_df['label'].value_counts().sort_index())}")

    return train_df


def tokenize_fn(df, tokenizer):
    """Токенизация пар premise/hypothesis."""
    return tokenizer(
        list(df.premise),
        list(df.hypothesis),
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


def train_iteration(iteration_num, hard_negative_paths,
                    base_model_name=BASE_MODEL_NAME,
                    from_checkpoint=None):
    """
    Одна итерация обучения BERT на расширенном датасете.

    По умолчанию обучает с нуля от base_model_name (воспроизводимее,
    не страдает от catastrophic forgetting). Параметр from_checkpoint
    позволяет эксперимент с continual fine-tuning.

    Args:
        iteration_num: номер итерации каскада
        hard_negative_paths: список путей к parquet-файлам с HN
        base_model_name: имя базовой модели HuggingFace
        from_checkpoint: путь к чекпоинту для continual fine-tuning (None = с нуля)

    Returns:
        dict: {"model_path": str, "metrics": dict, "dataset_size": int}
    """
    print(f"\n{'='*60}")
    print(f"  ОБУЧЕНИЕ - итерация {iteration_num}")
    print(f"{'='*60}")

    # 1. Собираем расширенный датасет
    print("\nСборка расширенного датасета:")
    train_df = build_augmented_dataset(TRAIN_PATH, hard_negative_paths)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # 2. Токенизация
    model_name_or_path = from_checkpoint if from_checkpoint else base_model_name
    print(f"\nЗагрузка модели: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        from_checkpoint if from_checkpoint else base_model_name
    )

    train_enc = tokenize_fn(train_df, tokenizer)
    val_enc = tokenize_fn(val_df, tokenizer)
    test_enc = tokenize_fn(test_df, tokenizer)

    # 3. HuggingFace Datasets
    train_dataset = Dataset.from_dict({**train_enc, "labels": list(train_df.label)})
    val_dataset = Dataset.from_dict({**val_enc, "labels": list(val_df.label)})
    test_dataset = Dataset.from_dict({**test_enc, "labels": list(test_df.label)})

    # 4. Модель
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=NUM_LABELS,
        hidden_dropout_prob=0.2,
    )

    # 5. TrainingArguments (идентичны train_model.py)
    training_args = TrainingArguments(
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=1.5e-5,
        weight_decay=0.05,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        output_dir=OUTPUT_DIR,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to="none",
        fp16=True,
        fp16_opt_level="O1",
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 7. Обучение
    trainer.train()

    # 8. Оценка на тесте
    print("\nОценка на тестовом наборе:")
    raw_preds = trainer.predict(test_dataset)
    metrics = compute_metrics(raw_preds)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 9. Сохранение модели
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        OUTPUT_DIR, "models", f"cascade_iter{iteration_num}_{timestamp}"
    )
    os.makedirs(save_path, exist_ok=True)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nМодель сохранена: {save_path}")

    # Освобождаем память
    del trainer
    del model
    torch.cuda.empty_cache()

    return {
        "model_path": save_path,
        "metrics": metrics,
        "dataset_size": len(train_df),
    }
