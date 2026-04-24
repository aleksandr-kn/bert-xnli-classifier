#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Извлечение эмбеддингов словаря BERT до и после fine-tuning.

Загружает оригинальную модель с HuggingFace и дообученную локальную модель,
извлекает матрицы word_embeddings (vocab_size, hidden_size) и сохраняет
в CSV (token, vector) вместе со статистикой сдвигов.

Запуск из корня проекта:
    python -m src.scripts.extract_embeddings
"""

import csv
import os

import numpy as np
from transformers import AutoModel, AutoTokenizer

# === Конфигурация ===
BASE_MODEL_NAME = "ai-forever/ruBert-large"
FINETUNED_MODEL_DIR = "outputs/models/xnli_ai-forever_ruBert-large_2026-02-09_00-21-03"
OUTPUT_DIR = "outputs/embeddings"
HF_CACHE_DIR = "F:/hf_cache"


def extract_embeddings(model_name_or_path, cache_dir=None):
    """Извлекает матрицу word_embeddings из BERT-модели."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    model.eval()

    embeddings = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
    vocab = tokenizer.get_vocab()

    # Упорядочиваем токены по индексу
    tokens = [""] * len(vocab)
    for token, idx in vocab.items():
        tokens[idx] = token

    return embeddings, tokens


def save_embeddings_csv(path, tokens, embeddings):
    """Сохраняет эмбеддинги в CSV: token, vector (через запятую)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "vector"])
        for i, token in enumerate(tokens):
            vec_str = ",".join(f"{v:.6f}" for v in embeddings[i])
            writer.writerow([token, vec_str])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Оригинальная модель (до fine-tuning) ---
    print(f"Загрузка базовой модели: {BASE_MODEL_NAME}")
    base_emb, tokens = extract_embeddings(BASE_MODEL_NAME)
    print(f"  Размер матрицы: {base_emb.shape}")

    base_path = os.path.join(OUTPUT_DIR, "embeddings_base.csv")
    save_embeddings_csv(base_path, tokens, base_emb)

    # --- Дообученная модель (после fine-tuning) ---
    print(f"Загрузка fine-tuned модели: {FINETUNED_MODEL_DIR}")
    ft_emb, _ = extract_embeddings(FINETUNED_MODEL_DIR)
    print(f"  Размер матрицы: {ft_emb.shape}")

    ft_path = os.path.join(OUTPUT_DIR, "embeddings_finetuned.csv")
    save_embeddings_csv(ft_path, tokens, ft_emb)

    print(f"\nСохранено в {OUTPUT_DIR}/:")
    print(f"  embeddings_base.csv      - {base_emb.shape}")
    print(f"  embeddings_finetuned.csv - {ft_emb.shape}")


if __name__ == "__main__":
    main()
