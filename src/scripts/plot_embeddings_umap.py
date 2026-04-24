#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UMAP-визуализация эмбеддингов словаря BERT до и после fine-tuning.

Читает CSV из outputs/embeddings/, строит общую UMAP-проекцию
и выводит два облака точек на одном графике для сравнения.

Запуск из корня проекта:
    python -m src.scripts.plot_embeddings_umap
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import umap

# === Конфигурация ===
EMBEDDINGS_DIR = "outputs/embeddings"
BASE_CSV = os.path.join(EMBEDDINGS_DIR, "embeddings_base.csv")
FINETUNED_CSV = os.path.join(EMBEDDINGS_DIR, "embeddings_finetuned.csv")
# Сколько токенов брать для UMAP (все 120k будут считаться долго)
SAMPLE_SIZE = 5000
RANDOM_SEED = 42


def load_embeddings_csv(path):
    """Загружает эмбеддинги из CSV (token, vector)."""
    tokens = []
    vectors = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # пропускаем заголовок
        for row in reader:
            tokens.append(row[0])
            vectors.append(np.fromstring(row[1], sep=",", dtype=np.float32))
    return tokens, np.array(vectors)


def main():
    print("Загрузка эмбеддингов...")
    tokens, base_emb = load_embeddings_csv(BASE_CSV)
    _, ft_emb = load_embeddings_csv(FINETUNED_CSV)

    # Сэмплируем подмножество токенов
    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.choice(len(tokens), size=min(SAMPLE_SIZE, len(tokens)), replace=False)

    base_sample = base_emb[indices]
    ft_sample = ft_emb[indices]

    # Объединяем для единой UMAP-проекции (чтобы пространство было общим)
    combined = np.vstack([base_sample, ft_sample])

    print(f"UMAP-проекция ({len(combined)} точек)...")
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.1)
    projection = reducer.fit_transform(combined)

    base_proj = projection[:len(indices)]
    ft_proj = projection[len(indices):]

    # --- График ---
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(base_proj[:, 0], base_proj[:, 1],
               s=3, alpha=0.4, c="royalblue", label="Base (pre-training)")
    ax.scatter(ft_proj[:, 0], ft_proj[:, 1],
               s=3, alpha=0.4, c="tomato", label="Fine-tuned (XNLI)")

    ax.set_title(f"UMAP: word embeddings до и после fine-tuning ({SAMPLE_SIZE} токенов)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=5)

    plt.tight_layout()
    out_path = os.path.join(EMBEDDINGS_DIR, "umap_base_vs_finetuned.png")
    fig.savefig(out_path, dpi=200)
    print(f"Сохранено: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
