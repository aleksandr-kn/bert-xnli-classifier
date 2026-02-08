#!/usr/bin/env python
# -*- coding: utf-8 -*-

# В данном скрипте планирую строить брать входной текст, разбивать его на предложения
# и строить отношения при помощи evaluation через модель
# Пока ваще пох как, все хардкодом, потом причешу.

import re
import networkx as nx
import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve,
    auc, classification_report, confusion_matrix
)

models_path = Path(__file__).resolve().parent.parent / "models"
sys.path.append(str(models_path))
# Класс предиктора
from nli_predictor import NLIPredictor  # если вынесен в отдельный файл

# === Константы, по сути это конфиг скрипта ===
MODEL_DIR = "outputs/models/2026-02-04_07-13-03" # Директория с моделью относительно корня
TEST_DATA_PATH = "data/tests/test_graphs_2.csv"
MAX_LEN = 256
NUM_LABELS = 3

LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

def build_sentence_graph(sentences, rel_map):
    G = nx.DiGraph()

    # === Вершины ===
    for i, sent in enumerate(sentences):
        G.add_node(
            i,
            text=sent
        )

    # === Рёбра ===
    for (i, j), info in rel_map.items():
        G.add_edge(
            i,
            j,
            label=LABEL_MAP[info["label"]],
            proba=info["proba"]
        )

    return G

def plot_sentence_graph(G, title=None, label_type="text"):
    print(G)
    """
    G : nx.DiGraph
        Граф предложений. Узлы должны содержать 'text'.
    title : str
        Заголовок графа.
    label_type : str
        'text' — выводить текст предложения,
        'index' — выводить номер узла.
    """
    pos = nx.spring_layout(G, seed=42)

    # Цвета рёбер по типу отношения
    edge_colors = []
    for _, _, d in G.edges(data=True):
        if d["label"] == "contradiction":
            edge_colors.append("red")
        elif d["label"] == "entailment":
            edge_colors.append("green")
        else:
            edge_colors.append("gray")

    plt.figure(figsize=(10, 8))

    # Рисуем узлы и рёбра
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=1500,
        node_color="lightblue",
        edge_color=edge_colors,
        arrows=True
    )

    # Подписи узлов
    if label_type == "text":
        node_labels = {
            n: G.nodes[n]["text"][:50] + ("..." if len(G.nodes[n]["text"]) > 50 else "")
            for n in G.nodes()
        }
    else:  # 'index'
        node_labels = {n: str(n) for n in G.nodes()}

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    # Подписи рёбер — тип отношения
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()

def split_sentences(text: str):
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

def build_sentence_pairs(sentences):
    pairs = []
    indices = []

    n = len(sentences)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sentences[i], sentences[j]))
            indices.append((i, j))

    return pairs, indices

def build_relation_map(indices, preds, probas):
    """
    returns: dict[(i, j)] -> {
        label: int,
        proba: np.array
    }
    """
    rel_map = {}

    for (i, j), label, proba in zip(indices, preds, probas):
        rel_map[(i, j)] = {
            "label": int(label),
            "proba": proba
        }

    return rel_map

def analyze_text_relations(text, predictor):
    sentences = split_sentences(text)

    if len(sentences) < 2:
        return sentences, {}

    pairs, indices = build_sentence_pairs(sentences)

    preds, probas = predictor.predict_batch(pairs)

    rel_map = build_relation_map(indices, preds, probas)

    return sentences, rel_map

def load_dataset(file_path):
    """
    Простая загрузка датасета в pandas.DataFrame.
    Поддерживает parquet, csv, tsv, json, excel (по расширению).
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in [".csv"]:
        return pd.read_csv(path)
    elif ext in [".tsv"]:
        return pd.read_csv(path, sep="\t")
    elif ext in [".json"]:
        return pd.read_json(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")

# Точка входа. Тут вся грязная работа
def main():
    predictor = NLIPredictor(MODEL_DIR, max_len=MAX_LEN)

    test_df = load_dataset(TEST_DATA_PATH)

    # Обходим каждое предложение из текста
    for row_idx, text in enumerate(test_df["text"]):
        print(f"\n=== TEXT {row_idx} ===")

        # Получаем предложения графа и их карту их отношений (neutral, ent., contr-ion.)
        sentences, rel_map = analyze_text_relations(text, predictor)

        g = build_sentence_graph(sentences, rel_map)

        for u, v, data in g.edges(data=True):
            proba = data["proba"]  # np.array([entailment, neutral, contradiction])

            print(
                f"[{u} -> {v}] "
                f"{g.nodes[u]['text'][:30]} ... || "
                f"{g.nodes[v]['text'][:30]} ... "
                f"=> {data['label']} "
                f"(proba: entail={proba[0]:.2f}, neutral={proba[1]:.2f}, contradiction={proba[2]:.2f})"
            )

        # === Визуализация графа ===
        plot_sentence_graph(
            g,
            title=f"Sentence relation graph (text {row_idx})"
        )

if __name__ == "__main__":
    main()
