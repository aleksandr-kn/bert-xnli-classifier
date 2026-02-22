#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Построение графа отношений между предложениями на основе NLI-предсказаний.

Модуль не зависит от torch — predictor передаётся как аргумент (duck typing).
Достаточно, чтобы predictor имел метод predict_batch(pairs) -> (preds, probas).
"""

import networkx as nx
from src.utils.text import split_sentences

# Маппинг числовых меток → строковые
LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

# Обратный маппинг строковых меток → числовые
LABEL_ID = {v: k for k, v in LABEL_MAP.items()}


def build_sentence_pairs(sentences):
    """
    Формирует все упорядоченные пары (i, j), i != j.
    Возвращает список пар текстов и список пар индексов.
    """
    pairs = []
    indices = []

    n = len(sentences)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append((sentences[i], sentences[j]))
            indices.append((i, j))

    return pairs, indices


def build_relation_map(indices, preds, probas):
    """
    Строит карту отношений между предложениями.

    returns: dict[(i, j)] -> {
        "label": int,
        "proba": np.array  # [P_entailment, P_neutral, P_contradiction]
    }
    """
    rel_map = {}

    for (i, j), label, proba in zip(indices, preds, probas):
        rel_map[(i, j)] = {
            "label": int(label),
            "proba": proba,
        }

    return rel_map


def build_nli_graph(sentences, rel_map):
    """
    Создаёт nx.DiGraph из предложений и карты отношений.

    Вершины: индекс → text (текст предложения).
    Рёбра: (i, j) → label (строковая метка), proba (массив вероятностей).
    """
    G = nx.DiGraph()

    for i, sent in enumerate(sentences):
        G.add_node(i, text=sent)

    for (i, j), info in rel_map.items():
        G.add_edge(
            i, j,
            label=LABEL_MAP[info["label"]],
            proba=info["proba"],
        )

    return G


def graph_from_text(text, predictor):
    """
    Полный пайплайн: текст → (граф, предложения).

    1. Разбивает текст на предложения
    2. Формирует все пары
    3. Получает NLI-предсказания через predictor
    4. Строит и возвращает граф

    Возвращает (nx.DiGraph, list[str]).
    Если предложений < 2, возвращает пустой граф.
    """
    sentences = split_sentences(text)

    if len(sentences) < 2:
        G = nx.DiGraph()
        for i, sent in enumerate(sentences):
            G.add_node(i, text=sent)
        return G, sentences

    pairs, indices = build_sentence_pairs(sentences)
    preds, probas = predictor.predict_batch(pairs)
    rel_map = build_relation_map(indices, preds, probas)
    G = build_nli_graph(sentences, rel_map)

    return G, sentences
