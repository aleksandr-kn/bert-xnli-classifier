#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Построение графа отношений между предложениями на основе NLI-предсказаний.

Модуль не зависит от torch — predictor передаётся как аргумент (duck typing).
Достаточно, чтобы predictor имел метод predict_batch(pairs) -> (preds, probas).
"""

import numpy as np
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


def build_sentence_pairs(sentences, bidirectional=False):
    """
    Формирует пары предложений для NLI-предсказаний.

    Args:
        sentences: список предложений
        bidirectional: если False (по умолчанию) — только пары (i, j) где i < j,
                       если True — все пары (i, j) и (j, i), i != j

    Returns:
        (pairs, indices): список пар текстов и список пар индексов.
    """
    pairs = []
    indices = []

    n = len(sentences)
    for i in range(n):
        # bidirectional=False: j > i (только вперёд, n*(n-1)/2 пар)
        # bidirectional=True:  j != i (в обе стороны, n*(n-1) пар)
        for j in range(n) if bidirectional else range(i + 1, n):
            if bidirectional and i == j:
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


def select_candidates(rel_map):
    """
    Отбирает пары для LLM-верификации.

    Критерий: label == 2 (BERT предсказал contradiction).

    Возвращает список индексов [(i, j), ...].
    """
    candidates = []
    for (i, j), info in rel_map.items():
        if info["label"] == 2:
            candidates.append((i, j))
    return candidates


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
        edge_attrs = {
            "label": LABEL_MAP[info["label"]],
            "proba": info["proba"],
        }
        if info.get("llm_verified"):
            edge_attrs["llm_verified"] = True
            edge_attrs["llm_reasoning"] = info.get("llm_reasoning", "")
        G.add_edge(i, j, **edge_attrs)

    return G


def graph_from_text(text, predictor, verifier=None, bidirectional=False):
    """
    Полный пайплайн: текст -> (граф, предложения).

    1. Разбивает текст на предложения
    2. Формирует все пары
    3. Получает NLI-предсказания через predictor
    4. (опционально) Верифицирует contradiction-пары через LLM-verifier
    5. Строит и возвращает граф

    Если verifier передан, отбирает пары, которые BERT пометил как
    contradiction, и проверяет их через LLM.

    Возвращает (nx.DiGraph, list[str]).
    Если предложений < 2, возвращает пустой граф.
    """
    sentences = split_sentences(text)

    if len(sentences) < 2:
        G = nx.DiGraph()
        for i, sent in enumerate(sentences):
            G.add_node(i, text=sent)
        return G, sentences

    pairs, indices = build_sentence_pairs(sentences, bidirectional=bidirectional)
    preds, probas = predictor.predict_batch(pairs)
    rel_map = build_relation_map(indices, preds, probas)

    # LLM-верификация кандидатов
    if verifier is not None:
        cand_indices = select_candidates(rel_map)
        print(f"LLM-верификация: {len(cand_indices)} кандидатов из {len(rel_map)} пар")

        if cand_indices:
            cand_data = []
            for i, j in cand_indices:
                info = rel_map[(i, j)]
                cand_data.append({
                    "premise": sentences[i],
                    "hypothesis": sentences[j],
                    "bert_label": info["label"],
                    "bert_proba": info["proba"],
                })

            results = verifier.verify_batch(cand_data, context=text)

            for (i, j), (new_label, confidence, reasoning) in zip(cand_indices, results):
                rel_map[(i, j)]["label"] = new_label
                rel_map[(i, j)]["llm_verified"] = True
                rel_map[(i, j)]["llm_reasoning"] = reasoning

    G = build_nli_graph(sentences, rel_map)

    return G, sentences
