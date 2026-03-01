#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Анализ противоречий в графе NLI-отношений.

Ключевая идея: предложение, наиболее «центральное» в подграфе противоречий,
является ядром (источником) противоречий в тексте.
"""

from dataclasses import dataclass, field
import networkx as nx


@dataclass
class ContradictionReport:
    """Результат анализа противоречий в тексте."""
    # Предложения текста
    sentences: list[str]
    # Подграф, содержащий только рёбра противоречий
    contradiction_subgraph: nx.DiGraph
    # Степенная центральность узлов в подграфе противоречий
    degree_centrality: dict[int, float]
    # Betweenness-центральность узлов
    betweenness_centrality: dict[int, float]
    # Взвешенная степень: сумма P_contradiction по инцидентным рёбрам
    weighted_degree: dict[int, float]
    # Индекс предложения-ядра противоречий
    core_node: int | None
    # Текст предложения-ядра
    core_sentence: str | None
    # Пары предложений, находящиеся в противоречии: [(i, j, P_contradiction), ...]
    contradiction_pairs: list[tuple[int, int, float]] = field(default_factory=list)


def extract_subgraph_by_label(G, label="contradiction", use_weight=True):
    """
    Извлекает подграф, содержащий только рёбра с заданной меткой.

    Если use_weight=True, копирует атрибут proba и добавляет weight =
    P_contradiction (индекс 2 в массиве proba).
    """
    sub = nx.DiGraph()

    # Копируем все узлы с атрибутами
    for node, data in G.nodes(data=True):
        sub.add_node(node, **data)

    for u, v, data in G.edges(data=True):
        if data.get("label") == label:
            edge_attrs = {"label": data["label"], "proba": data["proba"]}
            if use_weight:
                p_contr = float(data["proba"][2])
                edge_attrs["weight"] = p_contr
                # distance — инверсия для алгоритмов кратчайших путей
                # (betweenness centrality), где меньший вес = ближе
                edge_attrs["distance"] = 1.0 - p_contr
            sub.add_edge(u, v, **edge_attrs)

    # Удаляем изолированные узлы (без рёбер противоречий)
    isolated = [n for n in sub.nodes() if sub.degree(n) == 0]
    sub.remove_nodes_from(isolated)

    return sub


def compute_centrality_metrics(subgraph):
    """
    Вычисляет степенную и betweenness-центральность для подграфа.

    Возвращает (degree_centrality, betweenness_centrality).
    """
    if len(subgraph) == 0:
        return {}, {}

    degree = nx.degree_centrality(subgraph)
    betweenness = nx.betweenness_centrality(subgraph, weight="distance")

    return degree, betweenness


def compute_weighted_degree(subgraph):
    """
    Для каждого узла считает сумму P_contradiction по всем инцидентным рёбрам
    (входящим + исходящим).
    """
    w_deg = {}

    for node in subgraph.nodes():
        total = 0.0
        # Исходящие рёбра
        for _, _, data in subgraph.out_edges(node, data=True):
            total += data.get("weight", 0.0)
        # Входящие рёбра
        for _, _, data in subgraph.in_edges(node, data=True):
            total += data.get("weight", 0.0)
        w_deg[node] = total

    return w_deg


def find_contradiction_core(degree_centrality, weighted_degree, strategy="weighted"):
    """
    Определяет предложение-ядро противоречий.

    Стратегии:
    - "weighted" (по умолчанию): узел с максимальной взвешенной степенью
    - "degree": узел с максимальной степенной центральностью

    Возвращает индекс узла-ядра или None, если подграф пуст.
    """
    if strategy == "weighted":
        scores = weighted_degree
    elif strategy == "degree":
        scores = degree_centrality
    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}")

    if not scores:
        return None

    return max(scores, key=scores.get)


def analyze_contradictions(G, sentences, strategy="weighted"):
    """
    Полный пайплайн анализа противоречий.

    1. Извлекает подграф противоречий
    2. Вычисляет метрики центральности
    3. Определяет ядро противоречий
    4. Собирает пары противоречащих предложений

    Возвращает ContradictionReport.
    """
    sub = extract_subgraph_by_label(G, label="contradiction", use_weight=True)

    degree, betweenness = compute_centrality_metrics(sub)
    w_deg = compute_weighted_degree(sub)

    core = find_contradiction_core(degree, w_deg, strategy=strategy)
    core_sentence = sentences[core] if core is not None else None

    # Собираем пары противоречий с вероятностями
    pairs = []
    for u, v, data in G.edges(data=True):
        if data.get("label") == "contradiction":
            pairs.append((u, v, float(data["proba"][2])))

    return ContradictionReport(
        sentences=sentences,
        contradiction_subgraph=sub,
        degree_centrality=degree,
        betweenness_centrality=betweenness,
        weighted_degree=w_deg,
        core_node=core,
        core_sentence=core_sentence,
        contradiction_pairs=pairs,
    )
