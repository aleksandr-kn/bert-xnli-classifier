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


def compute_hallucination_metrics(G):
    """
    Вычисляет метрики галлюцинаций для двудольного кросс-документного NLI-графа.

    Вершины оригинала должны иметь тип 'source', а вершины пересказа - 'target'.
    """
    summary_nodes = [node for node, data in G.nodes(data=True) if data.get("type") == "target"]
    M = len(summary_nodes)

    if M == 0:
        return {
            "contradiction_ratio": 0.0,
            "unsupported_ratio": 0.0,
            "faithfulness_score": 0.0,
            "coherence_index": 0.0,
            "details": {}
        }

    supported_count = 0
    contradicted_count = 0
    unsupported_count = 0

    details = {}

    for v in summary_nodes:
        in_edges = G.in_edges(v, data=True)
        text_v = G.nodes[v].get("text", "")

        is_supported = False
        is_contradicted = False

        # Собираем распределение вероятностей по входящим ребрам
        entail_probas = []
        contradiction_probas = []

        for u, _, data in in_edges:
            label = data.get("label")
            proba = data.get("proba", [0.0, 1.0, 0.0])

            if label == "entailment":
                is_supported = True
                entail_probas.append(float(proba[0]))
            elif label == "contradiction":
                is_contradicted = True
                contradiction_probas.append(float(proba[2]))

        # Истинное противоречие: если хотя бы одно предложение оригинала противоречит,
        # И ПРИ ЭТОМ ни одно предложение оригинала не подтверждает (not is_supported).
        actual_contradicted = is_contradicted and not is_supported

        if is_supported:
            supported_count += 1
        if actual_contradicted:
            contradicted_count += 1

        is_unsupported = not is_supported and not is_contradicted
        if is_unsupported:
            unsupported_count += 1

        details[v] = {
            "text": text_v,
            "is_supported": is_supported,
            "is_contradicted": is_contradicted,
            "is_unsupported": is_unsupported,
            "max_entail_proba": max(entail_probas) if entail_probas else 0.0,
            "max_contradiction_proba": max(contradiction_probas) if contradiction_probas else 0.0
        }

    contradiction_ratio = contradicted_count / M
    unsupported_ratio = unsupported_count / M
    faithfulness_score = supported_count / M

    # Индекс когерентности: средний перевес макс. вероятности entailment над contradiction
    sum_coherence = 0.0
    for v_data in details.values():
        sum_coherence += (v_data["max_entail_proba"] - v_data["max_contradiction_proba"])
    
    coherence_index = sum_coherence / M if M > 0 else 0.0

    return {
        "contradiction_ratio": contradiction_ratio,
        "unsupported_ratio": unsupported_ratio,
        "faithfulness_score": faithfulness_score,
        "coherence_index": coherence_index,
        "details": details
    }

