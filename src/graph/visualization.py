#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Визуализация графов NLI-отношений и результатов анализа противоречий.
"""

import matplotlib.pyplot as plt
import networkx as nx


# Цвета рёбер по типу отношения
_EDGE_COLORS = {
    "contradiction": "red",
    "entailment": "green",
    "neutral": "gray",
}


def _get_edge_colors(G):
    """Возвращает список цветов рёбер по их меткам."""
    return [_EDGE_COLORS.get(d["label"], "gray") for _, _, d in G.edges(data=True)]


def _truncate(text, max_len=50):
    return text[:max_len] + ("..." if len(text) > max_len else "")


def plot_nli_graph(G, title=None, label_type="text", highlight_nodes=None, save_path=None):
    """
    Визуализация полного графа NLI-отношений.

    G : nx.DiGraph
        Граф предложений. Узлы содержат атрибут 'text'.
    title : str
        Заголовок графика.
    label_type : str
        'text' — текст предложения (обрезанный), 'index' — номер узла.
    highlight_nodes : set[int] | None
        Узлы, которые нужно выделить (например, ядро противоречий).
    save_path : str | None
        Путь для сохранения. Если None — показывает plt.show().
    """
    pos = nx.spring_layout(G, seed=42)

    edge_colors = _get_edge_colors(G)

    # Цвета узлов: выделяем highlight_nodes красным
    node_colors = []
    for n in G.nodes():
        if highlight_nodes and n in highlight_nodes:
            node_colors.append("salmon")
        else:
            node_colors.append("lightblue")

    plt.figure(figsize=(10, 8))

    nx.draw(
        G, pos,
        with_labels=False,
        node_size=1500,
        node_color=node_colors,
        edge_color=edge_colors,
        arrows=True,
    )

    # Подписи узлов
    if label_type == "text":
        node_labels = {n: _truncate(G.nodes[n]["text"]) for n in G.nodes()}
    else:
        node_labels = {n: str(n) for n in G.nodes()}

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    # Подписи рёбер
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    if title:
        plt.title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_contradiction_subgraph(G, core_node_id=None, title=None, save_path=None):
    """
    Визуализация подграфа противоречий с интенсивностью рёбер.

    Толщина и прозрачность рёбер пропорциональны P_contradiction.
    Узел-ядро выделяется красным цветом.
    """
    if len(G.edges()) == 0:
        print("Подграф противоречий пуст — нечего визуализировать.")
        return

    pos = nx.spring_layout(G, seed=42)

    # Цвета узлов
    node_colors = []
    for n in G.nodes():
        if n == core_node_id:
            node_colors.append("red")
        else:
            node_colors.append("lightyellow")

    # Толщина рёбер пропорциональна весу (P_contradiction)
    weights = [d.get("weight", 0.5) for _, _, d in G.edges(data=True)]
    edge_widths = [1.0 + w * 3.0 for w in weights]
    edge_alphas = [0.4 + w * 0.6 for w in weights]

    plt.figure(figsize=(10, 8))

    nx.draw(
        G, pos,
        with_labels=False,
        node_size=1500,
        node_color=node_colors,
        edge_color="red",
        width=edge_widths,
        arrows=True,
        alpha=0.9,
    )

    # Подписи узлов — индексы
    node_labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold")

    # Подписи рёбер — P_contradiction
    edge_labels = {
        (u, v): f"{d.get('weight', 0):.2f}"
        for u, v, d in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    if title:
        plt.title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_centrality_bar(centrality, sentences, title=None, core_node_id=None, save_path=None):
    """
    Столбчатая диаграмма метрики центральности по предложениям.

    centrality : dict[int, float]
        Метрика центральности для каждого узла.
    sentences : list[str]
        Список предложений для подписей.
    core_node_id : int | None
        Индекс ядра противоречий (выделяется красным).
    """
    if not centrality:
        print("Нет данных центральности — нечего визуализировать.")
        return

    nodes = sorted(centrality.keys())
    values = [centrality[n] for n in nodes]
    labels = [f"[{n}] {_truncate(sentences[n], 30)}" for n in nodes]
    colors = ["red" if n == core_node_id else "steelblue" for n in nodes]

    plt.figure(figsize=(10, max(4, len(nodes) * 0.6)))

    bars = plt.barh(labels, values, color=colors)
    plt.xlabel("Взвешенная степень (сумма P_contradiction)")

    if title:
        plt.title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
