#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Анализ противоречий в тексте через графовую центральность.

Для каждого текста из датасета:
1. Строит граф NLI-отношений между предложениями
2. Извлекает подграф противоречий
3. Определяет предложение-ядро через взвешенную степень
4. Выводит отчёт и визуализации

Запуск из корня проекта:
    python -m src.scripts.analyze_contradictions
"""

from src.models.nli_predictor import NLIPredictor
from src.utils.data import load_dataset
from src.graph.builder import graph_from_text
from src.graph.analysis import analyze_contradictions, ContradictionReport
from src.graph.visualization import (
    plot_nli_graph,
    plot_contradiction_subgraph,
    plot_centrality_bar,
)

# === Конфигурация ===
MODEL_DIR = "outputs/models/2026-02-04_07-13-03"
TEST_DATA_PATH = "data/tests/test_graphs_4.csv"
MAX_LEN = 256


def print_report(report: ContradictionReport, text_idx: int = 0):
    """Форматированный вывод отчёта об анализе противоречий."""

    print(f"\n{'='*60}")
    print(f"  ОТЧЁТ О ПРОТИВОРЕЧИЯХ — текст {text_idx}")
    print(f"{'='*60}")

    print(f"\nПредложений в тексте: {len(report.sentences)}")
    for i, s in enumerate(report.sentences):
        print(f"  [{i}] {s}")

    print(f"\nПар с противоречием: {len(report.contradiction_pairs)}")
    for i, j, p in report.contradiction_pairs:
        print(f"  [{i} -> {j}] P_contradiction = {p:.3f}")
        print(f"    premise:    {report.sentences[i]}")
        print(f"    hypothesis: {report.sentences[j]}")

    if report.core_node is not None:
        print(f"\n--- ЯДРО ПРОТИВОРЕЧИЙ ---")
        print(f"  Узел: [{report.core_node}]")
        print(f"  Текст: {report.core_sentence}")
        print(f"  Взвешенная степень: {report.weighted_degree.get(report.core_node, 0):.3f}")
        print(f"  Degree centrality:  {report.degree_centrality.get(report.core_node, 0):.3f}")
    else:
        print(f"\nПротиворечий не обнаружено.")

    if report.weighted_degree:
        print(f"\nВзвешенная степень всех узлов подграфа противоречий:")
        for node in sorted(report.weighted_degree, key=report.weighted_degree.get, reverse=True):
            marker = " <-- ядро" if node == report.core_node else ""
            print(f"  [{node}] {report.weighted_degree[node]:.3f}{marker}")

    print(f"{'='*60}\n")


def main():
    predictor = NLIPredictor(MODEL_DIR, max_len=MAX_LEN)
    test_df = load_dataset(TEST_DATA_PATH)

    for row_idx, text in enumerate(test_df["text"]):
        # 1. Строим граф
        G, sentences = graph_from_text(text, predictor)

        # 2. Анализируем противоречия
        report = analyze_contradictions(G, sentences, strategy="weighted")

        # 3. Выводим отчёт
        print_report(report, text_idx=row_idx)

        # 4. Визуализация: полный граф с выделением ядра
        highlight = {report.core_node} if report.core_node is not None else None
        plot_nli_graph(
            G,
            title=f"Полный граф NLI-отношений (текст {row_idx})",
            label_type="index",
            highlight_nodes=highlight,
        )

        # 5. Визуализация: подграф противоречий
        plot_contradiction_subgraph(
            report.contradiction_subgraph,
            core_node_id=report.core_node,
            title=f"Подграф противоречий (текст {row_idx})",
        )

        # 6. Визуализация: барчарт центральности
        plot_centrality_bar(
            report.weighted_degree,
            sentences,
            title=f"Взвешенная степень в подграфе противоречий (текст {row_idx})",
            core_node_id=report.core_node,
        )


if __name__ == "__main__":
    main()
