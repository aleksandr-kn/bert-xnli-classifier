#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Построение и визуализация графов NLI-отношений между предложениями текста.

Запуск из корня проекта:
    python -m src.scripts.plot_graph
"""

from src.models.nli_predictor import NLIPredictor
from src.utils.data import load_dataset
from src.graph.builder import graph_from_text
from src.graph.visualization import plot_nli_graph

# === Конфигурация ===
MODEL_DIR = "outputs/models/2026-02-04_07-13-03"
TEST_DATA_PATH = "data/tests/test_graphs_3.csv"
MAX_LEN = 256


def main():
    predictor = NLIPredictor(MODEL_DIR, max_len=MAX_LEN)
    test_df = load_dataset(TEST_DATA_PATH)

    for row_idx, text in enumerate(test_df["text"]):
        print(f"\n=== TEXT {row_idx} ===")

        G, sentences = graph_from_text(text, predictor)

        for u, v, data in G.edges(data=True):
            proba = data["proba"]
            print(
                f"[{u} -> {v}] "
                f"{G.nodes[u]['text'][:30]} ... || "
                f"{G.nodes[v]['text'][:30]} ... "
                f"=> {data['label']} "
                f"(proba: entail={proba[0]:.2f}, neutral={proba[1]:.2f}, contradiction={proba[2]:.2f})"
            )

        plot_nli_graph(
            G,
            title=f"Sentence relation graph (text {row_idx})",
        )


if __name__ == "__main__":
    main()
