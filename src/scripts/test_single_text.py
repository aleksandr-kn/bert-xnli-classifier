#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from src.models.nli_predictor import NLIPredictor
from src.graph.builder import graph_from_text
from src.graph.analysis import analyze_contradictions
from src.scripts.analyze_contradictions import print_report
from src.graph.visualization import plot_contradiction_subgraph

MODEL_DIR = "outputs/models/xnli_rubert-base-cased-conversational_2025-12-27_19-07-06"
LLM_MODEL_NAME = None

CUSTOM_TEXT = """
Сумма углов треугольника равна 180 градусам.
В любом треугольника сумма его углов всегда будет равна не более 120 градусов.
"""

def main(verbose=False):
    predictor = NLIPredictor(MODEL_DIR, max_len=256)

    verifier = None
    if LLM_MODEL_NAME:
        from src.models.llm_verifier import LLMVerifier
        verifier = LLMVerifier(LLM_MODEL_NAME)

    G, sentences = graph_from_text(CUSTOM_TEXT, predictor, verifier=verifier)
    
    if verbose:
        print("\n--- Все отношения между предложениями ---")
        for u, v, data in G.edges(data=True):
            proba = data["proba"]
            print(
                f"[{u} -> {v}] "
                f"{G.nodes[u]['text'][:30]} ... || "
                f"{G.nodes[v]['text'][:30]} ... "
                f"=> {data['label']} "
                f"(proba: entail={proba[0]:.2f}, neutral={proba[1]:.2f}, contradiction={proba[2]:.2f})"
            )
        print("------------------------------------------\n")

    report = analyze_contradictions(G, sentences, strategy="weighted")
    print_report(report, text_idx=1, graph=G)

    if report.contradiction_pairs:
        plot_contradiction_subgraph(
            report.contradiction_subgraph,
            core_node_id=report.core_node,
            title="Найденные противоречия в тексте"
        )
    else:
        print("Противоречий нет.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NLI cascade on a single text.")
    parser.add_argument("--verbose", action="store_true", help="Вывести все взаимоотношения предложений")
    args = parser.parse_args()
    main(verbose=args.verbose)
