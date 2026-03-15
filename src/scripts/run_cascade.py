#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Точка входа для запуска self-improving cascade.

Запуск из корня проекта:
    python -m src.scripts.run_cascade
"""

from src.cascade.engine import run_cascade
from src.cascade.visualization import plot_convergence, print_convergence_table

# === Конфигурация ===
INITIAL_MODEL_DIR = "outputs/models/xnli_rubert-base-cased-conversational_2025-12-27_19-07-06"
CORPUS_PATH = "data/tests/test_graphs_4.csv"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_ITERATIONS = 5
FROM_SCRATCH = True


def main():
    results = run_cascade(
        initial_model_dir=INITIAL_MODEL_DIR,
        corpus_path=CORPUS_PATH,
        llm_model_name=LLM_MODEL_NAME,
        max_iterations=MAX_ITERATIONS,
        from_scratch=FROM_SCRATCH,
    )

    # Вывод сводной таблицы
    log_path = "outputs/cascade/cascade_log.json"
    print_convergence_table(log_path)

    # Визуализация конвергенции
    plot_convergence(log_path, save_dir="outputs/cascade/plots")


if __name__ == "__main__":
    main()
