#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# ВАЖНО: Устанавливаем кеш до импорта любых библиотек transformers/huggingface
os.environ["HF_HOME"] = "F:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "F:/huggingface_cache"

"""
Точка входа для запуска self-improving cascade.

Запуск из корня проекта:
    python -m src.scripts.run_cascade
"""

import os
from src.cascade.engine import run_cascade
from src.cascade.visualization import plot_convergence, print_convergence_table

# === Конфигурация ===
INITIAL_MODEL_DIR = "outputs/models/xnli_rubert-base-cased-conversational_2025-12-27_19-07-06"
CORPUS_PATH = "data/ruwanli_subset_test.csv"  # Подмножество RuWANLI
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_ITERATIONS = 3  # Для теста достаточно 3 итераций
FROM_SCRATCH = True


def main():
    # Запуск каскада
    results, run_dir = run_cascade(
        initial_model_dir=INITIAL_MODEL_DIR,
        corpus_path=CORPUS_PATH,
        llm_model_name=LLM_MODEL_NAME,
        max_iterations=MAX_ITERATIONS,
        from_scratch=FROM_SCRATCH,
    )

    # Путь к логу в папке конкретного эксперимента
    log_path = os.path.join(run_dir, "cascade_log.json")
    
    # Вывод сводной таблицы в консоль
    print_convergence_table(log_path)

    # Визуализация конвергенции (сохраняем графики в папку эксперимента)
    plots_dir = os.path.join(run_dir, "plots")
    plot_convergence(log_path, save_dir=plots_dir)
    
    print(f"\nЭксперимент завершен. Все результаты в: {run_dir}")


if __name__ == "__main__":
    main()
