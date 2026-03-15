#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Визуализация результатов self-improving cascade.
Графики конвергенции для включения в диссертацию.
"""

import os

import matplotlib.pyplot as plt

from src.cascade.engine import load_cascade_log


def _load_log(log_path_or_dict):
    """Загружает лог, если передан путь; иначе использует dict напрямую."""
    if isinstance(log_path_or_dict, str):
        return load_cascade_log(log_path_or_dict)
    return log_path_or_dict


def plot_convergence(log_path_or_dict, save_dir=None):
    """
    4 subplot-а: F1 по итерациям, LLM overrides, LLM calls, размер датасета.

    Args:
        log_path_or_dict: путь к cascade_log.json или dict
        save_dir: директория для сохранения (None = plt.show())
    """
    log = _load_log(log_path_or_dict)
    iterations = log["iterations"]

    iters = [it["iteration"] for it in iterations]
    f1_vals = [it["metrics"].get("f1", 0) for it in iterations]
    overrides = [it["corpus_stats"].get("total_overrides", 0) for it in iterations]
    llm_calls = [it["corpus_stats"].get("total_llm_calls", 0) for it in iterations]
    dataset_sizes = [it.get("dataset_size", 0) for it in iterations]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Self-Improving Cascade - конвергенция", fontsize=14)

    # F1
    ax = axes[0, 0]
    ax.plot(iters, f1_vals, "o-", color="tab:blue", linewidth=2, markersize=8)
    ax.set_xlabel("Итерация")
    ax.set_ylabel("F1 (macro)")
    ax.set_title("F1 по итерациям")
    ax.grid(True, alpha=0.3)
    for i, v in zip(iters, f1_vals):
        ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # LLM overrides
    ax = axes[0, 1]
    ax.bar(iters, overrides, color="tab:orange", alpha=0.8)
    ax.set_xlabel("Итерация")
    ax.set_ylabel("LLM overrides")
    ax.set_title("Расхождения BERT vs LLM")
    ax.grid(True, alpha=0.3, axis="y")

    # LLM calls
    ax = axes[1, 0]
    ax.bar(iters, llm_calls, color="tab:green", alpha=0.8)
    ax.set_xlabel("Итерация")
    ax.set_ylabel("LLM вызовов")
    ax.set_title("Количество LLM-верификаций")
    ax.grid(True, alpha=0.3, axis="y")

    # Dataset size
    ax = axes[1, 1]
    ax.plot(iters, dataset_sizes, "s-", color="tab:red", linewidth=2, markersize=8)
    ax.set_xlabel("Итерация")
    ax.set_ylabel("Размер датасета")
    ax.set_title("Размер обучающего набора")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "convergence.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {path}")
    else:
        plt.show()
    plt.close()


def plot_metric_comparison(log_paths, labels, metric="f1", save_dir=None):
    """
    Сравнение нескольких прогонов каскада.

    Args:
        log_paths: список путей к cascade_log.json
        labels: список названий для легенды
        metric: метрика для сравнения (по умолчанию "f1")
        save_dir: директория для сохранения (None = plt.show())
    """
    plt.figure(figsize=(8, 5))

    for log_path, label in zip(log_paths, labels):
        log = _load_log(log_path)
        iterations = log["iterations"]
        iters = [it["iteration"] for it in iterations]
        vals = [it["metrics"].get(metric, 0) for it in iterations]
        plt.plot(iters, vals, "o-", label=label, linewidth=2, markersize=8)

    plt.xlabel("Итерация")
    plt.ylabel(metric.upper())
    plt.title(f"Сравнение прогонов каскада ({metric})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"comparison_{metric}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {path}")
    else:
        plt.show()
    plt.close()


def print_convergence_table(log_path_or_dict):
    """
    Текстовая таблица для включения в текст диссертации.

    Печатает итерации с ключевыми метриками и статистикой.
    """
    log = _load_log(log_path_or_dict)
    iterations = log["iterations"]
    config = log.get("config", {})

    print(f"\n{'='*80}")
    print(f"  Self-Improving Cascade - сводная таблица")
    print(f"  Причина остановки: {config.get('stop_reason', 'N/A')}")
    print(f"{'='*80}")

    header = (
        f"{'Iter':>4} | {'F1':>7} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | "
        f"{'ROC AUC':>7} | {'Overrides':>9} | {'Dataset':>7}"
    )
    print(header)
    print("-" * len(header))

    for it in iterations:
        m = it.get("metrics", {})
        cs = it.get("corpus_stats", {})
        print(
            f"{it['iteration']:>4} | "
            f"{m.get('f1', 0):>7.4f} | "
            f"{m.get('accuracy', 0):>7.4f} | "
            f"{m.get('precision', 0):>7.4f} | "
            f"{m.get('recall', 0):>7.4f} | "
            f"{m.get('roc_auc', 0):>7.4f} | "
            f"{cs.get('total_overrides', 0):>9} | "
            f"{it.get('dataset_size', 0):>7}"
        )

    print(f"{'='*80}\n")
