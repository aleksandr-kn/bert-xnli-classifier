#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Оркестрация итеративного цикла self-improving cascade.

Главный цикл: BERT предсказывает -> LLM верифицирует -> собираем hard negatives ->
дообучаем BERT -> повторяем, пока BERT не приблизится к качеству LLM.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime

import torch

from src.models.nli_predictor import NLIPredictor
from src.models.llm_verifier import LLMVerifier
from src.cascade.collector import collect_from_corpus, save_hard_negatives
from src.cascade.trainer import train_iteration


@dataclass
class IterationResult:
    iteration: int
    model_path: str
    metrics: dict
    hard_negatives_path: str
    corpus_stats: dict          # total_pairs, candidates, overrides
    dataset_size: int
    timestamp: str


def run_cascade(initial_model_dir, corpus_path, llm_model_name,
                max_iterations=5, convergence_threshold=0.001,
                from_scratch=True, target_labels=None,
                text_column="text"):
    """
    Главный цикл self-improving cascade с расширенным логированием.

    1. Создает уникальную папку эксперимента для изоляции результатов.
    2. Загружает LLM-верификатор (один раз на весь каскад).
    3. На каждой итерации:
       a. Создает NLIPredictor из текущей модели.
       b. Собирает hard negatives из корпуса (BERT предсказывает -> LLM верифицирует).
       c. Сохраняет HN в CSV для удобного анализа.
       d. Если overrides == 0 (нет расхождений) -> конвергенция.
       e. Обучает BERT на расширенном датасете (XNLI + накопленные HN).
       f. Проверяет delta F1 < threshold для остановки.
    4. Сохраняет JSON-лог и генерирует итоговый Markdown-отчет (REPORT.md).

    Args:
        initial_model_dir: путь к начальной модели BERT.
        corpus_path: путь к корпусу текстов.
        llm_model_name: имя LLM для верификации (например, Qwen/Qwen2.5-7B-Instruct).
        max_iterations: максимум итераций.
        convergence_threshold: порог delta F1 для остановки.
        from_scratch: True = обучение с нуля каждую итерацию (рекомендуется),
                      False = continual fine-tuning от предыдущей итерации.
        target_labels: метки для фильтрации кандидатов (по умолчанию [2] - contradiction).
        text_column: имя колонки с текстом в корпусе.

    Returns:
        list[IterationResult]: результаты каждой итерации.
    """
    # Создаем уникальную папку для запуска эксперимента
    timestamp_run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("outputs", "cascade", f"run_{timestamp_run}")
    hn_dir = os.path.join(run_dir, "hard_negatives")
    os.makedirs(hn_dir, exist_ok=True)

    config = {
        "run_id": timestamp_run,
        "initial_model_dir": initial_model_dir,
        "corpus_path": corpus_path,
        "llm_model_name": llm_model_name,
        "max_iterations": max_iterations,
        "convergence_threshold": convergence_threshold,
        "from_scratch": from_scratch,
        "target_labels": target_labels,
    }

    results = []
    current_model_dir = initial_model_dir
    all_hn_paths = []
    stop_reason = "max_iterations"

    # Загружаем LLM-верификатор один раз
    print(f"\nЗагрузка LLM-верификатора: {llm_model_name} ...")
    verifier = LLMVerifier(llm_model_name)
    print("LLM-верификатор загружен.\n")

    prev_f1 = None

    for iteration in range(max_iterations):
        print(f"\n{'#'*60}")
        print(f"  КАСКАД - ИТЕРАЦИЯ {iteration}")
        print(f"{'#'*60}")
        print(f"  Модель: {current_model_dir}")

        # a. Создаем BERT predictor
        print("\nЗагрузка NLI-предиктора...")
        predictor = NLIPredictor(current_model_dir)

        # b. Сбор hard negatives
        print(f"\nСбор hard negatives из {corpus_path}...")
        hard_negatives, corpus_stats = collect_from_corpus(
            corpus_path, predictor, verifier,
            text_column=text_column, target_labels=target_labels,
        )

        # Освобождаем predictor (GPU)
        del predictor
        torch.cuda.empty_cache()

        # c. Сохраняем hard negatives
        hn_path = os.path.join(hn_dir, f"iter_{iteration}.csv")
        save_hard_negatives(hard_negatives, hn_path)
        all_hn_paths.append(hn_path)

        # d. Проверка конвергенции: overrides == 0
        total_overrides = corpus_stats["total_overrides"]
        if total_overrides == 0:
            print(f"\nКонвергенция: LLM не нашла расхождений с BERT.")
            stop_reason = "no_overrides"
            
            # Сохраняем финальный результат
            result = IterationResult(
                iteration=iteration,
                model_path=current_model_dir,
                metrics=results[-1].metrics if results else {},
                hard_negatives_path=hn_path,
                corpus_stats=_clean_stats(corpus_stats),
                dataset_size=results[-1].dataset_size if results else 0,
                timestamp=datetime.now().isoformat(),
            )
            results.append(result)
            break

        # e. Выгружаем LLM перед обучением для освобождения GPU
        print("\nВыгрузка LLM для обучения...")
        del verifier
        torch.cuda.empty_cache()

        # f. Обучение BERT
        # Передаем run_dir чтобы trainer сохранял модели внутри папки эксперимента
        from_checkpoint = None if from_scratch else current_model_dir
        train_result = train_iteration(
            iteration_num=iteration,
            hard_negative_paths=all_hn_paths,
            from_checkpoint=from_checkpoint,
        )

        # g. Записываем результат итерации
        result = IterationResult(
            iteration=iteration,
            model_path=train_result["model_path"],
            metrics=train_result["metrics"],
            hard_negatives_path=hn_path,
            corpus_stats=_clean_stats(corpus_stats),
            dataset_size=train_result["dataset_size"],
            timestamp=datetime.now().isoformat(),
        )
        results.append(result)

        # h. Проверка конвергенции: delta F1
        current_f1 = train_result["metrics"].get("f1", 0)
        if prev_f1 is not None:
            delta_f1 = abs(current_f1 - prev_f1)
            print(f"\nDelta F1: {delta_f1:.6f} (порог: {convergence_threshold})")
            if delta_f1 < convergence_threshold:
                print(f"Конвергенция: delta F1 < {convergence_threshold}")
                stop_reason = "convergence_threshold"
                break

        prev_f1 = current_f1
        current_model_dir = train_result["model_path"]

        # Перезагружаем LLM для следующей итерации
        if iteration < max_iterations - 1:
            print(f"\nПерезагрузка LLM-верификатора...")
            verifier = LLMVerifier(llm_model_name)
            print("LLM-верификатор загружен.")

    # Сохраняем логи и генерируем отчет
    config["stop_reason"] = stop_reason
    log_path = os.path.join(run_dir, "cascade_log.json")
    save_cascade_log(results, config, log_path)
    
    # Генерация Markdown-отчета
    generate_markdown_report(results, config, run_dir)

    print(f"\n{'='*60}")
    print(f"  КАСКАД ЗАВЕРШЁН")
    print(f"  Эксперимент: {run_dir}")
    print(f"  Итераций: {len(results)}")
    print(f"  Причина остановки: {stop_reason}")
    print(f"  Отчет: {os.path.join(run_dir, 'REPORT.md')}")
    print(f"{'='*60}")

    return results


def generate_markdown_report(results, config, run_dir):
    """Генерирует подробный отчет об эксперименте в формате Markdown."""
    report_path = os.path.join(run_dir, "REPORT.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Отчет по эксперименту Self-Improving Cascade\n\n")
        f.write(f"**Дата запуска:** {config['run_id']}\n")
        f.write(f"**Причина остановки:** {config['stop_reason']}\n\n")
        
        f.write("## Конфигурация\n")
        f.write(f"- **Начальная модель:** `{config['initial_model_dir']}`\n")
        f.write(f"- **Корпус:** `{config['corpus_path']}`\n")
        f.write(f"- **LLM Верификатор:** `{config['llm_model_name']}`\n")
        f.write(f"- **Макс. итераций:** {config['max_iterations']}\n")
        f.write(f"- **Порог конвергенции:** {config['convergence_threshold']}\n")
        f.write(f"- **Обучение с нуля:** {config['from_scratch']}\n\n")
        
        f.write("## Сводная таблица результатов\n\n")
        f.write("| Iter | F1 | Acc | Prec | Rec | Overrides | Dataset |\n")
        f.write("|------|----|-----|------|-----|-----------|---------|\n")
        
        for r in results:
            m = r.metrics
            cs = r.corpus_stats
            f.write(f"| {r.iteration} | {m.get('f1', 0):.4f} | {m.get('accuracy', 0):.4f} | "
                    f"{m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} | "
                    f"{cs.get('total_overrides', 0)} | {r.dataset_size} |\n")
        
        f.write("\n## Детализация по итерациям\n")
        for r in results:
            f.write(f"\n### Итерация {r.iteration}\n")
            f.write(f"- **Путь к модели:** `{r.model_path}`\n")
            f.write(f"- **Hard Negatives:** `{r.hard_negatives_path}`\n")
            f.write("- **Статистика корпуса:**\n")
            for k, v in r.corpus_stats.items():
                f.write(f"  - {k}: {v}\n")
    
    print(f"Markdown-отчет создан: {report_path}")



def _clean_stats(corpus_stats):
    """Убирает per_text из статистики для JSON-сериализации."""
    return {k: v for k, v in corpus_stats.items() if k != "per_text"}


def save_cascade_log(results, config, output_path):
    """
    Сохраняет JSON-лог каскада.

    Содержит конфигурацию, результаты всех итераций с метриками,
    причину остановки.
    """
    log = {
        "config": config,
        "iterations": [asdict(r) for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)

    print(f"Лог каскада сохранён: {output_path}")


def load_cascade_log(log_path):
    """Загрузить лог каскада для анализа/визуализации."""
    with open(log_path, "r", encoding="utf-8") as f:
        return json.load(f)
