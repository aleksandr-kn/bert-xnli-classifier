#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт проведения эксперимента по безэталонной детекции галлюцинаций в AI-пересказах.
Строит кросс-документные NLI-графы и вычисляет метрики логической согласованности.
"""

import os
# Перенаправляем кеш Hugging Face на диск F:
os.environ["HF_HOME"] = "F:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "F:/huggingface_cache"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from src.models.nli_predictor import NLIPredictor
from src.models.llm_verifier import LLMVerifier
from src.graph.builder import graph_from_two_texts
from src.graph.analysis import compute_hallucination_metrics
from src.utils.model_registry import get_model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Эксперимент по детекции галлюцинаций через NLI-графы")
    parser.add_argument("--model", type=str, default="rubert-large-xnli",
                        help="Путь к модели или slug из реестра (по умолчанию: rubert-large-xnli)")
    parser.add_argument("--llm", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Имя генеративной модели для пересказов и верификации")
    parser.add_argument("--num_texts", type=int, default=1000,
                        help="Количество текстов для оценки (по умолчанию: 1000)")
    parser.add_argument("--use_verifier", action="store_true",
                        help="Использовать ли LLM-верификатор для перепроверки противоречий от BERT")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Размер батча для BERT NLI Predictor")
    parser.add_argument("--proba_threshold", type=float, default=0.35,
                        help="Мягкий порог вероятности противоречия для BERT фильтра")
    parser.add_argument("--save_dir", type=str, default="outputs/hallucinations",
                        help="Директория для сохранения результатов")
    return parser.parse_args()


def generate_summary(text, verifier, max_new_tokens=150):
    """
    Генерирует краткий пересказ текста (1-3 предложения) с помощью Qwen.
    """
    prompt = f"Напиши краткий пересказ (1-3 предложения) следующего текста. Выведи ТОЛЬКО текст пересказа на русском языке и ничего больше.\n\nТекст:\n{text}"
    messages = [
        {"role": "system", "content": "Ты — профессиональный редактор. Твоя единственная задача — писать сухие, точные и краткие выжимки текстов строго на чистом русском языке. Запрещено использовать другие языки (особенно китайский). Запрещено писать вводные фразы."},
        {"role": "user", "content": prompt}
    ]
    input_text = verifier.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = verifier.tokenizer(input_text, return_tensors="pt").to(verifier.model.device)

    with torch.no_grad():
        output_ids = verifier.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.05,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    summary = verifier.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return summary


def get_or_create_dataset(num_texts, verifier, save_path):
    """
    Загружает Gazeta.ru и генерирует пересказы, если файла с пересказами ещё нет.
    """
    if os.path.exists(save_path):
        print(f"Загрузка ранее сгенерированных пересказов из {save_path}...")
        df = pd.read_csv(save_path)
        # Если в файле меньше строк, чем запрошено, вернем сколько есть
        if len(df) >= num_texts:
            return df.head(num_texts)
        else:
            print(f"В файле только {len(df)} строк. Догенерируем новые...")
            existing_count = len(df)
    else:
        print("Файл с пересказами не найден. Начинаем генерацию...")
        df = pd.DataFrame(columns=["text_idx", "original_text", "human_summary", "ai_summary"])
        existing_count = 0

    print("Загрузка датасета IlyaGusev/gazeta (split='test')...")
    dataset = load_dataset("IlyaGusev/gazeta", split="test")

    records = df.to_dict("records")
    
    # Генерируем пересказы для недостающих строк
    for i in tqdm(range(existing_count, num_texts), desc="Генерация AI-пересказов"):
        if i >= len(dataset):
            break
        item = dataset[i]
        orig_text = item["text"]
        human_sum = item["summary"]

        # Используем verifier (модель Qwen) для генерации пересказа
        ai_sum = generate_summary(orig_text, verifier)

        records.append({
            "text_idx": i,
            "original_text": orig_text,
            "human_summary": human_sum,
            "ai_summary": ai_sum
        })

        # Периодически сохраняем, чтобы не потерять прогресс
        if i % 10 == 0 or i == num_texts - 1:
            pd.DataFrame(records).to_csv(save_path, index=False, encoding="utf-8-sig")

    df_result = pd.DataFrame(records)
    print(f"Пересказы успешно сохранены в {save_path}. Всего: {len(df_result)} строк.")
    return df_result.head(num_texts)


def main():
    args = parse_args()
    
    # 1. Создаем директорию для результатов
    os.makedirs(args.save_dir, exist_ok=True)
    plots_dir = os.path.join(args.save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Путь для сохранения/загрузки сгенерированных текстов
    summaries_csv_path = os.path.join(args.save_dir, "generated_summaries.csv")

    # 2. Инициализируем LLM-модель (она нужна и для генерации саммари, и для верификации)
    print(f"Загрузка LLM: {args.llm}...")
    # LLMVerifier загрузит модель на GPU в 4-bit или fp16
    verifier = LLMVerifier(args.llm)

    # 3. Готовим датасет (загрузка + генерация пересказов)
    df_data = get_or_create_dataset(args.num_texts, verifier, summaries_csv_path)

    # 4. Освобождаем память GPU от LLM, если верификатор НЕ будет использоваться при расчете графов
    if not args.use_verifier:
        print("Выгрузка LLM для экономии видеопамяти перед запуском BERT...")
        # Сохраняем ссылки для удаления
        del verifier
        torch.cuda.empty_cache()
        verifier_to_use = None
    else:
        verifier_to_use = verifier

    # 5. Загружаем NLI-предиктор (RuBERT)
    model_path = get_model_path(args.model)
    print(f"Загрузка NLI-предиктора из {model_path}...")
    predictor = NLIPredictor(model_path)

    # 6. Запускаем оценку
    results = []
    print("\nНачинаем построение графов и расчет метрик...")
    
    for idx, row in tqdm(df_data.iterrows(), total=len(df_data), desc="Расчет метрик галлюцинаций"):
        text_idx = int(row["text_idx"])
        source = row["original_text"]
        summary = row["ai_summary"]

        try:
            # Строим двудольный кросс-документный граф
            G, sentences_a, sentences_b = graph_from_two_texts(
                text_source=source,
                text_summary=summary,
                predictor=predictor,
                verifier=verifier_to_use,
                proba_threshold=args.proba_threshold
            )

            # Вычисляем метрики галлюцинаций
            metrics = compute_hallucination_metrics(G)

            results.append({
                "text_idx": text_idx,
                "num_sentences_source": len(sentences_a),
                "num_sentences_summary": len(sentences_b),
                "faithfulness_score": metrics["faithfulness_score"],
                "contradiction_ratio": metrics["contradiction_ratio"],
                "unsupported_ratio": metrics["unsupported_ratio"],
                "coherence_index": metrics["coherence_index"]
            })
        except Exception as e:
            print(f"Ошибка при обработке текста {text_idx}: {e}")
            continue

    # 7. Сохраняем детальный отчет
    df_report = pd.DataFrame(results)
    report_path = os.path.join(args.save_dir, "metrics_report.csv")
    df_report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\nДетальный отчет сохранен в: {report_path}")

    # 8. Рассчитываем и выводим агрегированную статистику
    summary_stats = {
        "total_evaluated_texts": len(df_report),
        "mean_sentences_source": float(df_report["num_sentences_source"].mean()),
        "mean_sentences_summary": float(df_report["num_sentences_summary"].mean()),
        "faithfulness_score": {
            "mean": float(df_report["faithfulness_score"].mean()),
            "median": float(df_report["faithfulness_score"].median()),
            "std": float(df_report["faithfulness_score"].std())
        },
        "contradiction_ratio": {
            "mean": float(df_report["contradiction_ratio"].mean()),
            "median": float(df_report["contradiction_ratio"].median()),
            "std": float(df_report["contradiction_ratio"].std())
        },
        "unsupported_ratio": {
            "mean": float(df_report["unsupported_ratio"].mean()),
            "median": float(df_report["unsupported_ratio"].median()),
            "std": float(df_report["unsupported_ratio"].std())
        },
        "coherence_index": {
            "mean": float(df_report["coherence_index"].mean()),
            "median": float(df_report["coherence_index"].median()),
            "std": float(df_report["coherence_index"].std())
        }
    }

    summary_json_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)
    
    print("\n=== АГРЕГИРОВАННАЯ СТАТИСТИКА ЭКСПЕРИМЕНТА ===")
    print(f"Всего оценено текстов: {summary_stats['total_evaluated_texts']}")
    print(f"Среднее число предложений (Оригинал): {summary_stats['mean_sentences_source']:.2f}")
    print(f"Среднее число предложений (Пересказ): {summary_stats['mean_sentences_summary']:.2f}")
    print("\nFaithfulness Score (Подтвержденные факты):")
    print(f"  Среднее: {summary_stats['faithfulness_score']['mean']:.4f}")
    print(f"  Медиана: {summary_stats['faithfulness_score']['median']:.4f}")
    print(f"  СКО (std): {summary_stats['faithfulness_score']['std']:.4f}")
    print("\nContradiction Ratio (Искаженные факты / Галлюцинации):")
    print(f"  Среднее: {summary_stats['contradiction_ratio']['mean']:.4f}")
    print(f"  Медиана: {summary_stats['contradiction_ratio']['median']:.4f}")
    print(f"  СКО (std): {summary_stats['contradiction_ratio']['std']:.4f}")
    print("\nUnsupported Ratio (Неподдерживаемые факты):")
    print(f"  Среднее: {summary_stats['unsupported_ratio']['mean']:.4f}")
    print(f"  Медиана: {summary_stats['unsupported_ratio']['median']:.4f}")
    print(f"  СКО (std): {summary_stats['unsupported_ratio']['std']:.4f}")
    print("\nCoherence Index (Индекс связности графа):")
    print(f"  Среднее: {summary_stats['coherence_index']['mean']:.4f}")
    print(f"\nСводные метрики сохранены в: {summary_json_path}")

    # 9. Отрисовка графиков
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # График распределения Faithfulness Score
        plt.subplot(1, 2, 1)
        plt.hist(df_report["faithfulness_score"], bins=20, color="skyblue", edgecolor="black", alpha=0.7)
        plt.title("Распределение Faithfulness Score")
        plt.xlabel("Faithfulness Score")
        plt.ylabel("Количество текстов")
        
        # График распределения Contradiction Ratio
        plt.subplot(1, 2, 2)
        plt.hist(df_report["contradiction_ratio"], bins=20, color="salmon", edgecolor="black", alpha=0.7)
        plt.title("Распределение Contradiction Ratio (Галлюцинации)")
        plt.xlabel("Contradiction Ratio")
        plt.ylabel("Количество текстов")
        
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "metrics_distribution.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Графики распределения сохранены в: {plot_path}")
    except Exception as e:
        print(f"Ошибка при отрисовке графиков: {e}")


if __name__ == "__main__":
    main()
