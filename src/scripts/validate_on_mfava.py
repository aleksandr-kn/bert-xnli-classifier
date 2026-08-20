#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт валидации NLI-графа на эталонном датасете (mFAVA/Mixed-Summ).
Архитектурно обособлен, переиспользует базовые компоненты проекта.
"""

import os
os.environ["HF_HOME"] = "F:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "F:/huggingface_cache"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import argparse
import pandas as pd
from tqdm import tqdm

# Импорты из существующей архитектуры (не ломаем ядро!)
from src.models.nli_predictor import NLIPredictor
from src.graph.builder import graph_from_two_texts
from src.graph.analysis import compute_hallucination_metrics
from src.utils.model_registry import get_model_path

def parse_args():
    parser = argparse.ArgumentParser(description="Инференс NLI-графа на валидационном датасете")
    parser.add_argument("--data_path", type=str, default="data/validation/mfava_ru_test.csv",
                        help="Путь к подготовленному CSV файлу")
    parser.add_argument("--model", type=str, default="rubert-large-xnli",
                        help="Путь к NLI модели (slug или локальный)")
    parser.add_argument("--use_llm_cascade", action="store_true",
                        help="Включить LLM-верификатор (Qwen) для повышения Recall")
    parser.add_argument("--llm_model", type=str, default="qwen2.5:14b",
                        help="Модель для верификатора (например, qwen2.5:14b для Ollama или Qwen/Qwen2.5-14B-Instruct для HF)")
    parser.add_argument("--verifier_type", type=str, default="Ollama",
                        choices=["Ollama", "HallucinationSpotter", "StrictNLI"],
                        help="Тип верификатора: Ollama (быстрый C++ API), HallucinationSpotter (HF 4-bit) или StrictNLI")
    parser.add_argument("--proba_threshold", type=float, default=0.30,
                        help="Мягкий порог вероятности противоречия")
    parser.add_argument("--save_dir", type=str, default="outputs/validation",
                        help="Директория для сохранения результатов предсказаний")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Ошибка: Файл данных {args.data_path} не найден!")
        print("Сначала запустите: python -m src.scripts.prepare_mfava")
        return
        
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Загрузка датасета из {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    model_path = get_model_path(args.model)
    print(f"Загрузка NLI-предиктора из {model_path}...")
    predictor = NLIPredictor(model_path)
    
    verifier = None
    if args.use_llm_cascade:
        if args.verifier_type == "Ollama":
            from src.models.verifiers import OllamaVerifier as VerifierCls
        elif args.verifier_type == "HallucinationSpotter":
            from src.models.verifiers import HallucinationSpotterVerifier as VerifierCls
        else:
            from src.models.verifiers import StrictNLIVerifier as VerifierCls
            
        print(f"Загрузка LLM-верификатора ({args.verifier_type}) с моделью {args.llm_model}...")
        verifier = VerifierCls(model_name=args.llm_model)
    
    results = []
    
    # Счетчики экономии вычислений
    total_sentences_processed = 0
    total_llm_calls = 0
    
    print("\nНачинаем валидацию (Inference)...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Анализ текстов"):
        text_id = row.get("id", idx)
        source = row["source"]
        summary = row["summary"]
        human_label = row.get("human_label", 0)
        
        # Защита от пустых текстов
        if pd.isna(source) or pd.isna(summary):
            continue
            
        try:
            # Строим граф используя вашу готовую функцию!
            G, sentences_a, sentences_b = graph_from_two_texts(
                text_source=str(source),
                text_summary=str(summary),
                predictor=predictor,
                verifier=verifier,
                proba_threshold=args.proba_threshold
            )
            
            metrics = compute_hallucination_metrics(G)
            
            # Собираем статистику маршрутизации
            total_sentences_processed += len(sentences_b)
            total_llm_calls += G.graph.get("llm_calls", 0)
            
            # Грамотное математическое решение (Soft Thresholding):
            # Используем индекс когерентности (max_entail_proba - max_contradiction_proba).
            # Если перевес в сторону подтверждения меньше 0.01 (или уходит в минус из-за сильного противоречия),
            # то мы бракуем текст. Это спасает от ложных тревог, вызванных случайными выбросами вероятностей.
            is_hallucination = metrics["coherence_index"] < 0.01
            predicted_label = 1 if is_hallucination else 0
            
            results.append({
                "id": text_id,
                "human_label": human_label,
                "predicted_label": predicted_label,
                "contradiction_ratio": metrics["contradiction_ratio"],
                "coherence_index": metrics["coherence_index"],
                "faithfulness_score": metrics["faithfulness_score"]
            })
            
        except Exception as e:
            print(f"Ошибка при обработке ID {text_id}: {e}")
            continue
            
    df_results = pd.DataFrame(results)
    
    save_path = os.path.join(args.save_dir, "validation_predictions.csv")
    df_results.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    # Сохраняем статистику вычислений (Compute Reduction)
    stats = {
        "total_sentences_processed": total_sentences_processed,
        "total_llm_calls_made": total_llm_calls,
        "llm_calls_saved": total_sentences_processed - total_llm_calls,
        "routing_rate_percent": round((total_llm_calls / total_sentences_processed) * 100, 2) if total_sentences_processed > 0 else 0,
        "compute_reduction_percent": round(((total_sentences_processed - total_llm_calls) / total_sentences_processed) * 100, 2) if total_sentences_processed > 0 else 0
    }
    
    stats_path = os.path.join(args.save_dir, "routing_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
        
    print(f"\nИнференс завершен! Результаты предсказаний сохранены в {save_path}")
    print(f"Статистика вычислений (Compute Reduction) сохранена в {stats_path}")
    print(f"Отправлено в LLM: {stats['total_llm_calls_made']} из {stats['total_sentences_processed']} предложений.")
    print(f"Сэкономлено {stats['compute_reduction_percent']}% ресурсов LLM!")
    print("Теперь можно запускать расчет ROC-AUC и F1 (Шаг 3).")

if __name__ == "__main__":
    main()
