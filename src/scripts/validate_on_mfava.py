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
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Модель для верификатора")
    parser.add_argument("--verifier_type", type=str, default="HallucinationSpotter",
                        choices=["StrictNLI", "HallucinationSpotter"],
                        help="Тип верификатора: StrictNLI или HallucinationSpotter")
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
        if args.verifier_type == "HallucinationSpotter":
            from src.models.verifiers import HallucinationSpotterVerifier as VerifierCls
        else:
            from src.models.verifiers import StrictNLIVerifier as VerifierCls
            
        print(f"Загрузка LLM-верификатора ({args.verifier_type}) из {args.llm_model}...")
        verifier = VerifierCls(model_name=args.llm_model)
    
    results = []
    
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
            
            # Наша бинарная логика предсказания:
            # Если Contradiction Ratio > 0, граф считает, что это галлюцинация
            predicted_label = 1 if metrics["contradiction_ratio"] > 0 else 0
            
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
    
    print(f"\nИнференс завершен! Результаты предсказаний сохранены в {save_path}")
    print("Теперь можно запускать расчет ROC-AUC и F1 (Шаг 3).")

if __name__ == "__main__":
    main()
