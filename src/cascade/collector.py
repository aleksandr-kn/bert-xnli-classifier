#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Сбор hard negatives из корпуса текстов.

Прогоняет тексты через BERT + LLM, собирает пары, где LLM не согласилась с BERT.
Эти пары используются для дообучения BERT в следующей итерации каскада.
"""

from dataclasses import dataclass, field
import pandas as pd

from src.utils.text import split_sentences
from src.utils.data import load_dataset
from src.graph.builder import build_sentence_pairs, build_relation_map


@dataclass
class HardNegative:
    premise: str
    hypothesis: str
    bert_label: int             # Исходная метка BERT
    llm_label: int              # Метка от LLM (ground truth для дообучения)
    bert_proba: list[float]     # [P_ent, P_neu, P_con]
    llm_reasoning: str
    source_text_idx: int
    sentence_i: int
    sentence_j: int


def select_candidates(rel_map, target_labels=None):
    """
    Отбирает пары для LLM-верификации.

    Args:
        rel_map: карта отношений от build_relation_map
        target_labels: список меток для фильтрации.
            По умолчанию [2] - только contradiction.
            Передать [0, 1, 2] для сбора всех типов.

    Returns:
        список индексов [(i, j), ...]
    """
    if target_labels is None:
        target_labels = [2]

    candidates = []
    for (i, j), info in rel_map.items():
        if info["label"] in target_labels:
            candidates.append((i, j))
    return candidates


def collect_from_text(text, text_idx, predictor, verifier,
                      target_labels=None):
    """
    Один текст -> список hard negatives + статистика.

    Args:
        text: исходный текст
        text_idx: индекс текста в корпусе
        predictor: NLIPredictor (метод predict_batch)
        verifier: LLMVerifier (метод verify_batch)
        target_labels: метки для фильтрации кандидатов

    Returns:
        (hard_negatives, stats): список HardNegative и dict со статистикой
    """
    sentences = split_sentences(text)

    stats = {
        "text_idx": text_idx,
        "num_sentences": len(sentences),
        "total_pairs": 0,
        "candidates": 0,
        "llm_calls": 0,
        "overrides": 0,
    }

    if len(sentences) < 2:
        return [], stats

    # NLI-предсказания для всех пар
    pairs, indices = build_sentence_pairs(sentences)
    preds, probas = predictor.predict_batch(pairs)
    rel_map = build_relation_map(indices, preds, probas)

    stats["total_pairs"] = len(rel_map)

    # Отбор кандидатов для LLM-верификации
    cand_indices = select_candidates(rel_map, target_labels)
    stats["candidates"] = len(cand_indices)

    if not cand_indices:
        return [], stats

    # Подготовка данных для LLM
    cand_data = []
    for i, j in cand_indices:
        info = rel_map[(i, j)]
        cand_data.append({
            "premise": sentences[i],
            "hypothesis": sentences[j],
            "bert_label": info["label"],
            "bert_proba": info["proba"],
        })

    # LLM-верификация
    results = verifier.verify_batch(cand_data, context=text)
    stats["llm_calls"] = len(results)

    # Сравнение: собираем hard negatives (LLM не согласилась с BERT)
    hard_negatives = []
    for (i, j), cand, (llm_label, confidence, reasoning) in zip(
            cand_indices, cand_data, results):
        if llm_label != cand["bert_label"]:
            hn = HardNegative(
                premise=cand["premise"],
                hypothesis=cand["hypothesis"],
                bert_label=cand["bert_label"],
                llm_label=llm_label,
                bert_proba=cand["bert_proba"].tolist() if hasattr(cand["bert_proba"], "tolist") else list(cand["bert_proba"]),
                llm_reasoning=reasoning,
                source_text_idx=text_idx,
                sentence_i=i,
                sentence_j=j,
            )
            hard_negatives.append(hn)

    stats["overrides"] = len(hard_negatives)

    return hard_negatives, stats


def collect_from_corpus(corpus_path, predictor, verifier,
                        text_column="text", target_labels=None):
    """
    Весь корпус -> аккумулированные hard negatives + общая статистика.

    Поддерживает два режима:
    1. Режим текста (text_column): разбивает текст на предложения и проверяет все пары.
    2. Режим готовых пар: если в файле есть колонки 'premise' и 'hypothesis',
       использует их напрямую.
    """
    df = load_dataset(corpus_path)

    all_hard_negatives = []
    corpus_stats = {
        "num_texts": len(df),
        "total_pairs": 0,
        "total_candidates": 0,
        "total_llm_calls": 0,
        "total_overrides": 0,
        "per_text": [],
    }

    # ПРОВЕРКА РЕЖИМА: готовые пары или сырой текст
    is_pair_mode = "premise" in df.columns and "hypothesis" in df.columns

    if is_pair_mode:
        print(f"Обнаружен режим готовых пар (premise/hypothesis).")
        # Группируем пары (в RuWANLI они независимы, но для статистики считаем как один блок)
        # Для простоты обрабатываем все пары сразу или батчами
        pairs_data = []
        for idx, row in df.iterrows():
            pairs_data.append({
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "bert_label": row["label"] if "label" in row else None,
            })
        
        # Прогон через BERT
        pairs = [(p["premise"], p["hypothesis"]) for p in pairs_data]
        preds, probas = predictor.predict_batch(pairs)
        
        # Подготовка кандидатов для LLM
        cand_indices = []
        cand_data = []
        target_labels = target_labels or [2]
        
        for i, (pred, proba) in enumerate(zip(preds, probas)):
            if pred in target_labels:
                cand_indices.append(i)
                cand_data.append({
                    "premise": pairs[i][0],
                    "hypothesis": pairs[i][1],
                    "bert_label": int(pred),
                    "bert_proba": proba,
                })
        
        corpus_stats["total_pairs"] = len(pairs)
        corpus_stats["total_candidates"] = len(cand_indices)
        
        if cand_indices:
            results = verifier.verify_batch(cand_data)
            corpus_stats["total_llm_calls"] = len(results)
            
            for idx_in_cand, (llm_label, confidence, reasoning) in enumerate(results):
                orig_idx = cand_indices[idx_in_cand]
                if llm_label != cand_data[idx_in_cand]["bert_label"]:
                    hn = HardNegative(
                        premise=cand_data[idx_in_cand]["premise"],
                        hypothesis=cand_data[idx_in_cand]["hypothesis"],
                        bert_label=cand_data[idx_in_cand]["bert_label"],
                        llm_label=llm_label,
                        bert_proba=cand_data[idx_in_cand]["bert_proba"].tolist(),
                        llm_reasoning=reasoning,
                        source_text_idx=0, # В режиме пар индекс размыт
                        sentence_i=orig_idx,
                        sentence_j=0,
                    )
                    all_hard_negatives.append(hn)
            
            corpus_stats["total_overrides"] = len(all_hard_negatives)
    else:
        # СТАНДАРТНЫЙ РЕЖИМ: разбиение текста на предложения
        for row_idx, text in enumerate(df[text_column]):
            print(f"\n--- Текст {row_idx + 1}/{len(df)} ---")
            hard_negatives, stats = collect_from_text(
                text, row_idx, predictor, verifier, target_labels
            )
            all_hard_negatives.extend(hard_negatives)

            corpus_stats["total_pairs"] += stats["total_pairs"]
            corpus_stats["total_candidates"] += stats["candidates"]
            corpus_stats["total_llm_calls"] += stats["llm_calls"]
            corpus_stats["total_overrides"] += stats["overrides"]
            corpus_stats["per_text"].append(stats)

            print(f"  Пар: {stats['total_pairs']}, "
                f"кандидатов: {stats['candidates']}, "
                f"overrides: {stats['overrides']}")

    print(f"\n=== Итого по корпусу ===")
    print(f"  Режим: {'Пары' if is_pair_mode else 'Текст'}")
    print(f"  Всего пар: {corpus_stats['total_pairs']}")
    print(f"  Кандидатов для LLM: {corpus_stats['total_candidates']}")
    print(f"  LLM overrides: {corpus_stats['total_overrides']}")

    return all_hard_negatives, corpus_stats


def save_hard_negatives(hard_negatives, output_path):
    """
    Сохранить hard negatives в CSV.

    Колонки premise/hypothesis/label совместимы с форматом XNLI
    для прямого использования в обучении.
    """
    if not hard_negatives:
        print("Нет hard negatives для сохранения.")
        return

    records = []
    for hn in hard_negatives:
        records.append({
            "premise": hn.premise,
            "hypothesis": hn.hypothesis,
            "label": hn.llm_label,  # Метка от LLM - ground truth
            "bert_label": hn.bert_label,
            "bert_proba": hn.bert_proba,
            "llm_reasoning": hn.llm_reasoning,
            "source_text_idx": hn.source_text_idx,
            "sentence_i": hn.sentence_i,
            "sentence_j": hn.sentence_j,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Hard negatives сохранены: {output_path} ({len(df)} записей)")


def load_hard_negatives(path):
    """Загрузить ранее собранные hard negatives."""
    return pd.read_csv(path)
