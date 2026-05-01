#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

"""
Реестр натренированных моделей.
Позволяет обращаться к моделям по удобным коротким именам (slugs),
а не по длинным сгенерированным путям с датами.
"""

MODEL_REGISTRY = {
    # ================================================================================
    # ДАТАСЕТ: RuWanli (Russian WordNet-based NLI)
    # ================================================================================
    
    # ai-forever/ruBert-large (Топ-1 по Accuracy/F1)
    # Accuracy: 0.7940 | F1: 0.7941 | ROC AUC: 0.9248
    "rubert-large-ruwanli": "outputs/models/ruwanli_ai-forever_ruBert-large_2026-02-21_17-07-50",

    # ================================================================================
    # ДАТАСЕТ: XNLI (Cross-lingual NLI - Russian part)
    # ================================================================================

    # ai-forever/ruBert-large (Лучшая XNLI модель)
    # Accuracy: 0.7854 | F1: 0.7847 | ROC AUC: 0.9280
    "rubert-large-xnli": "outputs/models/xnli_ai-forever_ruBert-large_2026-02-09_00-21-03",
    "rubert-large-baseline-xnli": "outputs/models/2026-02-04_07-13-03",

    # sberbank-ai/sbert_large_nlu_ru
    # Accuracy: 0.7818 | F1: 0.7812 | ROC AUC: 0.9261
    "sbert-large-xnli": "outputs/models/2026-02-05_05-30-00",

    # DeepPavlov/rubert-base-cased-conversational
    # Accuracy: 0.7730 | F1: 0.7731 | ROC AUC: 0.9203
    "rubert-base-xnli": "outputs/models/xnli_rubert-base-cased-conversational_2025-12-27_19-07-06",

    # DeepPavlov/rubert-base-cased
    # Accuracy: 0.7704 | F1: 0.7707 | ROC AUC: 0.9184
    "rubert-base-cased-xnli": "outputs/models/2025-11-30_07-16-47",

    # DeepPavlov/rubert-base-cased-sentence
    # Accuracy: 0.7610 | F1: 0.7610 | ROC AUC: 0.9134
    "rubert-base-sentence-xnli": "outputs/models/2025-12-01_11-54-01",
}

def get_model_path(slug: str) -> str:
    """
    Возвращает путь к модели по её короткому имени.
    Если передан существующий путь, возвращает его же.
    """
    if slug in MODEL_REGISTRY:
        path = MODEL_REGISTRY[slug]
        if not os.path.exists(path):
            print(f"Предупреждение: Путь {path} для модели '{slug}' не существует!")
        return path
    
    if os.path.exists(slug):
        return slug
        
    raise ValueError(
        f"Модель '{slug}' не найдена в реестре и не является существующим путем. "
        f"Доступные модели: {', '.join(MODEL_REGISTRY.keys())}"
    )
