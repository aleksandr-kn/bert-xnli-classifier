#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт расчета итоговых метрик валидации NLI-графа.
Сравнивает предсказания графа с 'золотой' разметкой mFAVA.
Рассчитывает Precision, Recall, F1-score и строит ROC-AUC.
"""

import os
import pandas as pd
import argparse
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(description="Расчет итоговых метрик качества")
    parser.add_argument("--predictions_path", type=str, default="outputs/validation/validation_predictions.csv",
                        help="Путь к результатам инференса")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.predictions_path):
        print(f"Ошибка: Файл {args.predictions_path} не найден!")
        return
        
    df = pd.read_csv(args.predictions_path)
    
    # Отфильтруем строки, где почему-то нет предсказаний
    df = df.dropna(subset=['human_label', 'predicted_label', 'contradiction_ratio'])
    
    y_true = df['human_label'].astype(int)
    y_pred = df['predicted_label'].astype(int)
    
    # Для ROC-AUC используем Contradiction Ratio как уверенность (probability) класса 1
    # Если хотим использовать Coherence Index, то логика обратная (чем меньше когерентность, тем больше вероятность галлюцинации).
    # Но проще взять CR.
    y_score = df['contradiction_ratio'].astype(float)
    
    print("=== РЕЗУЛЬТАТЫ ВАЛИДАЦИИ NLI-ГРАФА ===\n")
    print(f"Всего текстов оценено: {len(df)}")
    print(f"Из них реальных галлюцинаций (label=1): {sum(y_true)}")
    print(f"Чистых текстов (label=0): {len(y_true) - sum(y_true)}\n")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("--- Confusion Matrix ---")
    print(f"True Positives (Верно найдено галлюцинаций): {tp}")
    print(f"True Negatives (Верно подтверждено чистых): {tn}")
    print(f"False Positives (Ложная тревога): {fp}")
    print(f"False Negatives (Пропуск галлюцинации): {fn}\n")
    
    # 2. Метрики классификации
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    
    print("--- Главные Метрики ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}\n")
    
    # 3. Детальный отчет
    print("--- Детальный отчет по классам ---")
    print(classification_report(y_true, y_pred, target_names=["Clear Text (0)", "Hallucination (1)"]))

if __name__ == "__main__":
    main()
