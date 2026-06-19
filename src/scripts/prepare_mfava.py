#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для подготовки эталонного датасета mFAVA (WueNLP/mHallucination_Detection) для валидации.
Скачивает данные, фильтрует русскоязычный сплит и сохраняет в унифицированном формате CSV.
"""

import os
import re
import pandas as pd
from datasets import load_dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Подготовка mFAVA датасета")
    parser.add_argument("--dataset_name", type=str, default="WueNLP/mHallucination_Detection", 
                        help="Путь к датасету на HuggingFace")
    parser.add_argument("--save_path", type=str, default="data/validation/mfava_ru_test.csv",
                        help="Куда сохранить подготовленный CSV")
    return parser.parse_args()

def extract_hallucination_info(gold_text):
    """
    Парсит gold_annotations, ищет теги <contradictory> и <unverifiable>.
    Если теги есть, значит label = 1 (галлюцинация).
    Возвращает (label, hallucinated_span).
    """
    if pd.isna(gold_text):
        return 0, ""
        
    span = ""
    label = 0
    
    # Ищем теги
    matches = re.findall(r'<(contradictory|unverifiable)>(.*?)</\1>', gold_text, re.DOTALL)
    if matches:
        label = 1
        # Собираем все найденные спаны через разделитель
        span = " | ".join([m[1] for m in matches])
        
    return label, span

def main():
    args = parse_args()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    print(f"Загрузка датасета: {args.dataset_name}...")
    try:
        dataset = load_dataset(args.dataset_name)
        df = dataset['test'].to_pandas()
        
        print(f"Загружено {len(df)} записей. Фильтрация русского языка...")
        
        # В mFAVA язык называется 'russian'
        df = df[df['language'] == 'russian']
        print(f"Осталось {len(df)} записей на русском языке.")
        
        records = []
        for idx, row in df.iterrows():
            gold_text = row.get('gold_annotations', '')
            label, span = extract_hallucination_info(gold_text)
            
            # Если gold_annotations пустой, используем generated_text
            summary = row.get('generated_text', '')
            
            records.append({
                'id': idx,
                'source': row.get('references', ''),
                'summary': summary,
                'human_label': label,
                'hallucinated_span': span
            })
            
        unified_df = pd.DataFrame(records)
        unified_df.to_csv(args.save_path, index=False, encoding='utf-8-sig')
        print(f"Готово! Извлечено {unified_df['human_label'].sum()} галлюцинаций из {len(unified_df)} текстов.")
        print(f"Датасет сохранен в {args.save_path}")
        
    except Exception as e:
        print(f"Ошибка при обработке датасета: {e}")

if __name__ == "__main__":
    main()
