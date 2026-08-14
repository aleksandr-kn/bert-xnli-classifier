#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для подготовки эталонного датасета mFAVA (WueNLP/mHallucination_Detection) для валидации.
Переформулирован в Sentence-Level формат. Каждое предложение из сгенерированного текста 
проверяется на пересечение с человеческими тегами галлюцинаций (0 или 1).
"""

import os
import re
import pandas as pd
from datasets import load_dataset
import argparse
from razdel import sentenize

def parse_args():
    parser = argparse.ArgumentParser(description="Подготовка mFAVA датасета (Sentence-Level)")
    parser.add_argument("--dataset_name", type=str, default="WueNLP/mHallucination_Detection", 
                        help="Путь к датасету на HuggingFace")
    parser.add_argument("--save_path", type=str, default="data/validation/mfava_ru_sentences_test.csv",
                        help="Куда сохранить подготовленный CSV")
    return parser.parse_args()

def parse_gold_to_sentences(gold_text):
    """
    Разбирает текст с XML-тегами.
    Возвращает список словарей: [{'hypothesis': текст_предложения, 'label': 0/1}, ...]
    """
    if pd.isna(gold_text) or not str(gold_text).strip():
        return []
        
    clean_text = ""
    is_hallucinated = []
    
    # Разбиваем по любым XML-подобным тегам
    parts = re.split(r'(</?[a-zA-Z0-9_]+>)', str(gold_text))
    in_tag = False
    
    for part in parts:
        if part.startswith('</'):
            in_tag = False
        elif part.startswith('<'):
            in_tag = True
        else:
            clean_text += part
            # Запоминаем статус in_tag для каждого символа
            is_hallucinated.extend([in_tag] * len(part))
            
    sentences = []
    for s in sentenize(clean_text):
        # Если хотя бы один символ в предложении был внутри тега, предложение = галлюцинация
        label = 1 if any(is_hallucinated[s.start:s.stop]) else 0
        sentences.append({
            'hypothesis': s.text.strip(),
            'label': label
        })
        
    return sentences

def main():
    args = parse_args()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    print(f"Загрузка датасета: {args.dataset_name}...")
    try:
        dataset = load_dataset(args.dataset_name)
        df = dataset['test'].to_pandas()
        
        print(f"Загружено {len(df)} записей. Фильтрация русского языка...")
        
        df = df[df['language'] == 'russian']
        print(f"Осталось {len(df)} документов на русском языке.")
        
        records = []
        total_hallucinated = 0
        
        for idx, row in df.iterrows():
            gold_text = row.get('gold_annotations', '')
            
            # Разбираем текст на предложения
            if not pd.isna(gold_text) and str(gold_text).strip():
                sentences_data = parse_gold_to_sentences(gold_text)
            else:
                gen_text = row.get('generated_text', '')
                if pd.isna(gen_text):
                    continue
                sentences_data = parse_gold_to_sentences(gen_text) # Без тегов всё будет 0
                
            source = str(row.get('references', '')).strip()
            
            for s_idx, s_data in enumerate(sentences_data):
                if not s_data['hypothesis']:
                    continue
                records.append({
                    'id': f"{idx}_{s_idx}",
                    'source': source,
                    'summary': s_data['hypothesis'], # В старом коде колонка называлась summary
                    'human_label': s_data['label']
                })
                if s_data['label'] == 1:
                    total_hallucinated += 1
            
        unified_df = pd.DataFrame(records)
        unified_df.to_csv(args.save_path, index=False, encoding='utf-8-sig')
        print(f"Готово! Из {len(df)} документов извлечено {len(unified_df)} предложений.")
        print(f"Из них галлюцинаций (Label=1): {total_hallucinated}")
        print(f"Чистых фактов (Label=0): {len(unified_df) - total_hallucinated}")
        print(f"Базовый баланс (Base Rate) галлюцинаций: {total_hallucinated / len(unified_df) * 100:.1f}%")
        print(f"Датасет сохранен в {args.save_path}")
        
    except Exception as e:
        print(f"Ошибка при обработке датасета: {e}")

if __name__ == "__main__":
    main()
