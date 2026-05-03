#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Подготовка подмножества RuWANLI для теста каскада.
"""

import pandas as pd
import os

def main():
    input_path = "data/ruwanli/test.parquet"
    output_path = "data/ruwanli_subset_test.csv"
    
    if not os.path.exists(input_path):
        print(f"Ошибка: {input_path} не найден. Сначала запустите download_ruwanli.py")
        return

    print(f"Загрузка {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Берем первые 500 строк для быстрого теста
    subset = df.head(500)
    
    subset.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Создано подмножество: {output_path} (500 строк)")

if __name__ == "__main__":
    main()
