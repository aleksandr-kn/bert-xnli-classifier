#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path


def load_dataset(file_path) -> pd.DataFrame:
    """
    Простая загрузка датасета в pandas.DataFrame.
    Поддерживает parquet, csv, tsv, json, excel (по расширению).
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    elif ext == ".json":
        return pd.read_json(path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}")
