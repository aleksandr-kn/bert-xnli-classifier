#!/usr/bin/env python
# -*- coding: utf-8 -*-

from razdel import sentenize


def split_sentences(text: str) -> list[str]:
    """Разбивает текст на предложения (razdel - сегментация для русского языка)."""
    return [sent.text for sent in sentenize(text)]
