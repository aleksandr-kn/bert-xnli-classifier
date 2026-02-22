#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re


def split_sentences(text: str) -> list[str]:
    """Разбивает текст на предложения по знакам препинания (.!?)."""
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
