# -*- coding: utf-8 -*-
import re
from .base import BaseLLMVerifier, _LABEL_TO_ID

class HallucinationSpotterVerifier(BaseLLMVerifier):
    """
    Агрессивный верификатор (Факт-чекер) для суммаризаций.
    Жесткое правило: любая отсебятина, которой нет в оригинале = contradiction (галлюцинация).
    """
    _ANSWER_RE = re.compile(r"ВЕРДИКТ:\s*(entailment|contradiction|neutral)", re.IGNORECASE)

    def _build_prompt(self, premise, hypothesis, bert_label, context=None):
        sys_prompt = (
            "Ты — умный и справедливый судья, оценивающий качество генерации текста (RAG).\n\n"
            "ПРАВИЛА:\n"
            "1. Твоя главная задача — отличить безобидное перефразирование (синонимы, упрощения, логичные выводы) "
            "от реальных галлюцинаций (искажение фактов, добавление новых имен, дат или событий).\n"
            "2. Выдавай 'entailment', если смысл сгенерированного предложения ПОДТВЕРЖДАЕТСЯ оригинальным текстом. "
            "Синонимы и логические следствия ДОПУСТИМЫ.\n"
            "3. Выдавай 'contradiction', если предложение ПРЯМО ПРОТИВОРЕЧИТ оригиналу ИЛИ содержит 'отсебятину' (выдуманные детали).\n"
            "4. Выдавай 'neutral', только если фраза бессмысленна.\n\n"
            "Формат ответа: краткое объяснение (1-2 предложения), затем ВЕРДИКТ: <метка>"
        )
        
        # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: Не фокусируем LLM на конкретной цитате, на которой ошибся BERT. 
        # Заставляем ее смотреть на весь контекст целиком!
        user_msg = (
            f"ОРИГИНАЛЬНЫЙ ТЕКСТ:\n\"{context}\"\n\n"
            f"ПРОВЕРЯЕМОЕ ПРЕДЛОЖЕНИЕ:\n\"{hypothesis}\"\n\n"
            "Опираясь ТОЛЬКО на оригинальный текст, вынеси вердикт.\n"
            "ВЕРДИКТ: entailment, contradiction или neutral"
        )
        
        return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]

    def _parse_response(self, text, fallback_label):
        match = self._ANSWER_RE.search(text)
        if match:
            return _LABEL_TO_ID[match.group(1).lower()], match.group(1).lower()
        return fallback_label, None
