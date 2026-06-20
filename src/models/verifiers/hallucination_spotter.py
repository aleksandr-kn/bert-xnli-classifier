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
            "Ты — безжалостный редактор и факт-чекер. Твоя задача — находить галлюцинации в пересказе текста.\n\n"
            "ПРАВИЛА:\n"
            "1. Если в пересказе (гипотезе) есть ХОТЯ БЫ ОДНА ДЕТАЛЬ (имя, характеристика, действие, факт), "
            "которой не было в оригинальном тексте — это галлюцинация.\n"
            "2. Выдавай 'contradiction', если есть ПРЯМОЕ противоречие ИЛИ ДОБАВЛЕНЫ новые факты (домысел).\n"
            "3. Выдавай 'entailment', только если ВСЯ информация из пересказа на 100% подтверждается оригиналом.\n"
            "4. Выдавай 'neutral' только в крайнем случае, если фраза вообще не несет фактологической нагрузки.\n\n"
            "Формат ответа: краткое объяснение, затем ВЕРДИКТ: <метка>"
        )
        ctx_block = f"Оригинальный текст (контекст): \"{context}\"\n\n" if context else ""
        user_msg = (
            f"{ctx_block}Цитата из оригинала: \"{premise}\"\n"
            f"Предложение из пересказа: \"{hypothesis}\"\n\n"
            "Проверь, содержит ли пересказ галлюцинации или отсебятину по сравнению с оригиналом.\n"
            "ВЕРДИКТ: entailment, contradiction или neutral"
        )
        return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]

    def _parse_response(self, text, fallback_label):
        match = self._ANSWER_RE.search(text)
        if match:
            return _LABEL_TO_ID[match.group(1).lower()], match.group(1).lower()
        return fallback_label, None
