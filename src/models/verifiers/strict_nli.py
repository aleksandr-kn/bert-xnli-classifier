# -*- coding: utf-8 -*-
import re
from .base import BaseLLMVerifier, _ID_TO_LABEL, _LABEL_TO_ID

class StrictNLIVerifier(BaseLLMVerifier):
    """
    Классический верификатор формальной логики.
    Идеален для точного NLI, где отсутствие информации — это neutral.
    """
    _ANSWER_RE = re.compile(r"ОТВЕТ:\s*(entailment|contradiction|neutral)", re.IGNORECASE)

    def _build_prompt(self, premise, hypothesis, bert_label, context=None):
        sys_prompt = (
            "Ты — верификатор предсказаний модели Natural Language Inference (NLI).\n\n"
            "Определения меток:\n"
            "- entailment: если предпосылка истинна, то гипотеза ОБЯЗАТЕЛЬНО истинна.\n"
            "- contradiction: предпосылка и гипотеза НЕ МОГУТ быть истинны одновременно.\n"
            "- neutral: гипотеза может быть как истинной, так и ложной — информации недостаточно.\n\n"
            "Формат ответа: краткое рассуждение, затем ОТВЕТ: <метка>"
        )
        ctx_block = f"Контекст: \"{context}\"\n\n" if context else ""
        user_msg = (
            f"{ctx_block}Предпосылка: \"{premise}\"\n"
            f"Гипотеза: \"{hypothesis}\"\n"
            f"Предсказание классификатора: {_ID_TO_LABEL[bert_label]}\n\n"
            "Проверь это предсказание. Определи субъекты обоих высказываний, "
            "оцени логическое отношение и напиши свой вердикт.\n"
            "ОТВЕТ: entailment, contradiction или neutral"
        )
        return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]

    def _parse_response(self, text, fallback_label):
        match = self._ANSWER_RE.search(text)
        if match:
            return _LABEL_TO_ID[match.group(1).lower()], match.group(1).lower()
        return fallback_label, None
