#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Верификация NLI-предсказаний с помощью локальной LLM.

BERT-модель используется как быстрый фильтр для всех O(n^2) пар,
а LLM проверяет только кандидатов (contradiction + неуверенные пары),
выполняя пошаговое рассуждение с учётом полного контекста.
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Маппинг строковых меток LLM → числовые (совпадает с builder.LABEL_ID)
_LABEL_TO_ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}

# Обратный маппинг числовых меток → строковые
_ID_TO_LABEL = {v: k for k, v in _LABEL_TO_ID.items()}

# Regex для извлечения метки из ответа LLM
_ANSWER_RE = re.compile(r"ОТВЕТ:\s*(entailment|contradiction|neutral)", re.IGNORECASE)

_SYSTEM_PROMPT = """\
Ты — верификатор предсказаний модели Natural Language Inference (NLI).

NLI-классификатор определил отношение между парой высказываний. \
Твоя задача — проверить это предсказание, используя логический анализ и контекст.

Определения меток:
- entailment: если предпосылка истинна, то гипотеза ОБЯЗАТЕЛЬНО истинна.
- contradiction: предпосылка и гипотеза НЕ МОГУТ быть истинны одновременно.
- neutral: гипотеза может быть как истинной, так и ложной — информации недостаточно.

Формат ответа: краткое рассуждение, затем ОТВЕТ: <метка>"""

_USER_TEMPLATE = """\
{context_block}Предпосылка: "{premise}"
Гипотеза: "{hypothesis}"
Предсказание классификатора: {bert_label_name}

Проверь это предсказание. Определи субъекты обоих высказываний, \
оцени логическое отношение и напиши свой вердикт.
ОТВЕТ: entailment, contradiction или neutral"""


class LLMVerifier:
    """
    Верификатор NLI-предсказаний на основе генеративной LLM.

    Загружает модель в 4-bit квантизации (bitsandbytes) если доступно,
    иначе — в fp16. Формирует промпт с контекстом и парой предложений,
    парсит ответ.
    """

    def __init__(self, model_name, device=None, max_new_tokens=512):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Пробуем 4-bit квантизацию, при ошибке — fallback на fp16
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                # {"": 0} — всё на GPU 0 без offloading; обходит баг accelerate,
                # где dispatch_model вызывает .to() на квантизированной модели
                device_map={"": 0},
            )
            print("LLM загружена в 4-bit квантизации.")
        except Exception as e:
            print(f"4-bit квантизация недоступна ({e}), загрузка в fp16...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            print("LLM загружена в fp16.")

        self.model.eval()

    def _build_prompt(self, premise, hypothesis, bert_label, context=None):
        """Формирует список сообщений в chat-формате."""
        if context:
            context_block = f"Контекст: \"{context}\"\n\n"
        else:
            context_block = ""

        user_msg = _USER_TEMPLATE.format(
            context_block=context_block,
            premise=premise,
            hypothesis=hypothesis,
            bert_label_name=_ID_TO_LABEL[bert_label],
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def _parse_response(self, text, fallback_label):
        """Извлекает метку из ответа LLM. При неудаче возвращает fallback."""
        match = _ANSWER_RE.search(text)
        if match:
            label_str = match.group(1).lower()
            return _LABEL_TO_ID[label_str], label_str
        return fallback_label, None

    def verify(self, premise, hypothesis, bert_label, bert_proba, context=None):
        """
        Проверяет одну пару через LLM.

        Args:
            premise: текст предпосылки
            hypothesis: текст гипотезы
            bert_label: числовая метка от BERT (0/1/2)
            bert_proba: массив вероятностей от BERT [P_ent, P_neu, P_con]
            context: полный текст для контекста (опционально)

        Returns:
            (label, confidence, reasoning):
                label — числовая метка (0/1/2)
                confidence — максимальная вероятность BERT (для совместимости)
                reasoning — текст рассуждения LLM
        """
        messages = self._build_prompt(premise, hypothesis, bert_label, context)

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        # Декодируем только сгенерированные токены (без промпта)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        reasoning = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        label, parsed_label_str = self._parse_response(reasoning, bert_label)

        # confidence: если LLM дала ответ — 1.0, иначе берём BERT
        if parsed_label_str is not None:
            confidence = 1.0
        else:
            confidence = float(max(bert_proba))

        return label, confidence, reasoning

    def verify_batch(self, candidates, context=None):
        """
        Проверяет список кандидатов последовательно.

        Args:
            candidates: список dict с ключами:
                premise, hypothesis, bert_label, bert_proba
            context: полный текст (опционально)

        Returns:
            список (label, confidence, reasoning) для каждого кандидата
        """
        results = []
        total = len(candidates)
        for idx, cand in enumerate(candidates, 1):
            print(f"  LLM верификация: {idx}/{total} ...", end=" ", flush=True)
            result = self.verify(
                premise=cand["premise"],
                hypothesis=cand["hypothesis"],
                bert_label=cand["bert_label"],
                bert_proba=cand["bert_proba"],
                context=context,
            )
            label_name = {0: "entailment", 1: "neutral", 2: "contradiction"}[result[0]]
            print(f"-> {label_name}")
            results.append(result)
        return results
