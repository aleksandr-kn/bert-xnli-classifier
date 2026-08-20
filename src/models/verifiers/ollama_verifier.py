# -*- coding: utf-8 -*-

"""
Автономный верификатор на базе Ollama REST API (llama.cpp C++ CUDA).
Обеспечивает ультрабыстрый инференс (~40 токенов/сек) без использования PyTorch.
Полностью совместим по интерфейсу с BaseLLMVerifier / HallucinationSpotterVerifier.
"""

import re
import json
import urllib.request

_LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
_ID_TO_LABEL = {v: k for k, v in _LABEL_TO_ID.items()}

class OllamaVerifier:
    """
    Отдельный легковесный верификатор фактов через локальный сервер Ollama.
    """
    _ANSWER_RE = re.compile(r"ВЕРДИКТ:\s*(entailment|contradiction|neutral)", re.IGNORECASE)

    def __init__(self, model_name="qwen2.5:14b", ollama_url="http://localhost:11434", max_new_tokens=256, num_ctx=4096):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.num_ctx = num_ctx
        
        # Проверяем подключение к серверу
        self._check_connection()

    def _send_request(self, endpoint, payload=None):
        """Отправляет запрос к Ollama минуя системные прокси (localhost)."""
        url = f"{self.ollama_url}{endpoint}"
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
        
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers)
        
        with opener.open(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _check_connection(self):
        try:
            data = self._send_request("/api/tags")
            models = [m.get("name") for m in data.get("models", [])]
            print(f"[OllamaVerifier] Успешное подключение к {self.ollama_url}. Модель: '{self.model_name}' (num_ctx={self.num_ctx})")
            if not any(self.model_name in m for m in models):
                print(f"[OllamaVerifier] Внимание: модель '{self.model_name}' не найдена в списке {models}.")
        except Exception as e:
            print(f"[OllamaVerifier] Внимание: сервер Ollama недоступен ({e}).")

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

    def _generate(self, messages):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": self.max_new_tokens,
                "num_ctx": self.num_ctx
            }
        }
        try:
            result = self._send_request("/api/chat", payload)
            return result.get("message", {}).get("content", "")
        except Exception as e:
            print(f"\n[Ошибка Ollama API]: {e}")
            return ""

    def verify(self, premise, hypothesis, bert_label, bert_proba, context=None):
        messages = self._build_prompt(premise, hypothesis, bert_label, context)
        reasoning = self._generate(messages)
        label, parsed_label_str = self._parse_response(reasoning, bert_label)
        confidence = 1.0 if parsed_label_str is not None else float(max(bert_proba))
        return label, confidence, reasoning

    def verify_batch(self, candidates, context=None):
        results = []
        total = len(candidates)
        for idx, cand in enumerate(candidates, 1):
            print(f"  LLM верификация: {idx}/{total} ...", end=" ", flush=True)
            result = self.verify(cand["premise"], cand["hypothesis"], cand["bert_label"], cand["bert_proba"], context)
            print(f"-> {_ID_TO_LABEL[result[0]]}")
            results.append(result)
        return results
