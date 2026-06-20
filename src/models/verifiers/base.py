# -*- coding: utf-8 -*-
import os
os.environ["HF_HOME"] = "F:/huggingface_cache"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
_ID_TO_LABEL = {v: k for k, v in _LABEL_TO_ID.items()}

class BaseLLMVerifier:
    """
    Базовый класс для всех LLM-верификаторов.
    Берет на себя загрузку модели в 4-bit, токенизацию и генерацию.
    """
    def __init__(self, model_name, device=None, max_new_tokens=512):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"": 0})
            print("LLM загружена в 4-bit квантизации.")
        except Exception as e:
            print(f"4-bit квантизация недоступна ({e}), загрузка в fp16...")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            print("LLM загружена в fp16.")

        self.model.eval()

    def _build_prompt(self, premise, hypothesis, bert_label, context=None):
        raise NotImplementedError()

    def _parse_response(self, text, fallback_label):
        raise NotImplementedError()

    def verify(self, premise, hypothesis, bert_label, bert_proba, context=None):
        messages = self._build_prompt(premise, hypothesis, bert_label, context)
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        reasoning = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
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
