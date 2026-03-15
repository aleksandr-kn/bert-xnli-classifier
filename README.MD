# BERT XNLI Classifier

Простой эксперимент с использованием BERT-based модели для NLP задачи NLI на датасете XNLI.

## Описание
Проект обучает и оценивает модель BERT (DeepPavlov/rubert-base-cased или подобную) для классификации пар предложений по одному из трёх классов: **entailment** (следование), **contradiction** (противоречие) или **neutral** (нейтральное). 
Цель - получить базовые результаты метрик и дальше решать что с этим делать.

## Возможности
- Использует библиотеку HuggingFace Transformers
- XNLI dataset грузится через модуль `datasets`
- Скрипт обучения и оценки
- Базовая модель BERT (например: `DeepPavlov/rubert-base-cased`)
- визуализация метрик

## Self-Improving Cascade

Итеративный цикл улучшения BERT с помощью LLM-верификатора (Qwen 7B).

### Идея

BERT быстро классифицирует все O(n^2) пары предложений, но допускает ошибки. LLM точнее, но слишком медленная для всех пар. Каскад соединяет их:

1. BERT предсказывает метки для всех пар в корпусе
2. LLM проверяет кандидатов (например, только contradiction-пары)
3. Расхождения (LLM не согласилась с BERT) собираются как **hard negatives**
4. BERT дообучается на оригинальном XNLI + все накопленные hard negatives
5. Повторяем, пока BERT не сойдётся к качеству LLM

### Условия остановки

- LLM не нашла ни одного расхождения с BERT (полная сходимость)
- Delta F1 между итерациями < порога (улучшения исчерпаны)
- Достигнут лимит итераций

### Структура модуля

```
src/cascade/
  collector.py      - сбор hard negatives (BERT + LLM -> расхождения)
  trainer.py         - дообучение BERT на расширенном датасете
  engine.py          - оркестрация итеративного цикла
  visualization.py   - графики конвергенции для диссертации
```

### Запуск

```bash
python -m src.scripts.run_cascade
```

Конфигурация в `src/scripts/run_cascade.py`: начальная модель, корпус текстов, LLM, количество итераций.

### Артефакты

```
outputs/cascade/
  hard_negatives/    - parquet-файлы с hard negatives по итерациям
  cascade_log.json   - лог с метриками и конфигурацией
  plots/             - графики конвергенции
```

## Installation
TODO: будет создан requirements.txt
```bash
pip install -r requirements.txt