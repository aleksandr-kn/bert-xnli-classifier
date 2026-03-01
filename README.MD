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

## Installation
TODO: будет создан requirements.txt
```bash
pip install -r requirements.txt