"""
Скрипт парсинга задач из PDF-сборника
"""

import re
import csv
import pdfplumber

PDF_PATH = "data/pdf/logic_tasks.pdf"
OUTPUT_PATH = "data/pdf/logic_tasks.csv"

# Темы и диапазоны страниц (0-indexed), где начинаются задачи
THEMES = {
    1: "Предмет логики",
    2: "Основные законы логики",
    3: "Логика высказываний",
    4: "Имена",
    5: "Простой категорический силлогизм",
    6: "Практическое применение логики",
}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Извлекает весь текст из PDF."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)


def parse_tasks(text: str) -> list[dict]:
    """
    Парсит задачи из текста PDF.
    Ищет паттерны вида 'X.Y.' в начале строки или после пробела,
    где X — номер темы (1-6), Y — номер задачи.
    """
    # Паттерн: номер задачи вида X.Y. (или X.Y без точки для формул).
    # Допускаем артефакты OCR от водяного знака «Репозиторий ВГУ»:
    #   - лишние буквы (е, п, р, о, т и др.) перед номером или после второго числа
    # Примеры совпадений: "1.1. ", "е2.20. ", "1.38п. ", "3.3 ("
    task_pattern = re.compile(
        r'(?:^|\n)[^\S\n]*[а-яёa-z]?[^\S\n]*(\d+\.\d+)[а-яёп]?\.?\s*(?=\S)',
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(task_pattern.finditer(text))
    tasks = []

    for i, match in enumerate(matches):
        task_number = match.group(1)
        theme_num = int(task_number.split('.')[0])

        # Пропускаем, если номер темы вне диапазона 1-6
        if theme_num < 1 or theme_num > 6:
            continue

        # Начало текста задачи — сразу после номера
        start = match.end()

        # Конец текста — начало следующей задачи или конец текста
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        task_text = text[start:end].strip()

        # Убираем артефакты: номера страниц, водяные знаки
        task_text = re.sub(r'\n\s*\d+\s*$', '', task_text)  # номер страницы в конце
        task_text = re.sub(r'^\s*\d+\s*\n', '', task_text)  # номер страницы в начале

        # Убираем одиночные буквы-артефакты от водяного знака «Репозиторий ВГУ»
        # Только на концах строк (правый край страницы) — не убираем «и», «в», «о» и т.д.
        # между слов, т.к. это реальные русские слова-союзы/предлоги
        task_text = re.sub(r'\s+[А-Яа-яёЁ]\s*$', '', task_text, flags=re.MULTILINE)
        task_text = re.sub(r'^\s*[А-Яа-яёЁ]\s*\n', '', task_text, flags=re.MULTILINE)

        # Заменяем переносы строк на пробелы для однородности,
        # но оставляем двойные переносы как разделители абзацев
        task_text = re.sub(r'\n{2,}', '\n\n', task_text)
        task_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', task_text)

        # Убираем лишние пробелы
        task_text = re.sub(r'  +', ' ', task_text)
        # Убираем дефисы переноса слов
        task_text = re.sub(r'(\w)- (\w)', r'\1\2', task_text)
        task_text = task_text.strip()

        # Если текст пустой — пропускаем
        if not task_text:
            continue

        theme_name = THEMES.get(theme_num, "")

        tasks.append({
            "number": task_number,
            "theme": theme_name,
            "text": task_text,
        })

    return tasks


def save_to_csv(tasks: list[dict], output_path: str):
    """Сохраняет задачи в CSV файл."""
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["number", "theme", "text"])
        writer.writeheader()
        writer.writerows(tasks)


def main():
    print(f"Извлекаю текст из {PDF_PATH}...")
    text = extract_text_from_pdf(PDF_PATH)
    print(f"Извлечено {len(text)} символов.")

    print("Парсинг задач...")
    tasks = parse_tasks(text)
    print(f"Найдено {len(tasks)} задач.")

    # Выводим статистику по темам
    theme_counts = {}
    for t in tasks:
        theme_counts[t["theme"]] = theme_counts.get(t["theme"], 0) + 1
    for theme, count in theme_counts.items():
        print(f"  {theme}: {count} задач")

    print(f"\nПервые 5 задач:")
    for t in tasks[:5]:
        preview = t["text"][:80] + "..." if len(t["text"]) > 80 else t["text"]
        print(f"  [{t['number']}] {preview}")

    save_to_csv(tasks, output_path=OUTPUT_PATH)
    print(f"\nСохранено в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
