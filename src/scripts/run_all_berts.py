import os
import sys
import json
import subprocess
import datetime

# Добавляем корень проекта в sys.path для корректного импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.model_registry import MODEL_REGISTRY, get_model_path

DATASET_PATH = "data/validation/mfava_ru_sentences_test.csv"
OUTPUT_BASE_DIR = "outputs/validation/just_bert"

def get_base_architecture(model_path):
    """
    Извлекает оригинальное имя архитектуры из файла config.json.
    """
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("_name_or_path", "unknown_base_architecture")
        except Exception as e:
            return f"error_reading_config: {e}"
    return "config.json_not_found"

def main():
    print(f"Создание базовой директории: {OUTPUT_BASE_DIR}")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    if not os.path.exists(DATASET_PATH):
        print(f"ОШИБКА: Датасет не найден по пути {DATASET_PATH}")
        return

    for slug, rel_path in MODEL_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"ОБРАБОТКА МОДЕЛИ: {slug}")
        print(f"{'='*60}")
        
        model_path = get_model_path(slug)
        if not os.path.exists(model_path):
            print(f"ПРЕДУПРЕЖДЕНИЕ: Путь {model_path} не найден. Пропуск модели {slug}.")
            continue
            
        # 1. Извлечение базовой архитектуры
        base_arch = get_base_architecture(model_path)
        print(f"Базовая архитектура: {base_arch}")
        
        # 2. Создание директории для результатов модели
        model_out_dir = os.path.join(OUTPUT_BASE_DIR, slug)
        os.makedirs(model_out_dir, exist_ok=True)
        
        # 3. Генерация и сохранение метаданных (meta.json)
        meta = {
            "model_slug": slug,
            "local_weights_path": model_path,
            "base_architecture": base_arch,
            "validation_dataset_path": DATASET_PATH,
            "experiment_type": "Baseline NLI Evaluation (No LLM Cascade)",
            "description": "Оценка базовой модели для выбора оптимального роутера.",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        meta_path = os.path.join(model_out_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
        print(f"Создан файл метаданных: {meta_path}")
            
        # Использование жесткого пути к виртуальному окружению
        python_exe = os.path.join("venv", "Scripts", "python.exe")
        
        # 4. Запуск инференса (LLM каскад отключен)
        print(f"Запуск инференса (validate_on_mfava.py)...")
        infer_cmd = [
            python_exe, "-m", "src.scripts.validate_on_mfava",
            "--data_path", DATASET_PATH,
            "--model", slug,
            "--save_dir", model_out_dir
        ]
        
        try:
            # Параметр check=True вызывает исключение в случае ненулевого кода возврата
            subprocess.run(infer_cmd, check=True)
            print("Инференс завершен успешно.")
        except subprocess.CalledProcessError as e:
            print(f"ОШИБКА при выполнении инференса для модели {slug}: {e}")
            continue  # Переход к следующей модели в случае ошибки
            
        # 5. Расчет метрик и сохранение отчета
        print(f"Расчет метрик (calculate_validation_metrics.py)...")
        preds_path = os.path.join(model_out_dir, "validation_predictions.csv")
        metrics_cmd = [
            python_exe, "-m", "src.scripts.calculate_validation_metrics",
            "--predictions_path", preds_path
        ]
        
        try:
            # Идеальное решение: ОС сама перенаправит вывод напрямую в файл (минуя Python).
            # stderr=subprocess.STDOUT означает, что ошибки запишутся в этот же текстовый файл.
            metrics_out_path = os.path.join(model_out_dir, "metrics_report.txt")
            with open(metrics_out_path, "w", encoding="utf-8") as f_out:
                subprocess.run(metrics_cmd, stdout=f_out, stderr=subprocess.STDOUT, check=True)
                
            print(f"Отчет по метрикам сохранен: {metrics_out_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"ОШИБКА при расчете метрик для {slug}. Лог ошибки сохранен внутри {metrics_out_path}")
            
    print(f"\n{'='*60}")
    print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО.")
    print(f"Результаты сохранены в директории: {OUTPUT_BASE_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
