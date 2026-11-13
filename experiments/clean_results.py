"""Безопасно удалить сгенерированные результаты и графики проекта.

По умолчанию исходные коды, тесты и загруженные матрицы не удаляются.
Добавьте --include-data, чтобы также удалить data/matrices.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ["results", "results_quick", "results_real", "figures", "figures_quick", "figures_real"]
GENERATED_FILES: list[str] = []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-data", action="store_true",
                        help="также удалить загруженные внешние матрицы и реестр загрузок")
    args = parser.parse_args()
    targets = [ROOT / name for name in GENERATED]
    targets.extend(ROOT / name for name in GENERATED_FILES)
    if args.include_data:
        targets.append(ROOT / "data" / "matrices")

    print("Будут удалены только следующие пути:")
    for target in targets:
        print(f"  {target}")
    for target in targets:
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    print("Очистка завершена. Исходный код и тесты сохранены.")


if __name__ == "__main__":
    main()
