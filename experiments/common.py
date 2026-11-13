"""Общие настройки, пути и запись результатов экспериментальных серий."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOL = 1e-8
STATUS_DISPLAY = {
    "solved": "решение прошло проверку",
    "converged": "достигнута сходимость",
    "max iterations": "достигнут предел числа итераций",
    "time limit": "превышено ограничение времени",
    "diverged": "итерационный процесс расходится",
    "breakdown": "численная ошибка метода",
    "verification_failed": "проверка невязки не пройдена",
    "exception": "ошибка выполнения",
    "n/a": "метод неприменим",
    "not_available": "необходимые возможности библиотеки недоступны",
    "out_of_memory": "недостаточно оперативной памяти",
    "non_finite": "получены нечисловые значения или бесконечности",
}
PY, NP, LIB, SPARSE = "Python", "NumPy", "SciPy", "SciPy (разреженные матрицы)"

# Размеры заданы до запуска и не подбираются по полученным временам.
# Сокращённый профиль предназначен только для проверки работы программы.
PROFILES = {
    "full": {
        "dense_sizes": [25, 50, 100, 200, 400, 800, 1200],
        "pure_max": 200,
        "numpy_gauss_max": 800,
        "svd_max": 800,
        "iter_sizes": [50, 100, 200, 400],
        "cond_n": 100,
        "cond_exponents": [0, 2, 4, 6, 8, 10, 12, 14],
        "cond_repeats": 5,
        "pivot_sizes": [25, 50, 100, 200, 400],
        "pivot_repeats": 10,
        "poisson_N": [16, 32, 64, 128, 256],
        "poisson_stationary_max_N": 64,
        "tri_sizes": [100, 300, 1000, 3000, 10000, 30000, 100000, 300000],
        "tri_dense_max": 3000,
        "sparse_sizes": [1000, 2000, 4000, 8000, 16000, 100000],
        "sparse_lu_max": 8000,
        "right_hand_side_n": 800,
        "right_hand_side_counts": [1, 2, 5, 10, 20, 50],
        "budget": 1.5,
    },
    "quick": {
        "dense_sizes": [20, 40, 80, 120],
        "pure_max": 80,
        "numpy_gauss_max": 120,
        "svd_max": 120,
        "iter_sizes": [20, 40, 80],
        "cond_n": 30,
        "cond_exponents": [0, 2, 4, 6, 8],
        "cond_repeats": 2,
        "pivot_sizes": [10, 20, 40],
        "pivot_repeats": 2,
        "poisson_N": [8, 16, 24],
        "poisson_stationary_max_N": 16,
        "tri_sizes": [50, 200, 1000, 5000],
        "tri_dense_max": 500,
        "sparse_sizes": [200, 500, 1000, 5000],
        "sparse_lu_max": 1000,
        "right_hand_side_n": 100,
        "right_hand_side_counts": [1, 2, 5],
        "budget": 0.25,
    },
}


class Recorder:
    """Накопить строки одной серии и атомарно записать CSV."""

    def __init__(self, name: str, output_dir: str | Path | None = None):
        self.name = name
        self.output_dir = Path(output_dir) if output_dir is not None else Path(RESULTS)
        self.rows: list[dict] = []
        self.started = time.perf_counter()
        print(f"\nСерия: {name}", flush=True)

    def add(self, **row):
        row.pop("result", None)
        row.pop("times", None)
        self.rows.append(row)
        elapsed = row.get("time_median")
        label = f"время={elapsed:.3e} с" if elapsed is not None and pd.notna(elapsed) else ""
        extras = []
        if row.get("iters") is not None:
            extras.append(f"итераций={row['iters']}")
        if row.get("rel_residual") is not None and pd.notna(row["rel_residual"]):
            extras.append(f"невязка={row['rel_residual']:.1e}")
        if row.get("status"):
            extras.append(f"результат={STATUS_DISPLAY.get(str(row['status']), row['status'])}")
        print(f"  {str(row.get('matrix', '')):<16} {str(row.get('method', '')):<30} "
              f"размер={row.get('n', '')!s:<8} {label} {' '.join(extras)}", flush=True)

    def save(self) -> pd.DataFrame:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.rows)
        path = self.output_dir / f"{self.name}.csv"
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        print(f"  -> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} "
              f"({len(frame)} строк, {time.perf_counter() - self.started:.1f} с)", flush=True)
        return frame


def stats(measurement: dict) -> dict:
    keys = ("time_median", "time_min", "time_q1", "time_q3", "time_mean",
            "time_std", "repeats", "number", "warmup_time")
    return {key: measurement[key] for key in keys if key in measurement}
