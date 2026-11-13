"""Измерение времени, точности и характеристик вычислительной среды."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np


def measure(
    fn: Callable,
    setup: Callable,
    budget: float = 1.0,
    min_repeats: int = 3,
    max_repeats: int = 7,
    raw: bool = False,
    target_batch_seconds: float = 0.002,
    max_batch_size: int = 32,
) -> dict:
    """Измерить ``fn(*setup())``.

    Подготовка аргументов не входит во время решения. Первый вызов является
    прогревочным и не входит в статистику. Долгие запуски не повторяются
    принудительно: возвращается одна выборка и ``repeats=1``.

    Важно: передаваемая функция не должна изменять аргументы. Если для одного
    запуска требуется новая матрица/вектор, это должно выполняться в ``setup``.
    """
    if not callable(fn) or not callable(setup):
        raise TypeError("решатель и функция подготовки данных должны быть вызываемыми объектами")
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("время, отведённое на измерения, должно быть положительным конечным числом")
    if (not isinstance(min_repeats, int) or not isinstance(max_repeats, int)
            or min_repeats < 1 or max_repeats < min_repeats):
        raise ValueError("минимальное и максимальное числа повторов должны быть целыми и согласованными: 1 <= min <= max")
    if (not np.isfinite(target_batch_seconds) or target_batch_seconds <= 0
            or not isinstance(max_batch_size, int) or max_batch_size < 1):
        raise ValueError("целевое время серии измерений и максимальное число объединяемых решений заданы некорректно")

    initial_args = setup()
    was_enabled = gc.isenabled()
    gc.collect()
    try:
        start = time.perf_counter()
        result = fn(*initial_args)
        warmup_time = time.perf_counter() - start
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()

    if warmup_time >= budget / min_repeats:
        samples = [warmup_time]
        number = 1
        repeats = 1
    else:
        number = 1 if warmup_time >= target_batch_seconds else min(
            max_batch_size,
            max(1, int(math.ceil(target_batch_seconds / max(warmup_time, 1e-12)))),
        )
        estimated = max(warmup_time * number, 1e-9)
        repeats = max(min_repeats, min(max_repeats, int(budget / estimated)))
        samples = []
        for _ in range(repeats):
            batch = [setup() for _ in range(number)]
            gc.collect()
            previous_gc_state = gc.isenabled()
            gc.disable()
            try:
                start = time.perf_counter()
                for args in batch:
                    result = fn(*args)
                elapsed = time.perf_counter() - start
            finally:
                if previous_gc_state:
                    gc.enable()
                else:
                    gc.disable()
            samples.append(elapsed / number)

    q1, median, q3 = np.quantile(np.asarray(samples, dtype=np.float64), [0.25, 0.5, 0.75])
    output = {
        "time_median": float(median),
        "time_min": float(min(samples)),
        "time_q1": float(q1),
        "time_q3": float(q3),
        "time_mean": float(statistics.fmean(samples)),
        "time_std": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        "repeats": len(samples),
        "number": int(number),
        "warmup_time": float(warmup_time),
        "result": result,
    }
    if raw:
        output["times"] = [float(value) for value in samples]
    return output


def rel_error(x, x_true) -> float:
    """Относительная прямая ошибка ``||x-x_true||₂ / ||x_true||₂``."""
    solution = np.asarray(x, dtype=np.float64)
    exact = np.asarray(x_true, dtype=np.float64)
    if solution.shape != exact.shape:
        raise ValueError("x и x_true должны иметь одинаковую форму")
    denominator = float(np.linalg.norm(exact))
    numerator = float(np.linalg.norm(solution - exact))
    return numerator / denominator if denominator > 0 else numerator


def rel_residual(A, x, b) -> float:
    """Относительная невязка ``||b-Ax||₂ / ||b||₂``."""
    rhs = np.asarray(b, dtype=np.float64)
    solution = np.asarray(x, dtype=np.float64)
    residual = rhs - A @ solution
    denominator = float(np.linalg.norm(rhs))
    numerator = float(np.linalg.norm(residual))
    return numerator / denominator if denominator > 0 else numerator


def backward_error(A, x, b) -> float:
    """Нормовая обратная ошибка в согласованной бесконечной норме.

    ``||b-Ax||_inf / (||A||_inf*||x||_inf + ||b||_inf)``.
    Для нескольких правых частей матричная бесконечная норма используется
    одинаково в числителе и знаменателе.
    """
    rhs = np.asarray(b, dtype=np.float64)
    solution = np.asarray(x, dtype=np.float64)
    residual = rhs - A @ solution
    numerator = float(np.linalg.norm(residual, ord=np.inf))
    try:
        import scipy.sparse as sp
        if sp.issparse(A):
            row_sums = np.asarray(abs(A).sum(axis=1), dtype=np.float64).ravel()
            anorm = float(row_sums.max(initial=0.0))
        else:
            anorm = float(np.linalg.norm(A, ord=np.inf))
    except Exception:
        anorm = float("nan")
    denominator = (anorm * float(np.linalg.norm(solution, ord=np.inf))
                   + float(np.linalg.norm(rhs, ord=np.inf)))
    return numerator / denominator if denominator > 0 and np.isfinite(denominator) else numerator


def fit_power(ns, ts) -> tuple[float, float]:
    """Оценить ``t = c*n**p`` линейной регрессией в log-log координатах."""
    sizes = np.asarray(ns, dtype=np.float64)
    times = np.asarray(ts, dtype=np.float64)
    mask = np.isfinite(sizes) & np.isfinite(times) & (sizes > 0) & (times > 0)
    if mask.sum() < 2:
        raise ValueError("для степенной аппроксимации нужны минимум две положительные точки")
    slope, intercept = np.polyfit(np.log(sizes[mask]), np.log(times[mask]), 1)
    return float(slope), float(np.exp(intercept))


def environment() -> dict:
    """Версии библиотек, сведения о процессоре, операционной системе и вычислительных потоках."""
    import scipy
    info = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count_logical": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": importlib.metadata.version("pandas"),
        "matplotlib": importlib.metadata.version("matplotlib"),
        "threadpoolctl": importlib.metadata.version("threadpoolctl"),
    }
    try:
        from threadpoolctl import threadpool_info
        info["threadpools"] = threadpool_info()
    except ImportError:
        info["threadpools"] = []
    try:
        info["psutil"] = importlib.metadata.version("psutil")
    except importlib.metadata.PackageNotFoundError:
        info["psutil"] = None
    try:
        import psutil
        info["ram_bytes"] = int(psutil.virtual_memory().total)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        info["ram_bytes"] = None
        info["cpu_count_physical"] = None
    return info


def save_environment(path: str | os.PathLike, **metadata) -> dict:
    info = environment()
    info.update(metadata)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    return info
