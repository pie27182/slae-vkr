"""Эксперименты на локальных матрицах из коллекции SuiteSparse.

Примеры:
    python -m experiments.real_matrices --seed 20261113
    python -m experiments.real_matrices --only nos4 bcsstk14
    python -m experiments.real_matrices --report

Симметричность сама по себе не доказывает положительную определённость.
Метод сопряжённых градиентов и разложение Холецкого применяются к реальным
матрицам только тогда, когда положительная определённость указана в манифесте
и матрица прошла проверку симметричности. Если свойство неизвестно, эти методы
отмечаются как неприменимые.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import secrets
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

from experiments import common
from slae import bench as B
from slae import direct as D
from slae import iterative as I

ROOT = common.ROOT
OK = {"solved", "converged"}


def load_matrix(path: str | Path) -> sp.csr_array:
    """Загрузить квадратную вещественную матрицу и преобразовать её в разреженный формат с построчным хранением."""
    try:
        matrix = sio.mmread(str(path), spmatrix=False)
    except TypeError:  # совместимость с версиями SciPy без spmatrix
        matrix = sio.mmread(str(path))
    if np.iscomplexobj(matrix.data if sp.issparse(matrix) else matrix):
        raise ValueError("комплексные матрицы не поддерживаются")
    matrix = sp.csr_array(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("матрица должна быть непустой и квадратной")
    if not np.isfinite(matrix.data).all():
        raise ValueError("матрица содержит нечисловые значения или бесконечности")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _metadata_bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "spd", "positive definite", "positive_definite"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def load_manifest(data_dir: str | Path) -> dict[str, dict]:
    path = Path(data_dir) / "manifest.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["name"]: row for row in csv.DictReader(handle) if row.get("name")}


def properties(name: str, A, metadata: dict | None = None) -> dict:
    metadata = metadata or {}
    n = A.shape[0]
    diagonal = np.asarray(A.diagonal(), dtype=np.float64)
    absolute = abs(A)
    scale = float(absolute.max()) if A.nnz else 0.0
    difference = abs(A - A.T)
    asymmetry = float(difference.max()) if difference.nnz else 0.0
    symmetric = asymmetry <= 100 * np.finfo(float).eps * max(scale, np.finfo(float).tiny)
    off_diagonal = np.asarray(absolute.sum(axis=1)).ravel() - np.abs(diagonal)
    positive_definite_metadata = _metadata_bool(metadata.get("posdef", ""))
    return {
        "matrix": name,
        "group": metadata.get("group", ""),
        "area": metadata.get("kind", ""),
        "n": n,
        "nnz": int(A.nnz),
        "nnz_per_row": float(A.nnz / n),
        "symmetric": bool(symmetric),
        "symmetry_error_inf": asymmetry,
        "positive_definite_metadata": positive_definite_metadata if positive_definite_metadata is not None else "unknown",
        "zero_diagonal": int(np.count_nonzero(diagonal == 0.0)),
        "diag_dominant_rows_fraction": float(np.mean(np.abs(diagonal) > off_diagonal)),
    }


def _right_hand_side_seed(initial_seed: int, name: str) -> int:
    """Получить независимое и воспроизводимое начальное значение генератора правой части для заданной матрицы."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") ^ int(initial_seed)


def file_sha256(path: str | Path) -> str:
    """Хешировать исходный файл матрицы потоково, не загружая файл второй раз в оперативную память."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def methods(A, props, time_limit: float, max_dense_n: int, max_lu_nnz: int,
            max_pure_n: int):
    """Вернуть описание тестов: метод, семейство, функция, формат аргументов, применимость."""
    n = props["n"]
    symmetric = props["symmetric"]
    is_positive_definite = props["positive_definite_metadata"] is True and symmetric
    nonzero_diagonal = props["zero_diagonal"] == 0
    diagonal = np.asarray(A.diagonal(), dtype=np.float64)
    positive_diagonal = bool(np.all(diagonal > 0.0))
    iterative_kw = {"tol": common.TOL, "maxiter": 100_000, "time_limit": time_limit}
    entries = []

    if n <= max_pure_n:
        dense = A.toarray()
        entries.append(("Метод Гаусса (Python)", "direct_dense", D.gauss_pure,
                        (dense.tolist(),), "list", True, ""))
    if n <= max_dense_n:
        dense = A.toarray()
        entries.extend([
            ("Метод Гаусса (NumPy)", "direct_dense", D.gauss_numpy, (dense,), "dense", True, ""),
            ("Решатель NumPy", "direct_dense", D.solve_numpy, (dense,), "dense", True, ""),
            ("QR-разложение (SciPy)", "direct_dense", D.solve_qr_scipy, (dense,), "dense", True, ""),
            ("Метод наименьших квадратов (SciPy)", "direct_dense", D.solve_svd_lstsq, (dense,), "dense", True, ""),
        ])
        if is_positive_definite:
            entries.append(("Разложение Холецкого (SciPy)", "direct_dense", D.solve_cholesky_scipy,
                            (dense,), "dense", True, ""))
        else:
            entries.append(("Разложение Холецкого (SciPy)", "direct_dense", None, (), "dense", False,
                            "положительная определённость не подтверждена манифестом"))
    else:
        entries.append(("Плотные решатели", "direct_dense", None, (), "dense", False,
                        f"размер матрицы превышает установленный предел {max_dense_n}"))

    entries.append(("Разреженное LU-разложение (SciPy)", "direct_sparse", D.solve_sparse_lu, (A,), "sparse",
                    A.nnz <= max_lu_nnz,
                    "число ненулевых элементов превышает установленный предел" if A.nnz > max_lu_nnz else ""))
    entries.append(("Якоби", "stationary", lambda mat, rhs: I.jacobi(mat, rhs, **iterative_kw),
                    (A,), "sparse", nonzero_diagonal,
                    "нулевые элементы диагонали" if not nonzero_diagonal else ""))
    entries.append(("Гаусс — Зейдель", "stationary",
                    lambda mat, rhs: I.gauss_seidel(mat, rhs, **iterative_kw),
                    (A,), "sparse", nonzero_diagonal,
                    "нулевые элементы диагонали" if not nonzero_diagonal else ""))
    if is_positive_definite:
        entries.extend([
            ("Метод сопряжённых градиентов", "krylov", lambda mat, rhs: I.cg(mat, rhs, **iterative_kw),
             (A,), "sparse", True, ""),
            ("Метод сопряжённых градиентов с диагональным предобуславливанием", "krylov", lambda mat, rhs: I.cg(
                mat, rhs, M_solve=lambda r: r / diagonal, **iterative_kw),
             (A,), "sparse", positive_diagonal,
             "диагональ предобусловливателя не положительна" if not positive_diagonal else ""),
            ("Метод сопряжённых градиентов с симметричным предобуславливанием", "krylov", lambda mat, rhs: I.cg(
                mat, rhs, M_solve=I.symmetric_successive_over_relaxation_preconditioner(mat, 1.0), **iterative_kw),
             (A,), "sparse", positive_diagonal,
             "для симметричного предобуславливателя требуется положительная диагональ" if not positive_diagonal else ""),
        ])
    else:
        reason = "положительная определённость матрицы не подтверждена; одной симметричности недостаточно"
        entries.extend([
            ("Метод сопряжённых градиентов", "krylov", None, (), "sparse", False, reason),
            ("Метод сопряжённых градиентов с диагональным предобуславливанием", "krylov", None, (), "sparse", False, reason),
            ("Метод сопряжённых градиентов с симметричным предобуславливанием", "krylov", None, (), "sparse", False, reason),
        ])
    entries.extend([
        ("Стабилизированный метод бисопряжённых градиентов (SciPy)", "krylov", lambda mat, rhs: I.scipy_bicgstab(
            mat, rhs, tol=common.TOL, maxiter=100_000, time_limit=time_limit),
         (A,), "sparse", True, ""),
        ("Обобщённый метод минимальной невязки (SciPy)", "krylov", lambda mat, rhs: I.scipy_gmres(
            mat, rhs, tol=common.TOL, maxiter=10_000, restart=50, time_limit=time_limit),
         (A,), "sparse", True, ""),
        ("Стабилизированный метод бисопряжённых градиентов с неполным LU-предобуславливанием", "krylov", lambda mat, rhs: I.scipy_bicgstab(
            mat, rhs, tol=common.TOL, maxiter=100_000, time_limit=time_limit,
            M=I.ilu_preconditioner(mat)),
         (A,), "sparse", True, ""),
        ("Обобщённый метод минимальной невязки с неполным LU-предобуславливанием", "krylov", lambda mat, rhs: I.scipy_gmres(
            mat, rhs, tol=common.TOL, maxiter=10_000, restart=50, time_limit=time_limit,
            M=I.ilu_preconditioner(mat)),
         (A,), "sparse", True, ""),
    ])
    return entries


def _attempt(fn, args):
    started = time.perf_counter()
    try:
        result = fn(*args)
        elapsed = time.perf_counter() - started
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            x, info = result
        else:
            x, info = result, {"status": "solved"}
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all():
            return None, {**info, "status": "non_finite"}, elapsed
        return x, info, elapsed
    except MemoryError:
        return None, {"status": "out_of_memory"}, time.perf_counter() - started
    except Exception as exc:
        return None, {"status": "exception", "error": f"{type(exc).__name__}: {exc}"[:240]}, time.perf_counter() - started


def run_matrix(name, A, props, initial_seed, time_limit, budget, rows,
               max_dense_n, max_lu_nnz, max_pure_n):
    right_hand_side_seed = _right_hand_side_seed(initial_seed, name)
    rng = np.random.default_rng(right_hand_side_seed)
    x_true = rng.standard_normal(A.shape[0])
    b = np.asarray(A @ x_true, dtype=np.float64)

    # Проверяем, что правая часть не потеряла масштаб из-за вырожденной матрицы.
    if not np.isfinite(b).all():
        raise ValueError("умножение матрицы на заданный вектор породило нечисловые значения или бесконечности")
    for method, family, fn, prefix_args, kind, applicable, reason in methods(
            A, props, time_limit, max_dense_n, max_lu_nnz, max_pure_n):
        row = {"matrix": name, "n": A.shape[0], "nnz": A.nnz,
               "method": method, "family": family, "right_hand_side_seed": right_hand_side_seed,
               "matrix_sha256": props.get("file_sha256")}
        if not applicable:
            rows.append({**row, "status": "n/a", "reason": reason})
            continue

        if kind == "list":
            args = (*prefix_args, b.tolist())
        else:
            args = (*prefix_args, b)
        x, info, elapsed = _attempt(fn, args)
        row.update(status=info.get("status", "solved"), iters=info.get("iters"),
                   time=elapsed, solver_residual=info.get("resid"),
                   error_message=info.get("error"))
        if x is not None:
            row["relative_error_to_constructed_x"] = B.rel_error(x, x_true)
            row["relative_residual"] = B.rel_residual(A, x, b)
            row["backward_error"] = B.backward_error(A, x, b)
            row["solver_status"] = row["status"]
            if row["status"] in {"solved", "converged"} and (
                    not np.isfinite(row["relative_residual"]) or row["relative_residual"] > common.TOL):
                row["status"] = "verification_failed"
            # Повторные измерения выполняются только для быстрых прямых методов.
            if family.startswith("direct") and row["status"] in {"solved", "converged"} and elapsed < budget:
                try:
                    measurement = B.measure(fn, lambda: args, budget=budget, min_repeats=2)
                    row.update(common.stats(measurement))
                    measured_x = np.asarray(measurement["result"], dtype=np.float64)
                    row["relative_residual"] = B.rel_residual(A, measured_x, b)
                    row["relative_error_to_constructed_x"] = B.rel_error(measured_x, x_true)
                    row["backward_error"] = B.backward_error(A, measured_x, b)
                    if (not np.isfinite(row["relative_residual"])
                            or row["relative_residual"] > common.TOL):
                        row["status"] = "verification_failed"
                except Exception as exc:
                    row["repeat_measurement_error"] = f"{type(exc).__name__}: {exc}"[:240]
        rows.append(row)
        print(f"  {method:<52} {REPORT_STATUS_LABELS.get(str(row['status']), row['status']):<34} {elapsed:9.3g} с"
              + (f"  итераций {row['iters']}" if row.get("iters") is not None else "")
              + (f"  невязка {row['relative_residual']:.1e}" if row.get("relative_residual") is not None else ""),
              flush=True)


REPORT_STATUS_LABELS = {
    "solved": "решение прошло проверку",
    "converged": "достигнута сходимость",
    "max iterations": "достигнут предел числа итераций",
    "time limit": "превышено ограничение времени",
    "diverged": "итерационный процесс расходится",
    "breakdown": "численный сбой метода",
    "verification_failed": "независимая проверка невязки не пройдена",
    "exception": "ошибка выполнения",
    "n/a": "метод неприменим",
    "out_of_memory": "недостаточно оперативной памяти",
    "not_available": "необходимые возможности библиотеки недоступны",
    "not_available": "необходимые возможности библиотеки недоступны",
    "non_finite": "получены неконечные значения",
    "loaded": "матрица загружена и проверена",
    "load_error": "ошибка чтения или проверки файла матрицы",
    "matrix_run_error": "ошибка обработки матрицы",
}


MATRIX_COLUMN_LABELS = {
    "matrix": "Имя матрицы",
    "group": "Группа в коллекции SuiteSparse",
    "area": "Область применения по каталогу",
    "n": "Размер матрицы",
    "nnz": "Число ненулевых элементов",
    "nnz_per_row": "Среднее число ненулевых элементов на строку",
    "symmetric": "Матрица симметрична",
    "symmetry_error_inf": "Отклонение от симметричности",
    "positive_definite_metadata": "Положительная определённость по каталогу",
    "zero_diagonal": "Число нулей на главной диагонали",
    "diag_dominant_rows_fraction": "Доля строк с диагональным преобладанием",
    "file_sha256": "Контрольная сумма файла SHA-256",
    "status": "Статус файла",
    "error": "Причина ошибки",
}

def _translate_report_status(value):
    return REPORT_STATUS_LABELS.get(str(value), str(value))


def _display_matrix_frame(frame: pd.DataFrame) -> pd.DataFrame:
    visible = frame.copy()
    if "status" in visible:
        visible["status"] = visible["status"].map(_translate_report_status)
    for column in ("symmetric", "positive_definite_metadata"):
        if column in visible:
            visible[column] = visible[column].map(lambda value: {
                "True": "да", "False": "нет", "true": "да", "false": "нет",
                "unknown": "не указано", "": "не указано",
            }.get(str(value), value))
    return visible.rename(columns={key: value for key, value in MATRIX_COLUMN_LABELS.items()
                                   if key in visible.columns})


def report(results_dir: Path, figures_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    real_path = results_dir / "real.csv"
    matrices_path = results_dir / "matrices.csv"
    if not real_path.exists() or not matrices_path.exists():
        raise FileNotFoundError("для отчёта нужны real.csv и matrices.csv")
    results = pd.read_csv(real_path, keep_default_na=False)
    matrices = pd.read_csv(matrices_path, keep_default_na=False)
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Старые изображения не должны оставаться в отчёте, если текущая выборка
    # не загрузилась или не дала ни одного решения.
    (figures_dir / "real_status.png").unlink(missing_ok=True)
    (figures_dir / "real_time.png").unlink(missing_ok=True)
    if matrices.empty:
        (results_dir / "summary.md").write_text("# Реальные матрицы\n\nНет данных о матрицах.\n", encoding="utf-8")
        return
    if results.empty or "method" not in results.columns:
        summary = ["# Эксперименты на реальных матрицах\n", "Нет строк с результатами решателей.\n",
                   "## Свойства и статусы загрузки матриц\n",
                   _display_matrix_frame(matrices).to_markdown(index=False)]
        (results_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        return

    # Строки с ошибкой чтения не содержат размер и число ненулевых элементов.
    # Они остаются в таблицах, но не участвуют в построении шкал графиков.
    valid_matrices = matrices.copy()
    valid_matrices["n_numeric"] = pd.to_numeric(valid_matrices.get("n"), errors="coerce")
    valid_matrices["nnz_numeric"] = pd.to_numeric(valid_matrices.get("nnz"), errors="coerce")
    valid_matrices = valid_matrices[valid_matrices.n_numeric.notna()]
    matrix_order = valid_matrices.sort_values("nnz_numeric").matrix.tolist()
    method_order = list(dict.fromkeys(results.method))
    statuses = results.pivot_table(index="matrix", columns="method", values="status", aggfunc="first")

    fig, ax = plt.subplots(figsize=(max(11, 0.70 * len(method_order) + 5), max(7, 0.45 * len(matrix_order) + 3)))
    palette = {"solved": "#4c9a5b", "converged": "#4c9a5b", "max iterations": "#e0a43a",
               "time limit": "#e0a43a", "diverged": "#c8553d", "breakdown": "#c8553d",
               "verification_failed": "#8b0000", "exception": "#c8553d",
               "out_of_memory": "#8b0000", "non_finite": "#8b0000",
               "matrix_run_error": "#8b0000", "n/a": "#dddddd"}
    result_index = results.set_index(["matrix", "method"])
    for i, matrix_name in enumerate(matrix_order):
        for j, method in enumerate(method_order):
            if (matrix_name, method) not in result_index.index:
                continue
            row = result_index.loc[(matrix_name, method)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            status = str(row.get("status", "unknown"))
            color = palette.get(status, "#999999")
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white"))
            elapsed = pd.to_numeric(pd.Series([row.get("time")]), errors="coerce").iloc[0]
            label = f"{elapsed:.2g}" if status in OK and pd.notna(elapsed) else ""
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=6)
    ax.set_xlim(0, max(1, len(method_order)))
    ax.set_ylim(max(1, len(matrix_order)), 0)
    ax.set_xticks(np.arange(len(method_order)) + 0.5, method_order, rotation=60, ha="right", fontsize=7)
    valid_index = valid_matrices.set_index("matrix")
    y_labels = [f"{name} (n={int(valid_index.loc[name, 'n_numeric']):,})" for name in matrix_order]
    ax.set_yticks(np.arange(len(matrix_order)) + 0.5, y_labels, fontsize=7)
    ax.set_title("Результаты методов на реальных матрицах\nВ ячейках указано время успешного решения, с")
    status_values = set(results.get("status", pd.Series(dtype=str)).astype(str))
    handles = [Patch(facecolor=palette[status], label=_translate_report_status(status))
               for status in palette if status in status_values]
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, fontsize=7, title="Результат")
    fig.subplots_adjust(left=0.10, right=0.78, bottom=0.48, top=0.88)
    fig.savefig(figures_dir / "real_status.png", dpi=160)
    plt.close(fig)

    successful = results[results.status.isin(OK)].copy()
    if "time" not in successful.columns:
        successful["time"] = np.nan
    successful["time"] = pd.to_numeric(successful.time, errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 5))
    x_positions = {name: index for index, name in enumerate(matrix_order)}
    for method, frame in successful.groupby("method", sort=False):
        frame = frame[frame.time > 0]
        frame = frame[frame.matrix.isin(x_positions)]
        if not frame.empty:
            ax.plot([x_positions[name] for name in frame.matrix], frame.time,
                    marker="o", linewidth=1.0, markersize=4, label=method)
    ax.set_yscale("log")
    ax.set_xticks(range(len(matrix_order)), matrix_order, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Время, с")
    ax.set_title("Время успешных запусков")
    if ax.lines:
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "real_time.png", dpi=160)
    plt.close(fig)

    visible_statuses = statuses.reindex(matrix_order).reset_index().rename(columns={"matrix": "Имя матрицы"})
    for column in visible_statuses.columns[1:]:
        visible_statuses[column] = visible_statuses[column].map(_translate_report_status)
    summary = ["# Эксперименты на реальных матрицах\n",
               r"Правая часть формируется по правилу \(b=Ax\). Вектор \(x\) создаётся генератором случайных чисел. "
               "Исходные файлы проверяются по контрольной сумме, указанной в таблице свойств матриц.\n",
               "Относительную ошибку относительно заданного вектора нельзя считать ошибкой относительно "
               "единственного решения, если невырожденность матрицы не подтверждена.\n",
               "## Свойства матриц\n", _display_matrix_frame(matrices).to_markdown(index=False),
               "\n## Результаты методов\n", visible_statuses.to_markdown(index=False)]
    if not successful.empty:
        summary.extend(["\n## Время успешных запусков, с\n",
                        successful.pivot_table(index="matrix", columns="method", values="time")
                        .reindex(matrix_order).reset_index().rename(columns={"matrix": "Имя матрицы"})
                        .to_markdown(index=False)])
        summary.extend(["\n## Число успешных запусков по методу\n",
                        successful.groupby("method").size().rename("Число успешных запусков")
                        .reset_index().rename(columns={"method": "Метод"}).to_markdown(index=False)])
    else:
        summary.append("\nУспешных запусков нет.\n")
    (results_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Отчёт: {results_dir / 'summary.md'}; графики: {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "data" / "matrices"),
                        help="каталог с исходными файлами матриц в формате Matrix Market")
    parser.add_argument("--out", default=str(ROOT / "results_real"),
                        help="каталог для таблиц результатов")
    parser.add_argument("--figures", default=str(ROOT / "figures_real"),
                        help="каталог для графиков")
    parser.add_argument("--only", nargs="*", default=None, metavar="ИМЯ",
                        help="рассчитать только перечисленные матрицы")
    parser.add_argument("--all-local", action="store_true",
                        help="включить все локальные файлы .mtx, в том числе отсутствующие в реестре загрузок")
    parser.add_argument("--time-limit", type=float, default=30.0,
                        help="предельное время одного запуска решателя, в секундах (по умолчанию 30)")
    parser.add_argument("--budget", type=float, default=1.0,
                        help="время, отводимое на повторные измерения прямого метода, в секундах (по умолчанию 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="начальное значение генератора; случайное значение по умолчанию сохраняется в environment.json")
    parser.add_argument("--max-dense-n", type=int, default=2000,
                        help="максимальный размер матрицы для плотных решателей")
    parser.add_argument("--max-pure-n", type=int, default=250,
                        help="максимальный размер матрицы для учебного решателя на Python")
    parser.add_argument("--max-lu-nnz", type=int, default=1_000_000,
                        help="максимальное число ненулевых элементов для разреженного LU-разложения")
    parser.add_argument("--report", action="store_true",
                        help="перестроить отчёт по уже созданным таблицам, не запуская решатели")
    args = parser.parse_args()
    if args.time_limit <= 0 or args.budget <= 0 or args.max_dense_n < 1 or args.max_pure_n < 1 or args.max_lu_nnz < 1:
        parser.error("лимиты должны быть положительными")
    data_dir, out_dir, figures_dir = Path(args.data), Path(args.out), Path(args.figures)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.report:
        report(out_dir, figures_dir)
        return
    if any(out_dir.iterdir()):
        parser.error(f"каталог результатов не пустой: {out_dir}. Используйте clean_results "
                     "или задайте новый каталог через --out.")

    files = sorted(data_dir.glob("*.mtx"))
    manifest = load_manifest(data_dir)
    if manifest and not args.all_local:
        selected_names = {name for name, row in manifest.items() if row.get("status") == "ok"}
        files = [path for path in files if path.stem in selected_names]
    if args.only:
        wanted = set(args.only)
        files = [path for path in files if path.stem in wanted]
        missing = wanted - {path.stem for path in files}
        if missing:
            parser.error(f"не найдены матрицы: {', '.join(sorted(missing))}")
    if not files:
        parser.error(f"нет файлов .mtx в {data_dir}; сначала загрузите выбранные матрицы")

    seed = int(args.seed) if args.seed is not None else secrets.randbits(32)
    B.save_environment(out_dir / "environment.json", experiment="real_matrices",
                       initial_seed=seed, tolerance=common.TOL,
                       time_limit=args.time_limit, budget=args.budget)
    print(f"Начальное значение генератора для этого запуска: {seed}")
    rows: list[dict] = []
    matrix_rows: list[dict] = []
    for path in files:
        name = path.stem
        try:
            A = load_matrix(path)
            props = properties(name, A, manifest.get(name, {}))
            props["file_sha256"] = file_sha256(path)
            expected_hash = str(manifest.get(name, {}).get("sha256", "")).strip().lower()
            if expected_hash and props["file_sha256"].lower() != expected_hash:
                raise ValueError("контрольная сумма исходного файла не совпадает с записью в реестре загрузок")
        except Exception as exc:
            print(f"{name}: пропущена: {type(exc).__name__}: {exc}")
            matrix_rows.append({"matrix": name, "status": "load_error",
                                "error": f"{type(exc).__name__}: {exc}"[:240]})
            pd.DataFrame(rows, columns=["matrix", "n", "nnz", "method", "family", "right_hand_side_seed",
                                        "matrix_sha256",
                                        "status", "time", "iters", "solver_residual",
                                        "relative_error_to_constructed_x", "relative_residual",
                                        "backward_error", "error_message"]).to_csv(out_dir / "real.csv", index=False)
            pd.DataFrame(matrix_rows).to_csv(out_dir / "matrices.csv", index=False)
            continue
        matrix_rows.append({**props, "status": "loaded"})
        print(f"\nМатрица {name}: размер={props['n']:,}, ненулевых элементов={props['nnz']:,}, "
              f"симметрична={props['symmetric']}, положительная определённость по каталогу={props['positive_definite_metadata']}", flush=True)
        try:
            run_matrix(name, A, props, seed, args.time_limit, args.budget, rows,
                       args.max_dense_n, args.max_lu_nnz, args.max_pure_n)
        except Exception as exc:
            rows.append({"matrix": name, "status": "matrix_run_error",
                         "error_message": f"{type(exc).__name__}: {exc}"[:240]})
        # Прогресс сохраняется после каждой матрицы, чтобы прерывание не удаляло уже готовые строки.
        pd.DataFrame(rows, columns=["matrix", "n", "nnz", "method", "family", "right_hand_side_seed",
                                    "matrix_sha256",
                                    "status", "time", "iters", "solver_residual",
                                    "relative_error_to_constructed_x", "relative_residual",
                                    "backward_error", "error_message"]).to_csv(out_dir / "real.csv", index=False)
        pd.DataFrame(matrix_rows).to_csv(out_dir / "matrices.csv", index=False)
    report(out_dir, figures_dir)


if __name__ == "__main__":
    main()
