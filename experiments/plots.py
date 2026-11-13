"""Построение графиков и сводного отчёта по CSV текущего запуска."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import ROOT

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 7,
})

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "p"]


def load(directory: Path, name: str):
    path = directory / f"{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def md_table(frame: pd.DataFrame, float_format="{:.3g}") -> str:
    if frame.empty:
        return "(нет данных)"
    columns = list(frame.columns)
    rows = ["| " + " | ".join(map(str, columns)) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(float_format.format(float(value)))
            else:
                values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def save(fig, directory: Path, filename: str):
    fig.tight_layout()
    fig.savefig(directory / filename)
    plt.close(fig)


COLUMN_LABELS = {
    "matrix": "Матрица",
    "matrix_type": "Тип матрицы",
    "method": "Метод",
    "level": "Вариант реализации",
    "family": "Группа метода",
    "n": "Размер системы",
    "N": "Размер сетки",
    "nnz": "Число ненулевых элементов",
    "time": "Время, с",
    "time_median": "Медианное время, с",
    "time_min": "Минимальное время, с",
    "time_q1": "Первый квартиль времени, с",
    "time_q3": "Третий квартиль времени, с",
    "time_mean": "Среднее время, с",
    "time_std": "Стандартное отклонение времени, с",
    "iters": "Число итераций",
    "cond_target": "Заданное число обусловленности",
    "cond_actual": "Фактическое число обусловленности",
    "rel_error": "Относительная прямая ошибка",
    "rel_residual": "Относительная невязка",
    "backward_error": "Обратная ошибка",
    "status": "Результат проверки",
    "solver_status": "Статус решателя",
    "right_hand_side_count": "Число векторов правой части",
    "library_threads": "Число вычислительных потоков",
    "successful_runs": "Число успешных запусков",
    "n_repeats": "Число повторов",
    "count": "Число измерений",
    "median": "Медианное время, с",
    "min": "Минимальное время, с",
    "max": "Максимальное время, с",
    "std": "Стандартное отклонение времени, с",
    "run": "Номер повтора",
    "matrix_seed": "Начальное значение генератора матрицы",
    "right_hand_side_seed": "Начальное значение генератора правой части",
    "order_seed": "Начальное значение генератора порядка запусков",
}

VALUE_LABELS = {
    "general": "общая матрица",
    "symmetric_positive_definite": "симметричная положительно определённая матрица",
    "Hilbert": "матрица Гильберта",
    "random_normal": "случайная матрица с нормальным распределением элементов",
    "dense": "плотная матрица",
    "Poisson": "матрица дискретной задачи Пуассона",
    "direct_dense": "плотный прямой метод",
    "direct_sparse": "разреженный прямой метод",
    "stationary": "стационарный итерационный метод",
    "krylov": "метод подпространств Крылова",
    "direct": "прямой разреженный метод",
    "Python": "Python",
    "NumPy": "NumPy",
    "SciPy": "SciPy",
    "SciPy (разреженные матрицы)": "SciPy для разреженных матриц",
}


def add_summary(parts: list[str], title: str, frame: pd.DataFrame):
    visible = frame.copy()
    for column in ("status", "solver_status"):
        if column in visible.columns:
            visible[column] = visible[column].map(translate_status)
    for column in ("matrix_type", "family", "level"):
        if column in visible.columns:
            visible[column] = visible[column].map(lambda value: VALUE_LABELS.get(str(value), value))
    visible = visible.rename(columns={key: value for key, value in COLUMN_LABELS.items()
                                      if key in visible.columns})
    parts.extend([f"\n## {title}\n", md_table(visible)])


def plot_lines(frame: pd.DataFrame, x: str, y: str, group: str,
               directory: Path, filename: str, title: str, xlabel: str, ylabel: str):
    required = {x, y, group}
    if frame is None or not required.issubset(frame.columns):
        return
    data = frame.copy()
    # Времена неуспешных, прерванных и не прошедших проверку запусков
    # нельзя выдавать за время успешного решения.
    if "status" in data.columns:
        data = data[data.status.isin({"solved", "converged"})]
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data[np.isfinite(data[x]) & np.isfinite(data[y]) & (data[x] > 0) & (data[y] > 0)]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for index, (label, group_frame) in enumerate(data.groupby(group, sort=False)):
        group_frame = group_frame.sort_values(x)
        ax.plot(group_frame[x], group_frame[y], marker=MARKERS[index % len(MARKERS)],
                markersize=4, linewidth=1.2, label=str(label))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    save(fig, directory, filename)



STATUS_LABELS = {
    "solved": "решение прошло проверку",
    "converged": "достигнута сходимость",
    "max iterations": "достигнут предел итераций",
    "time limit": "превышено ограничение времени",
    "diverged": "расходимость",
    "breakdown": "численный сбой алгоритма",
    "verification_failed": "проверка невязки не пройдена",
    "exception": "ошибка выполнения",
    "not_available": "эксперимент недоступен",
    "n/a": "метод неприменим",
    "out_of_memory": "недостаточно памяти",
    "not_available": "необходимые возможности библиотеки недоступны",
    "non_finite": "получены неконечные значения",
    "load_error": "ошибка чтения матрицы",
    "matrix_run_error": "ошибка обработки матрицы",
    "loaded": "матрица загружена",
}


def translate_status(value):
    if pd.isna(value):
        return "статус не указан"
    return STATUS_LABELS.get(str(value), str(value))

def build(results: Path, figures: Path):
    figures.mkdir(parents=True, exist_ok=True)
    # Удаляем только графики, создаваемые этим построителем: старый график
    # не должен пережить новый CSV-набор и выдаваться за его результат.
    generated = {
        "dense_time.png", "iterative_methods.png", "accuracy_general.png", "accuracy_symmetric_positive_definite.png",
        "accuracy_Hilbert.png", "accuracy_random_normal.png", "conjugate_gradient_condition.png",
        "poisson_time.png", "tridiagonal_time.png", "sparse_time.png", "multiple_right_hand_sides.png",
        "library_threads.png", "timing_variability.png",
    }
    for filename in generated:
        (figures / filename).unlink(missing_ok=True)
    parts = ["# Сводка результатов вычислительных экспериментов\n",
             "Таблицы построены по файлам данных текущего запуска. Отсутствующие значения "
             "не заменялись предположениями.\n"]
    config_file = results / "run_config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text(encoding="utf-8"))
        profile_name = {"quick": "сокращённый", "full": "основной"}.get(config.get("profile"), str(config.get("profile")))
        initial_seed = config.get("initial_seed", config.get("seed"))
        parts.append(f"Профиль: **{profile_name}**. Начальное значение генератора: **{initial_seed}**.\n")
    environment_file = results / "environment.json"
    if environment_file.exists():
        env = json.loads(environment_file.read_text(encoding="utf-8"))
        parts.append("Среда: Python {python}, NumPy {numpy}, SciPy {scipy}, процессор: {cpu}.\n".format(
            python=env.get("python", "?"), numpy=env.get("numpy", "?"),
            scipy=env.get("scipy", "?"), cpu=env.get("processor") or env.get("machine", "?")))

    dense = load(results, "dense")
    if dense is not None and not dense.empty:
        plot_frame = dense.copy()
        if "level" in plot_frame:
            plot_frame["label"] = plot_frame["method"].astype(str) + " / " + plot_frame["level"].astype(str)
        else:
            plot_frame["label"] = plot_frame["method"].astype(str)
        plot_lines(plot_frame, "n", "time_median", "label", figures, "dense_time.png",
                   "Плотные системы: время решения", "Размер системы", "Время, с")
        dense_ok = dense[dense.status.isin({"solved", "converged"})] if "status" in dense else dense
        add_summary(parts, "Плотные системы: медианное время успешных решений, с",
                    dense_ok.pivot_table(index="n", columns="method", values="time_median").reset_index())

    iter_data = load(results, "iterative_methods")
    if iter_data is not None and not iter_data.empty:
        it = iter_data.copy()
        it["label"] = it["method"].astype(str) + " / " + it["level"].astype(str)
        plot_lines(it, "n", "time_median", "label", figures, "iterative_methods.png",
                   "Итерационные методы: уровни реализации", "Размер системы", "Время, с")
        add_summary(parts, "Итерационные методы: время и точность",
                    iter_data[[c for c in ("method", "level", "n", "time_median", "iters",
                                           "rel_error", "rel_residual", "status") if c in iter_data]])

    accuracy = load(results, "accuracy")
    if accuracy is not None and not accuracy.empty:
        for matrix_type in ("general", "symmetric_positive_definite", "Hilbert", "random_normal"):
            subset = accuracy[accuracy.get("matrix_type", pd.Series(index=accuracy.index, dtype=str)) == matrix_type]
            if subset.empty:
                continue
            xcol = "cond_actual" if "cond_actual" in subset and subset["cond_actual"].notna().any() else "cond_target"
            if xcol not in subset or "rel_error" not in subset:
                continue
            fig, ax = plt.subplots(figsize=(7, 4.5))
            subset = subset.copy()
            subset[xcol] = pd.to_numeric(subset[xcol], errors="coerce")
            subset["rel_error"] = pd.to_numeric(subset["rel_error"], errors="coerce")
            subset = subset[(subset[xcol] > 0) & (subset["rel_error"] > 0) & np.isfinite(subset["rel_error"])]
            for i, (method, group_frame) in enumerate(subset.groupby("method", sort=False)):
                grouped = group_frame.groupby(xcol).rel_error.median().reset_index()
                ax.plot(grouped[xcol], grouped.rel_error, marker=MARKERS[i % len(MARKERS)],
                        linewidth=1.2, label=str(method))
            if not subset.empty:
                ax.set_xscale("log")
                ax.set_yscale("log")
                matrix_type_labels = {
                    "general": "общие матрицы",
                    "symmetric_positive_definite": "симметричные положительно определённые матрицы",
                    "Hilbert": "матрицы Гильберта",
                    "random_normal": "случайные матрицы с нормальным распределением элементов",
                }
                ax.set(title=f"Погрешность: {matrix_type_labels.get(matrix_type, matrix_type)}", xlabel="Число обусловленности (фактическое)",
                       ylabel="Относительная прямая ошибка")
                ax.legend()
                save(fig, figures, f"accuracy_{matrix_type}.png")
        add_summary(parts, "Обусловленность и точность",
                    accuracy[[c for c in ("matrix_type", "method", "n", "cond_target", "cond_actual",
                                          "rel_error", "rel_residual", "status") if c in accuracy]])

    cg = load(results, "conjugate_gradient_condition")
    if cg is not None and not cg.empty:
        if {"cond_actual", "iters"}.issubset(cg.columns):
            plot_lines(cg, "cond_actual", "iters", "method", figures, "conjugate_gradient_condition.png",
                       "Сопряжённые градиенты: число итераций и обусловленность", "Число обусловленности", "Итерации")
        add_summary(parts, "Сопряжённые градиенты: зависимость от обусловленности",
                    cg[[c for c in ("cond_target", "cond_actual", "iters", "time_median",
                                    "rel_error", "rel_residual", "status") if c in cg]])

    poisson = load(results, "poisson")
    if poisson is not None and not poisson.empty:
        if "family" in poisson:
            plot_lines(poisson, "n", "time_median", "method", figures, "poisson_time.png",
                       "Задача Пуассона: время решения", "Размер системы", "Время, с")
        add_summary(parts, "Задача Пуассона",
                    poisson[[c for c in ("N", "n", "method", "family", "time_median", "iters",
                                         "rel_error", "rel_residual", "status") if c in poisson]])

    tri = load(results, "tridiagonal")
    if tri is not None and not tri.empty:
        plot_lines(tri, "n", "time_median", "method", figures, "tridiagonal_time.png",
                   "Трёхдиагональные системы", "Размер системы", "Время, с")
        tri_ok = tri[tri.status.isin({"solved", "converged"})] if "status" in tri else tri
        add_summary(parts, "Трёхдиагональные системы: успешные решения",
                    tri_ok.pivot_table(index="n", columns="method", values="time_median").reset_index())

    sparse = load(results, "random_sparse")
    if sparse is not None and not sparse.empty:
        plot_lines(sparse, "n", "time_median", "method", figures, "sparse_time.png",
                   "Синтетические разреженные системы", "Размер системы", "Время, с")
        add_summary(parts, "Разреженные системы",
                    sparse[[c for c in ("n", "nnz", "method", "time_median", "iters",
                                        "rel_error", "rel_residual", "status") if c in sparse]])

    rhs = load(results, "multiple_right_hand_sides")
    if rhs is not None and not rhs.empty:
        plot_lines(rhs, "right_hand_side_count", "time_median", "method", figures, "multiple_right_hand_sides.png",
                   "Несколько правых частей", "Число правых частей", "Время, с")
        rhs_ok = rhs[rhs.status.isin({"solved", "converged"})] if "status" in rhs else rhs
        add_summary(parts, "Несколько правых частей: успешные решения",
                    rhs_ok[[c for c in ("right_hand_side_count", "method", "time_median", "rel_error", "rel_residual", "backward_error")
                            if c in rhs_ok]])

    threads = load(results, "library_threads")
    if threads is not None and not threads.empty:
        if {"n", "library_threads", "time_median"}.issubset(threads.columns):
            thread_data = threads[threads.status.isin({"solved", "converged"})] if "status" in threads else threads
            pivot = thread_data.pivot_table(index="n", columns="library_threads", values="time_median")
            if len(pivot.columns) >= 2:
                fig, ax = plt.subplots(figsize=(6.5, 4))
                speedup = pivot.iloc[:, 0] / pivot.iloc[:, -1]
                ax.plot(speedup.index, speedup.values, "o-")
                ax.axhline(1.0, color="gray", linewidth=0.8)
                ax.set_xscale("log")
                ax.set(title="Влияние числа вычислительных потоков", xlabel="Размер системы",
                       ylabel=f"Отношение времени: {pivot.columns[0]} / {pivot.columns[-1]} потоков")
                save(fig, figures, "library_threads.png")
            add_summary(parts, "Влияние числа вычислительных потоков", pivot.reset_index())

    spread = load(results, "timing_variability")
    if spread is not None and not spread.empty and {"method", "time"}.issubset(spread.columns):
        if "status" in spread.columns:
            spread = spread[spread.status == "solved"].copy()
        if spread.empty:
            spread = None
    if spread is not None and not spread.empty and {"method", "time"}.issubset(spread.columns):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        names, data = [], []
        for name, group_frame in spread.groupby("method", sort=False):
            values = pd.to_numeric(group_frame.time, errors="coerce").dropna().to_numpy()
            if values.size:
                names.append(str(name))
                data.append(values / np.median(values))
        if data:
            # Matplotlib 3.9 переименовал labels в tick_labels; сохраняем
            # совместимость с поддерживаемыми версиями 3.7–3.8.
            try:
                ax.boxplot(data, tick_labels=names, showfliers=True)
            except TypeError:  # Matplotlib < 3.9
                ax.boxplot(data, labels=names, showfliers=True)
            ax.set(title="Разброс отдельных измерений", ylabel="Время / медиана")
            ax.tick_params(axis="x", rotation=25)
            save(fig, figures, "timing_variability.png")
            summary = spread.groupby("method").time.agg(["count", "median", "min", "max", "std"])
            add_summary(parts, "Разброс измерений", summary.reset_index())

    status_files = [load(results, name) for name in
                    ("dense", "iterative_methods", "accuracy", "conjugate_gradient_condition", "poisson", "tridiagonal",
                     "random_sparse", "multiple_right_hand_sides", "timing_variability", "library_threads")]
    status_frames = [frame for frame in status_files if frame is not None and "status" in frame]
    if status_frames:
        statuses = pd.concat([frame[["status"]] for frame in status_frames], ignore_index=True)
        counts = statuses.status.map(translate_status).value_counts(dropna=False).rename_axis("Статус").reset_index(name="Число запусков")
        add_summary(parts, "Сводка статусов", counts)

    summary_path = results / "summary.md"
    summary_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Графики: {figures}; сводка: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "results"), help="каталог CSV")
    parser.add_argument("--figures", default=None, help="каталог графиков")
    args = parser.parse_args()
    result_dir = Path(args.results).resolve()
    figure_dir = Path(args.figures).resolve() if args.figures else result_dir.parent / "figures"
    if not result_dir.exists():
        parser.error(f"каталог результатов не существует: {result_dir}")
    build(result_dir, figure_dir)


if __name__ == "__main__":
    main()
