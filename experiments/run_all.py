"""Запуск синтетических экспериментов.

Примеры:
    python -m experiments.run_all --quick --seed 20261113
    python -m experiments.run_all --seed 20261113
    python -m experiments.run_all poisson tridiagonal --seed 20261113

Синтетические матрицы задают контролируемые условия, а не заменяют проверку
на независимом наборе реальных матриц. Начальное значение генератора случайных чисел сохраняется в результатах.
"""

from __future__ import annotations

import argparse
import json
import secrets
import time
from functools import partial
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from experiments import common
from experiments.common import LIB, NP, PROFILES, PY, SPARSE, TOL, Recorder, stats
from slae import bench as B
from slae import direct as D
from slae import iterative as I
from slae import matrices as M

P: dict = {}


def _seed(*parts: int) -> int:
    sequence = np.random.SeedSequence([int(P["seed"]), *map(int, parts)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _unpack(result):
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        return result[0], result[1]
    return result, {}


def run(rec: Recorder, fn, args, x_true, A, b, **meta):
    """Измерить метод и сохранить время, фактическую невязку и прямую ошибку."""
    base = dict(meta)
    start = time.perf_counter()
    try:
        measurement = B.measure(
            fn, lambda: args, budget=P["budget"], min_repeats=2,
            max_repeats=3 if P["profile"] == "quick" else 7,
            target_batch_seconds=0.001 if P["profile"] == "quick" else 0.002,
            max_batch_size=4 if P["profile"] == "quick" else 32,
        )
        solution, info = _unpack(measurement["result"])
        solution = np.asarray(solution, dtype=np.float64)
        if solution.shape != np.asarray(x_true).shape or not np.isfinite(solution).all():
            raise ArithmeticError("решатель вернул вектор неправильной формы или не конечные значения")
        rel_err = B.rel_error(solution, x_true)
        rel_res = B.rel_residual(A, solution, b)
        back_err = B.backward_error(A, solution, b)
        solver_status = info.get("status", "solved")
        status = solver_status
        if solver_status in {"solved", "converged"} and (not np.isfinite(rel_res) or rel_res > TOL):
            status = "verification_failed"
        row = {
            **base,
            **stats(measurement),
            "rel_error": rel_err,
            "rel_residual": rel_res,
            "backward_error": back_err,
            "iters": info.get("iters"),
            "converged": info.get("converged"),
            "solver_status": solver_status,
            "status": status,
            "error_message": None,
        }
    except Exception as exc:  # сбой отдельного метода не должен уничтожать всю серию
        row = {
            **base,
            "time_median": time.perf_counter() - start,
            "rel_error": np.nan,
            "rel_residual": np.nan,
            "backward_error": np.nan,
            "iters": None,
            "converged": False,
            "status": "exception",
            "error_message": f"{type(exc).__name__}: {exc}"[:240],
        }
    rec.add(**row)
    return row


# 1. Плотные прямые методы: сравнение алгоритма и уровня реализации.
def s_dense():
    rec = Recorder("dense")
    for n in P["dense_sizes"]:
        seed_a, seed_s, seed_x = _seed(1, n), _seed(2, n), _seed(3, n)
        A = M.dense_diag_dominant(n, seed=seed_a)
        x, b = M.right_hand_side_for(A, seed=seed_x)
        S = M.dense_symmetric_positive_definite(n, seed=seed_s)
        xs, bs = M.right_hand_side_for(S, seed=seed_x)
        use_pure = n <= P["pure_max"]
        Al, bl = (A.tolist(), b.tolist()) if use_pure else (None, None)
        Sl, bsl = (S.tolist(), bs.tolist()) if use_pure else (None, None)
        cases = [
            ("Гаусс с выбором главного элемента", PY, D.gauss_pure, (Al, bl), A, b, x,
             use_pure, seed_a, seed_x),
            ("Разложение Холецкого", PY, D.cholesky_pure, (Sl, bsl), S, bs, xs,
             use_pure, seed_s, seed_x),
            ("Гаусс с выбором главного элемента", NP, D.gauss_numpy, (A, b), A, b, x,
             n <= P["numpy_gauss_max"], seed_a, seed_x),
            ("Разложение Холецкого", NP, D.cholesky_numpy, (S, bs), S, bs, xs,
             n <= P["numpy_gauss_max"], seed_s, seed_x),
            ("Решатель NumPy", LIB, D.solve_numpy, (A, b), A, b, x, True, seed_a, seed_x),
            ("LU-разложение (SciPy)", LIB, D.solve_lu_scipy, (A, b), A, b, x, True, seed_a, seed_x),
            ("Разложение Холецкого (SciPy)", LIB, D.solve_cholesky_scipy, (S, bs), S, bs, xs,
             True, seed_s, seed_x),
            ("QR-разложение (SciPy)", LIB, D.solve_qr_scipy, (A, b), A, b, x, True, seed_a, seed_x),
            ("Метод наименьших квадратов (сингулярное разложение)", LIB, D.solve_svd_lstsq, (A, b), A, b, x,
             n <= P["svd_max"], seed_a, seed_x),
        ]
        for method, level, fn, args, matrix, rhs, exact, applicable, matrix_seed, right_hand_side_seed in cases:
            if applicable:
                run(rec, fn, args, exact, matrix, rhs, matrix_type="dense", method=method,
                    level=level, n=n, matrix_seed=matrix_seed, right_hand_side_seed=right_hand_side_seed)
    rec.save()


# 2. Реализации стационарных методов и методов подпространств Крылова на одинаковых системах.
def s_iterative_methods():
    rec = Recorder("iterative_methods")
    for n in P["iter_sizes"]:
        seed_a, seed_xa = _seed(4, n), _seed(5, n)
        seed_s, seed_xs = _seed(6, n), _seed(7, n)
        A = M.dense_diag_dominant(n, seed=seed_a)
        x, b = M.right_hand_side_for(A, seed=seed_xa)
        S = M.dense_symmetric_positive_definite(n, seed=seed_s)
        xs, bs = M.right_hand_side_for(S, seed=seed_xs)
        cases = [
            ("Якоби", PY, partial(I.jacobi_pure, tol=TOL), (A.tolist(), b.tolist()),
             A, b, x, seed_a, seed_xa),
            ("Якоби", NP, partial(I.jacobi, tol=TOL), (A, b), A, b, x, seed_a, seed_xa),
            ("Гаусс — Зейдель", PY, partial(I.gauss_seidel_pure, tol=TOL),
             (A.tolist(), b.tolist()), A, b, x, seed_a, seed_xa),
            ("Гаусс — Зейдель", NP, partial(I.gauss_seidel, tol=TOL),
             (A, b), A, b, x, seed_a, seed_xa),
            ("Метод сопряжённых градиентов", PY, partial(I.cg_pure, tol=TOL),
             (S.tolist(), bs.tolist()), S, bs, xs, seed_s, seed_xs),
            ("Метод сопряжённых градиентов", NP, partial(I.cg, tol=TOL),
             (S, bs), S, bs, xs, seed_s, seed_xs),
        ]
        for method, level, fn, args, matrix, rhs, exact, matrix_seed, right_hand_side_seed in cases:
            run(rec, fn, args, exact, matrix, rhs, matrix_type="dense", method=method,
                level=level, n=n, matrix_seed=matrix_seed, right_hand_side_seed=right_hand_side_seed)
    rec.save()


# 3. Устойчивость к обусловленности и влияние выбора главного элемента.
def s_accuracy():
    rec = Recorder("accuracy")
    methods_general = [
        ("Гаусс без выбора главного элемента", partial(D.gauss_numpy, pivot=False)),
        ("Гаусс с выбором главного элемента", D.gauss_numpy),
        ("Решатель NumPy", D.solve_numpy),
        ("QR-разложение", D.solve_qr_scipy),
        ("Метод наименьших квадратов (сингулярное разложение)", D.solve_svd_lstsq),
    ]
    for exponent in P["cond_exponents"]:
        for repeat in range(P["cond_repeats"]):
            seed = _seed(8, exponent, repeat)
            for is_positive_definite in (False, True):
                A = M.with_condition(P["cond_n"], 10.0 ** exponent,
                                     seed=seed, positive_definite=is_positive_definite)
                right_hand_side_seed = _seed(9, exponent, repeat, int(is_positive_definite))
                x, b = M.right_hand_side_for(A, seed=right_hand_side_seed)
                actual_cond = float(np.linalg.cond(A))
                selected = [("Разложение Холецкого (SciPy)", D.solve_cholesky_scipy),
                            ("Решатель NumPy", D.solve_numpy)] if is_positive_definite else methods_general
                for name, fn in selected:
                    started = time.perf_counter()
                    try:
                        solution = fn(A, b)
                        rel_error = B.rel_error(solution, x)
                        rel_residual = B.rel_residual(A, solution, b)
                        row = {"matrix_type": "symmetric_positive_definite" if is_positive_definite else "general",
                               "method": name, "n": P["cond_n"], "matrix_seed": seed,
                               "right_hand_side_seed": right_hand_side_seed,
                               "cond_target": 10.0 ** exponent, "cond_actual": actual_cond,
                               "rel_error": rel_error, "rel_residual": rel_residual,
                               "backward_error": B.backward_error(A, solution, b),
                               "solver_status": "solved",
                               "status": "solved" if np.isfinite(rel_residual) and rel_residual <= TOL else "verification_failed",
                               "time": time.perf_counter() - started}
                    except Exception as exc:
                        row = {"matrix_type": "symmetric_positive_definite" if is_positive_definite else "general",
                               "method": name, "n": P["cond_n"], "matrix_seed": seed,
                               "right_hand_side_seed": right_hand_side_seed,
                               "cond_target": 10.0 ** exponent, "cond_actual": actual_cond,
                               "rel_error": np.nan, "rel_residual": np.nan,
                               "status": "exception", "error_message": f"{type(exc).__name__}: {exc}"[:240],
                               "time": time.perf_counter() - started}
                    rec.add(**row)
    for n in range(2, 13):
        A = M.hilbert(n)
        right_hand_side_seed = _seed(10, n)
        x, b = M.right_hand_side_for(A, seed=right_hand_side_seed)
        for name, fn in methods_general + [("Разложение Холецкого (SciPy)", D.solve_cholesky_scipy)]:
            started = time.perf_counter()
            try:
                solution = fn(A, b)
                rel_error = B.rel_error(solution, x)
                rel_residual = B.rel_residual(A, solution, b)
                rec.add(matrix_type="Hilbert", method=name, n=n, matrix_seed="deterministic:hilbert",
                        right_hand_side_seed=right_hand_side_seed, cond_actual=float(np.linalg.cond(A)), rel_error=rel_error,
                        rel_residual=rel_residual, backward_error=B.backward_error(A, solution, b),
                        solver_status="solved",
                        status="solved" if np.isfinite(rel_residual) and rel_residual <= TOL else "verification_failed",
                        time=time.perf_counter() - started)
            except Exception as exc:
                rec.add(matrix_type="Hilbert", method=name, n=n, matrix_seed="deterministic:hilbert",
                        right_hand_side_seed=right_hand_side_seed, cond_actual=float(np.linalg.cond(A)), rel_error=np.nan,
                        rel_residual=np.nan, status="exception",
                        error_message=f"{type(exc).__name__}: {exc}"[:240],
                        time=time.perf_counter() - started)
    for n in P["pivot_sizes"]:
        for repeat in range(P["pivot_repeats"]):
            seed = _seed(11, n, repeat)
            A = M.dense_general(n, seed=seed)
            right_hand_side_seed = _seed(12, n, repeat)
            x, b = M.right_hand_side_for(A, seed=right_hand_side_seed)
            for name, fn in methods_general[:3]:
                started = time.perf_counter()
                try:
                    solution = fn(A, b)
                    rel_error = B.rel_error(solution, x)
                    rel_residual = B.rel_residual(A, solution, b)
                    rec.add(matrix_type="random_normal", method=name, n=n, matrix_seed=seed, right_hand_side_seed=right_hand_side_seed,
                            cond_actual=float(np.linalg.cond(A)), rel_error=rel_error,
                            rel_residual=rel_residual, backward_error=B.backward_error(A, solution, b),
                            solver_status="solved",
                            status="solved" if np.isfinite(rel_residual) and rel_residual <= TOL else "verification_failed",
                            time=time.perf_counter() - started)
                except Exception as exc:
                    rec.add(matrix_type="random_normal", method=name, n=n, matrix_seed=seed, right_hand_side_seed=right_hand_side_seed,
                            cond_actual=float(np.linalg.cond(A)), rel_error=np.nan,
                            rel_residual=np.nan, status="exception",
                            error_message=f"{type(exc).__name__}: {exc}"[:240],
                            time=time.perf_counter() - started)
    rec.save()


# 4. Влияние обусловленности на метод сопряжённых градиентов.
def s_conjugate_gradient_condition():
    rec = Recorder("conjugate_gradient_condition")
    n = max(P["cond_n"], 100)
    time_limit = 1.0 if P["profile"] == "quick" else 30.0
    for exponent in P["cond_exponents"]:
        seed = _seed(13, exponent)
        A = M.with_condition(n, 10.0 ** exponent, seed=seed, positive_definite=True)
        right_hand_side_seed = _seed(14, exponent)
        x, b = M.right_hand_side_for(A, seed=right_hand_side_seed)
        run(rec, partial(I.cg, tol=TOL, maxiter=20 * n, time_limit=time_limit), (A, b), x, A, b,
            matrix_type="symmetric_positive_definite", method="Метод сопряжённых градиентов", n=n,
            cond_actual=float(np.linalg.cond(A)), cond_target=10.0 ** exponent,
            matrix_seed=seed, right_hand_side_seed=right_hand_side_seed)
    rec.save()


# 5. Модельная задача Пуассона с симметричной положительно определённой матрицей.
def s_poisson():
    rec = Recorder("poisson")
    time_limit = 1.0 if P["profile"] == "quick" else 30.0
    for N in P["poisson_N"]:
        A, x, b = M.poisson2d_problem(N)
        n = N * N
        cases = [
            ("Якоби", "stationary", partial(I.jacobi, tol=TOL, maxiter=1_000_000, time_limit=time_limit),
             N <= P["poisson_stationary_max_N"]),
            ("Гаусс — Зейдель", "stationary", partial(I.gauss_seidel, tol=TOL, maxiter=1_000_000, time_limit=time_limit),
             N <= P["poisson_stationary_max_N"]),
            ("Последовательная верхняя релаксация (теоретический параметр)", "stationary",
             partial(I.successive_over_relaxation, relaxation=I.successive_over_relaxation_optimal_parameter_poisson(N), tol=TOL, maxiter=1_000_000, time_limit=time_limit), True),
            ("Метод сопряжённых градиентов", "krylov", partial(I.cg, tol=TOL, maxiter=1_000_000, time_limit=time_limit), True),
            ("Метод сопряжённых градиентов с симметричным предобуславливателем на основе верхней релаксации", "krylov",
             lambda mat, rhs: I.cg(mat, rhs, tol=TOL, maxiter=1_000_000,
                                    M_solve=I.symmetric_successive_over_relaxation_preconditioner(mat, relaxation=1.0), time_limit=time_limit), True),
            ("Метод сопряжённых градиентов (SciPy)", "krylov", partial(I.scipy_cg, tol=TOL, maxiter=1_000_000, time_limit=time_limit), True),
            ("Стабилизированный метод бисопряжённых градиентов (SciPy)", "krylov", partial(I.scipy_bicgstab, tol=TOL, maxiter=1_000_000, time_limit=time_limit), True),
            ("Обобщённый метод минимальной невязки (SciPy)", "krylov", partial(I.scipy_gmres, tol=TOL, maxiter=20_000, time_limit=time_limit), True),
            ("Разреженное LU-разложение (SciPy)", "direct", D.solve_sparse_lu, True),
        ]
        for method, family, fn, applicable in cases:
            if applicable:
                run(rec, fn, (A, b), x, A, b, matrix_type="Poisson", method=method,
                    family=family, n=n, N=N, matrix_seed="deterministic:poisson2d",
                    right_hand_side_seed="deterministic:poisson2d_problem")
    rec.save()


# 6. Трёхдиагональные системы: специализированный метод против общих решателей.
def s_tridiagonal():
    rec = Recorder("tridiagonal")
    for n in P["tri_sizes"]:
        seed = _seed(15, n)
        lower, diag, upper = M.tridiagonal_components(n, seed=seed)
        right_hand_side_seed = _seed(16, n)
        x = np.random.default_rng(right_hand_side_seed).standard_normal(n)
        b = M.tridiagonal_matvec(lower, diag, upper, x)
        sparse = M.tridiagonal_to_sparse(lower, diag, upper)
        cases = [
            ("Алгоритм Томаса (Python)", PY, D.thomas_pure,
             (lower.tolist(), diag.tolist(), upper.tolist(), b.tolist()), True),
            ("Алгоритм Томаса (NumPy)", NP, D.thomas_numpy_loop,
             (lower, diag, upper, b), True),
            ("Трёхдиагональный решатель (SciPy)", LIB, D.solve_tridiagonal_scipy,
             (M.tridiagonal_to_scipy_storage(lower, diag, upper), b), True),
            ("Разреженное LU-разложение (SciPy)", SPARSE, D.solve_sparse_lu, (sparse, b), True),
        ]
        if n <= P["tri_dense_max"]:
            cases.append(("Плотный решатель NumPy", LIB, D.solve_numpy,
                          (M.tridiagonal_to_dense(lower, diag, upper), b), True))
        for method, level, fn, args, applicable in cases:
            if applicable:
                matrix = M.tridiagonal_to_dense(lower, diag, upper) if n <= P["tri_dense_max"] else sparse
                run(rec, fn, args, x, matrix, b, matrix_type="tridiagonal",
                    method=method, level=level, n=n, matrix_seed=seed, right_hand_side_seed=right_hand_side_seed)
    rec.save()


# 7. Случайные разреженные диагонально преобладающие системы.
def s_sparse():
    rec = Recorder("random_sparse")
    time_limit = 1.0 if P["profile"] == "quick" else 30.0
    for n in P["sparse_sizes"]:
        seed = _seed(17, n)
        A = M.sparse_diag_dominant(n, nnz_per_row=10, seed=seed)
        right_hand_side_seed = _seed(18, n)
        x, b = M.right_hand_side_for(A, seed=right_hand_side_seed)
        methods = [
            ("Якоби", partial(I.jacobi, tol=TOL, maxiter=1_000_000, time_limit=time_limit)),
            ("Гаусс — Зейдель", partial(I.gauss_seidel, tol=TOL, maxiter=1_000_000, time_limit=time_limit)),
            ("Стабилизированный метод бисопряжённых градиентов (SciPy)", partial(I.scipy_bicgstab, tol=TOL, maxiter=100_000, time_limit=time_limit)),
            ("Обобщённый метод минимальной невязки (SciPy)", partial(I.scipy_gmres, tol=TOL, maxiter=5000, time_limit=time_limit)),
        ]
        if n <= P["sparse_lu_max"]:
            methods.append(("Разреженное LU-разложение (SciPy)", D.solve_sparse_lu))
        for name, fn in methods:
            run(rec, fn, (A, b), x, A, b, matrix_type="random_sparse", method=name,
                n=n, nnz=A.nnz, matrix_seed=seed, right_hand_side_seed=right_hand_side_seed)
    rec.save()


# 8. Несколько правых частей: сравнение повторного решения и переиспользования LU.
def s_multiple_right_hand_sides():
    rec = Recorder("multiple_right_hand_sides")
    n = P["right_hand_side_n"]
    seed = _seed(19, n)
    A = M.dense_diag_dominant(n, seed=seed)
    for count in P["right_hand_side_counts"]:
        right_hand_side_seed = _seed(20, n, count)
        rng = np.random.default_rng(right_hand_side_seed)
        X = rng.standard_normal((n, count))
        Bm = A @ X

        def solve_each():
            return np.column_stack([np.linalg.solve(A, Bm[:, j]) for j in range(count)])

        def lu_once():
            factor = sla.lu_factor(A)
            return sla.lu_solve(factor, Bm)

        methods = [("Отдельный вызов решателя для каждой правой части", solve_each),
                   ("Однократное LU-разложение с повторным использованием множителей", lu_once),
                   ("Один вызов решателя для всех правых частей", lambda: np.linalg.solve(A, Bm))]
        for method, fn in methods:
            try:
                measurement = B.measure(
                    fn, lambda: (), budget=P["budget"], min_repeats=2,
                    max_repeats=3 if P["profile"] == "quick" else 7,
                    target_batch_seconds=0.001 if P["profile"] == "quick" else 0.002,
                    max_batch_size=4 if P["profile"] == "quick" else 32,
                )
                estimate = np.asarray(measurement["result"], dtype=np.float64)
                rel_error = B.rel_error(estimate, X)
                rel_residual = B.rel_residual(A, estimate, Bm)
                rec.add(matrix_type="dense", method=method, n=n, right_hand_side_count=count,
                        matrix_seed=seed, right_hand_side_seed=right_hand_side_seed,
                        **stats(measurement), rel_error=rel_error, rel_residual=rel_residual,
                        backward_error=B.backward_error(A, estimate, Bm), solver_status="solved",
                        status="solved" if np.isfinite(rel_residual) and rel_residual <= TOL else "verification_failed")
            except Exception as exc:
                rec.add(matrix_type="dense", method=method, n=n, right_hand_side_count=count,
                        matrix_seed=seed, right_hand_side_seed=right_hand_side_seed,
                        status="exception", error_message=f"{type(exc).__name__}: {exc}"[:240])
    rec.save()


# 9. Число вычислительных потоков сравнивается только при наличии разных доступных настроек.
def s_library_threads():
    rec = Recorder("library_threads")
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError:
        rec.add(status="not_available", reason="установите threadpoolctl")
        rec.save()
        return
    # Запоминаем исходную настройку библиотеки до применения общего ограничения.
    # Это настройка программной библиотеки, а не оценка максимума процессора.
    configured_count = int(P.get("default_library_threads", 1))
    thread_counts = sorted(set([1, configured_count]))
    if len(thread_counts) < 2:
        rec.add(status="not_available", reason="библиотека изначально настроена на один поток; сравнение невозможно")
        rec.save()
        return
    for n in P["dense_sizes"]:
        matrix_seed, right_hand_side_seed = _seed(21, n), _seed(22, n)
        A = M.dense_diag_dominant(n, seed=matrix_seed)
        x, b = M.right_hand_side_for(A, seed=right_hand_side_seed)
        for count in thread_counts:
            with threadpool_limits(limits=count):
                run(rec, D.solve_numpy, (A, b), x, A, b, matrix_type="dense",
                    method="Решатель NumPy", n=n, library_threads=count,
                    matrix_seed=matrix_seed, right_hand_side_seed=right_hand_side_seed)
    rec.save()


# 10. Разброс индивидуальных измерений с чередованием порядка методов.
def s_timing_variability():
    rec = Recorder("timing_variability")
    n = min(max(P["iter_sizes"]), 200)
    A = M.dense_diag_dominant(n, seed=_seed(23, n))
    right_hand_side_seed = _seed(24, n)
    x, b = M.right_hand_side_for(A, seed=right_hand_side_seed)
    systems = [
        ("Решатель NumPy", D.solve_numpy, (A, b)),
        ("LU-разложение (SciPy)", D.solve_lu_scipy, (A, b)),
        ("QR-разложение (SciPy)", D.solve_qr_scipy, (A, b)),
    ]
    order_seed = _seed(25, n)
    rng = np.random.default_rng(order_seed)
    repeats = 25 if P["profile"] == "full" else 5
    schedule = [(i, j) for i in range(repeats) for j in range(len(systems))]
    rng.shuffle(schedule)
    for run_number, method_index in schedule:
        name, fn, args = systems[method_index]
        start = time.perf_counter()
        try:
            solution = fn(*args)
            elapsed = time.perf_counter() - start
            rel_residual = B.rel_residual(A, solution, b)
            rec.rows.append({"method": name, "n": n, "run": run_number,
                             "time": elapsed, "matrix_seed": _seed(23, n), "right_hand_side_seed": right_hand_side_seed,
                             "order_seed": order_seed,
                             "rel_error": B.rel_error(solution, x),
                             "rel_residual": rel_residual,
                             "backward_error": B.backward_error(A, solution, b),
                             "solver_status": "solved",
                             "status": "solved" if np.isfinite(rel_residual) and rel_residual <= TOL else "verification_failed"})
        except Exception as exc:
            rec.rows.append({"method": name, "n": n, "run": run_number,
                             "time": time.perf_counter() - start,
                             "matrix_seed": _seed(23, n), "right_hand_side_seed": right_hand_side_seed,
                             "order_seed": order_seed,
                             "status": "exception",
                             "error_message": f"{type(exc).__name__}: {exc}"[:240]})
    rec.save()


SERIES = {
    "dense": s_dense,
    "iterative_methods": s_iterative_methods,
    "accuracy": s_accuracy,
    "conjugate_gradient_condition": s_conjugate_gradient_condition,
    "poisson": s_poisson,
    "tridiagonal": s_tridiagonal,
    "sparse": s_sparse,
    "multiple_right_hand_sides": s_multiple_right_hand_sides,
    "library_threads": s_library_threads,
    "timing_variability": s_timing_variability,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("series", nargs="*", help=f"серии: {', '.join(SERIES)}; по умолчанию все")
    parser.add_argument("--quick", action="store_true", help="сокращённый набор размеров и повторов для проверки работы программы")
    parser.add_argument("--seed", type=int, default=None, help="начальное значение генератора; по умолчанию выбирается случайно")
    parser.add_argument("--out", default=None, help="каталог CSV-результатов")
    parser.add_argument("--library-threads", type=int, default=1,
                        help="число потоков вычислительных библиотек в основных сериях (по умолчанию 1)")
    args = parser.parse_args()
    if args.library_threads < 1:
        parser.error("--library-threads должен быть положительным")
    unknown = set(args.series) - set(SERIES)
    if unknown:
        parser.error(f"неизвестные серии: {', '.join(sorted(unknown))}")

    P.update(PROFILES["quick" if args.quick else "full"])
    P["profile"] = "quick" if args.quick else "full"
    P["seed"] = int(args.seed) if args.seed is not None else secrets.randbits(32)
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
        P["default_library_threads"] = max(
            (int(item.get("num_threads", 1)) for item in pools if item.get("user_api") == "blas"),
            default=1,
        )
    except ImportError:
        P["default_library_threads"] = 1
    output = Path(args.out).resolve() if args.out else common.ROOT / ("results_quick" if args.quick else "results")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"каталог результатов не пустой: {output}. Очистите его командой clean_results "
                     "или укажите новый каталог через --out.")
    common.RESULTS = output
    config = {"profile": P["profile"], "initial_seed": P["seed"],
              "seed_derivation": "начальные значения детерминированно получаются из общего значения, номера серии и параметров",
              "tolerance": TOL, "library_threads_detected": P["default_library_threads"],
              "series": args.series or list(SERIES),
              "sizes": {k: v for k, v in P.items()
                        if isinstance(v, (list, int, float)) and k not in {"seed", "default_library_threads"}}}
    (output / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    B.save_environment(output / "environment.json", experiment_config=config,
                       library_threads_for_main_series=args.library_threads)
    print(f"Начальное значение генератора для этого запуска: {P['seed']}")
    print(f"Каталог результатов: {output}")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        print("Предупреждение: библиотека threadpoolctl не установлена; ограничение числа потоков не применяется.")
        for name in args.series or SERIES:
            SERIES[name]()
    else:
        for name in args.series or SERIES:
            # Серия потоков выполняется вне общего лимита, иначе вложенное
            # переопределение лимита и его восстановление может зависеть от
            # вычислительной библиотеки. Для всех остальных серий ограничение применяется
            # отдельно и гарантированно снимается до следующей серии.
            if name == "library_threads":
                SERIES[name]()
            else:
                with threadpool_limits(limits=args.library_threads):
                    SERIES[name]()


if __name__ == "__main__":
    main()
