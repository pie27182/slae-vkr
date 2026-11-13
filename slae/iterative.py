"""Итерационные методы решения системы ``A x = b``.

Реализованы методы Якоби, Гаусса — Зейделя, последовательной верхней
релаксации, сопряжённых градиентов, бисопряжённых градиентов и минимальной
невязки. Для некоторых методов доступны предобуславливатели.

Решатели возвращают пару ``(x, info)``. Сходимость подтверждается повторным
вычислением фактической невязки ``b - A @ x``. Внутреннюю рекуррентную невязку
метода сопряжённых градиентов нельзя считать достаточным доказательством
сходимости из-за накопления ошибки округления.

В словаре ``info`` указываются число итераций, признак сходимости, фактическая
относительная невязка и текстовый статус. Для метода минимальной невязки
с перезапуском число итераций означает количество завершённых циклов, а не
количество отдельных шагов внутри этих циклов.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla


DIVERGED = 1.0e8


def _check_control(tol: float, maxiter: int, time_limit=None) -> None:
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("допуск сходимости должен быть положительным конечным числом")
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 0:
        raise ValueError("предельное число итераций должно быть целым неотрицательным числом")
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("ограничение времени должно быть положительным конечным числом")


def _check_system(A, b):
    rhs = np.asarray(b, dtype=np.float64)
    if rhs.ndim != 1:
        raise ValueError("b должна быть одномерным вектором")
    if A.shape != (rhs.size, rhs.size) or rhs.size == 0:
        raise ValueError("матрица A должна быть квадратной и согласованной по размеру с непустым вектором b")
    if not np.isfinite(rhs).all():
        raise ValueError("b должна содержать только конечные значения")
    if sp.issparse(A):
        if not np.isfinite(A.data).all():
            raise ValueError("A должна содержать только конечные значения")
        matrix = A.astype(np.float64, copy=True)
    else:
        matrix = np.asarray(A, dtype=np.float64).copy()
        if not np.isfinite(matrix).all():
            raise ValueError("A должна содержать только конечные значения")
    return matrix, rhs.copy()


def _check_symmetric(A) -> None:
    """Проверить симметричность без дорогого спектрального разложения.

    Это необходимое, но не достаточное условие для метода сопряжённых градиентов: положительная
    определённость остаётся ответственностью вызывающего кода/метаданных.
    """
    difference = A - A.T
    if sp.issparse(A):
        asymmetry = float(abs(difference).max()) if difference.nnz else 0.0
        scale = float(abs(A).max()) if A.nnz else 0.0
        n = A.shape[0]
    else:
        asymmetry = float(np.max(np.abs(difference)))
        scale = float(np.max(np.abs(A)))
        n = A.shape[0]
    tolerance = 100.0 * np.finfo(float).eps * max(n, 1) * max(scale, np.finfo(float).tiny)
    if asymmetry > tolerance:
        raise ValueError("метод сопряжённых градиентов требует симметричную матрицу")


def _norm(v) -> float:
    return float(np.linalg.norm(v))


def _relative_residual(r, bnorm: float) -> float:
    # При b == 0 стартовое приближение x=0 уже является точным решением.
    return _norm(r) / (bnorm if bnorm > 0.0 else 1.0)


def _info(iters, converged, resid, status=None, **extra):
    if status is None:
        status = "converged" if converged else "max iterations"
    result = {
        "iters": int(iters),
        "converged": bool(converged),
        "resid": float(resid),
        "status": str(status),
    }
    result.update(extra)
    return result


def _early_status(rel: float, start: float, time_limit) -> str | None:
    if not np.isfinite(rel) or rel > DIVERGED:
        return "diverged"
    if time_limit is not None and time.perf_counter() - start >= time_limit:
        return "time limit"
    return None


def _initial_state(A, b):
    x = np.zeros_like(b, dtype=np.float64)
    bnorm = _norm(b)
    r = b.copy()
    return x, r, bnorm, _relative_residual(r, bnorm)


def _final_info(A, b, x, iters, tol, status=None, **extra):
    bnorm = _norm(b)
    residual = b - A @ x
    rel = _relative_residual(residual, bnorm)
    ok = bool(np.isfinite(rel) and rel <= tol)
    if ok:
        status = "converged"
    elif status is None:
        status = "max iterations"
    return x, _info(iters, ok, rel, status, **extra)


def sparse_triangular_solver(M):
    """Подготовить быстрый решатель треугольной разреженной системы."""
    if not sp.issparse(M) or M.shape[0] != M.shape[1]:
        raise ValueError("матрица предобуславливателя должна быть квадратной и разреженной")
    if not np.isfinite(M.data).all():
        raise ValueError("матрица предобуславливателя должна содержать только конечные значения")
    # spsolve_triangular не требует факторизации общего вида и сохраняет
    # структуру треугольной матрицы. Функция возвращает callable для повторных вызовов.
    # spsolve_triangular надёжнее работает с csr_matrix; некоторые версии
    # SciPy для csr_array с 64-битными индексами завершаются TypeError.
    mat = sp.csr_matrix(M, dtype=np.float64, copy=True)
    lower = bool(sp.triu(mat, k=1).nnz == 0)

    def solve(rhs):
        return spla.spsolve_triangular(mat, np.asarray(rhs, dtype=np.float64),
                                       lower=lower, unit_diagonal=False)
    return solve


def _pure_arrays(A: Sequence[Sequence[float]], b: Sequence[float]):
    a = np.asarray(A, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or rhs.ndim != 1 or rhs.size != a.shape[0]:
        raise ValueError("ожидаются квадратная матрица A и вектор b согласованных размеров")
    if a.size == 0 or not np.isfinite(a).all() or not np.isfinite(rhs).all():
        raise ValueError("A и b должны быть непустыми и конечными")
    return a.tolist(), rhs.tolist()


def _norm_pure(v) -> float:
    return math.sqrt(sum(float(t) * float(t) for t in v))


def _residual_pure(A, b, x):
    return [bi - sum(aij * xj for aij, xj in zip(row, x)) for row, bi in zip(A, b)]


def jacobi_pure(A, b, tol: float = 1e-8, maxiter: int = 10_000):
    _check_control(tol, maxiter)
    a, rhs = _pure_arrays(A, b)
    n = len(rhs)
    diag = [a[i][i] for i in range(n)]
    if any(v == 0.0 for v in diag):
        raise ZeroDivisionError("метод Якоби требует ненулевые диагональные элементы")
    x = [0.0] * n
    bnorm = _norm_pure(rhs)
    denom = bnorm if bnorm > 0 else 1.0
    for k in range(maxiter + 1):
        r = _residual_pure(a, rhs, x)
        rel = _norm_pure(r) / denom
        if not np.isfinite(rel) or rel > DIVERGED:
            return x, _info(k, False, rel, "diverged")
        if rel <= tol:
            return x, _info(k, True, rel)
        if k == maxiter:
            break
        x = [x[i] + r[i] / diag[i] for i in range(n)]
    return x, _info(maxiter, False, rel, "max iterations")


def successive_over_relaxation_pure(A, b, relaxation: float = 1.0, tol: float = 1e-8, maxiter: int = 10_000):
    """Метод последовательной верхней релаксации на чистом Python.

    Параметр ``relaxation`` задаёт степень релаксации и должен находиться
    строго между нулём и двумя. Это ограничение само по себе не гарантирует
    сходимость для произвольной матрицы.
    """
    _check_control(tol, maxiter)
    if not np.isfinite(relaxation) or not 0.0 < relaxation < 2.0:
        raise ValueError("для метода последовательной верхней релаксации параметр должен быть больше нуля и меньше двух")
    a, rhs = _pure_arrays(A, b)
    n = len(rhs)
    if any(a[i][i] == 0.0 for i in range(n)):
        raise ZeroDivisionError("метод последовательной верхней релаксации требует ненулевые диагональные элементы")
    x = [0.0] * n
    bnorm = _norm_pure(rhs)
    denom = bnorm if bnorm > 0 else 1.0
    for k in range(maxiter + 1):
        r = _residual_pure(a, rhs, x)
        rel = _norm_pure(r) / denom
        if not np.isfinite(rel) or rel > DIVERGED:
            return x, _info(k, False, rel, "diverged")
        if rel <= tol:
            return x, _info(k, True, rel)
        if k == maxiter:
            break
        correction = [0.0] * n
        for i in range(n):
            value = r[i] - sum(a[i][j] * correction[j] for j in range(i))
            correction[i] = relaxation * value / a[i][i]
        x = [x[i] + correction[i] for i in range(n)]
    return x, _info(maxiter, False, rel, "max iterations")


def gauss_seidel_pure(A, b, tol: float = 1e-8, maxiter: int = 10_000):
    return successive_over_relaxation_pure(A, b, relaxation=1.0, tol=tol, maxiter=maxiter)


def cg_pure(A, b, tol: float = 1e-8, maxiter: int = 10_000):
    """Непредобусловленный метод сопряжённых градиентов на Python.

    Требуется симметричная положительно определённая матрица."""
    _check_control(tol, maxiter)
    a, rhs = _pure_arrays(A, b)
    _check_symmetric(np.asarray(a, dtype=np.float64))
    n = len(rhs)
    x = [0.0] * n
    r = rhs.copy()
    p = r.copy()
    bnorm = _norm_pure(rhs)
    target = tol * (bnorm if bnorm > 0 else 1.0)
    rr = sum(value * value for value in r)
    if math.sqrt(rr) <= target:
        return x, _info(0, True, _norm_pure(r) / (bnorm if bnorm > 0 else 1.0))
    k = 0
    status = None
    while k < maxiter:
        rel = math.sqrt(rr) / (bnorm if bnorm > 0 else 1.0)
        if not math.isfinite(rel) or rel > DIVERGED:
            status = "diverged"
            break
        Ap = [sum(aij * pj for aij, pj in zip(row, p)) for row in a]
        pAp = sum(pi * api for pi, api in zip(p, Ap))
        if not math.isfinite(pAp) or pAp <= 0.0:
            status = "breakdown"
            break
        alpha = rr / pAp
        x = [x[i] + alpha * p[i] for i in range(n)]
        r = [r[i] - alpha * Ap[i] for i in range(n)]
        k += 1
        rr_new = sum(value * value for value in r)
        if math.sqrt(rr_new) <= target:
            # Проверяем настоящую невязку, а не только рекуррентную.
            r_true = [rhs[i] - sum(a[i][j] * x[j] for j in range(n)) for i in range(n)]
            rr_true = sum(value * value for value in r_true)
            if math.sqrt(rr_true) <= target:
                r, rr = r_true, rr_true
                break
            r, rr = r_true, rr_true
            p = r.copy()  # надёжный перезапуск из истинной невязки
            continue
        beta = rr_new / rr
        p = [r[i] + beta * p[i] for i in range(n)]
        rr = rr_new
    if status is None and k >= maxiter and math.sqrt(rr) > target:
        status = "max iterations"
    true_r = [rhs[i] - sum(a[i][j] * x[j] for j in range(n)) for i in range(n)]
    rel_true = _norm_pure(true_r) / (bnorm if bnorm > 0 else 1.0)
    converged = math.isfinite(rel_true) and rel_true <= tol
    if status is None and not converged:
        status = "max iterations"
    return x, _info(k, converged, rel_true, status)


def jacobi(A, b, tol: float = 1e-8, maxiter: int = 10_000, time_limit=None):
    _check_control(tol, maxiter, time_limit)
    matrix, rhs = _check_system(A, b)
    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    if np.any(diagonal == 0.0):
        raise ZeroDivisionError("метод Якоби требует ненулевые диагональные элементы")
    x, r, bnorm, rel = _initial_state(matrix, rhs)
    start = time.perf_counter()
    for k in range(maxiter + 1):
        rel = _relative_residual(r, bnorm)
        if rel <= tol:
            return _final_info(matrix, rhs, x, k, tol)
        if k == maxiter:
            break
        state = _early_status(rel, start, time_limit)
        if state:
            return _final_info(matrix, rhs, x, k, tol, state)
        x += r / diagonal
        r = rhs - matrix @ x
    return _final_info(matrix, rhs, x, maxiter, tol, "max iterations")


def _successive_over_relaxation_solver(A, relaxation: float):
    diagonal = np.asarray(A.diagonal(), dtype=np.float64)
    if np.any(diagonal == 0.0):
        raise ZeroDivisionError("метод последовательной верхней релаксации требует ненулевые диагональные элементы")
    if sp.issparse(A):
        lower = (sp.tril(A, k=-1) + sp.diags_array(diagonal / relaxation)).tocsr()
        return sparse_triangular_solver(lower)
    lower = np.tril(np.asarray(A, dtype=np.float64), k=-1)
    lower[np.diag_indices_from(lower)] = diagonal / relaxation
    return lambda r: sla.solve_triangular(lower, r, lower=True, check_finite=False)


def successive_over_relaxation(A, b, relaxation: float = 1.0, tol: float = 1e-8, maxiter: int = 10_000, time_limit=None):
    """Метод последовательной верхней релаксации для плотных и разреженных матриц.

    Параметр ``relaxation`` задаёт степень релаксации и должен находиться
    строго между нулём и двумя. Это ограничение само по себе не гарантирует
    сходимость для произвольной матрицы.
    """
    _check_control(tol, maxiter, time_limit)
    if not np.isfinite(relaxation) or not 0.0 < relaxation < 2.0:
        raise ValueError("для метода последовательной верхней релаксации параметр должен быть больше нуля и меньше двух")
    matrix, rhs = _check_system(A, b)
    solve_correction = _successive_over_relaxation_solver(matrix, relaxation)
    x, r, bnorm, _ = _initial_state(matrix, rhs)
    start = time.perf_counter()
    for k in range(maxiter + 1):
        rel = _relative_residual(r, bnorm)
        if rel <= tol:
            return _final_info(matrix, rhs, x, k, tol)
        if k == maxiter:
            break
        state = _early_status(rel, start, time_limit)
        if state:
            return _final_info(matrix, rhs, x, k, tol, state)
        x += solve_correction(r)
        r = rhs - matrix @ x
    return _final_info(matrix, rhs, x, maxiter, tol, "max iterations")


def gauss_seidel(A, b, tol: float = 1e-8, maxiter: int = 10_000, time_limit=None):
    return successive_over_relaxation(A, b, relaxation=1.0, tol=tol, maxiter=maxiter, time_limit=time_limit)


def cg(A, b, tol: float = 1e-8, maxiter: int = 10_000, M_solve=None, time_limit=None):
    """Предобусловленный метод сопряжённых градиентов. Он применим к системам
    с симметричной положительно определённой матрицей.

    Положительная определённость матрицы должна быть обоснована вызывающим кодом.
    ``M_solve`` должен задавать симметричный положительно определённый оператор.
    """
    _check_control(tol, maxiter, time_limit)
    matrix, rhs = _check_system(A, b)
    _check_symmetric(matrix)
    x, r, bnorm, _ = _initial_state(matrix, rhs)
    target = tol * (bnorm if bnorm > 0 else 1.0)
    # Если правая часть нулевая, решение уже найдено при x=0: не вызываем предобусловливатель
    # и не допускаем ложной ошибки из-за его поведения на нулевом векторе.
    if np.linalg.norm(r) <= target:
        return _final_info(matrix, rhs, x, 0, tol)
    z = np.asarray(M_solve(r) if M_solve is not None else r, dtype=np.float64)
    if z.shape != rhs.shape or not np.isfinite(z).all():
        raise ValueError("предобусловливатель вернул некорректный вектор")
    p = z.copy()
    rz = float(r @ z)
    if not np.isfinite(rz) or rz <= 0.0:
        return _final_info(matrix, rhs, x, 0, tol, "breakdown")

    start = time.perf_counter()
    status = None
    k = 0
    while k < maxiter:
        rel = _relative_residual(r, bnorm)
        status = _early_status(rel, start, time_limit)
        if status:
            break
        Ap = matrix @ p
        pAp = float(p @ Ap)
        if not np.isfinite(pAp) or pAp <= 0.0:
            status = "breakdown"
            break
        alpha = rz / pAp
        if not np.isfinite(alpha):
            status = "breakdown"
            break
        x += alpha * p
        r -= alpha * Ap
        k += 1

        # При достижении рекуррентного допуска проверяем настоящую невязку.
        # Если накопилась ошибка округления, перезапускаем направление поиска.
        if np.linalg.norm(r) <= target:
            r_true = rhs - matrix @ x
            if np.linalg.norm(r_true) <= target:
                r = r_true
                break
            r = r_true
            z = np.asarray(M_solve(r) if M_solve is not None else r, dtype=np.float64)
            if z.shape != rhs.shape or not np.isfinite(z).all():
                status = "breakdown"
                break
            rz = float(r @ z)
            if not np.isfinite(rz) or rz <= 0.0:
                status = "breakdown"
                break
            p = z.copy()
            continue

        z = np.asarray(M_solve(r) if M_solve is not None else r, dtype=np.float64)
        if z.shape != rhs.shape or not np.isfinite(z).all():
            status = "breakdown"
            break
        rz_new = float(r @ z)
        if not np.isfinite(rz_new) or rz_new <= 0.0 or rz == 0.0:
            status = "breakdown"
            break
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    if status is None and k >= maxiter and np.linalg.norm(r) > target:
        status = "max iterations"
    return _final_info(matrix, rhs, x, k, tol, status)


def successive_over_relaxation_optimal_parameter_poisson(N: int) -> float:
    """Теоретический оптимальный параметр последовательной верхней релаксации для стандартной двумерной сетки Пуассона."""
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N должен быть положительным целым")
    return 2.0 / (1.0 + math.sin(math.pi / (N + 1)))


def symmetric_successive_over_relaxation_preconditioner(A, relaxation: float = 1.0):
    """Создать симметричный предобуславливатель на основе верхней релаксации.

    Параметр ``relaxation`` задаёт степень релаксации и должен находиться
    строго между нулём и двумя. Матрица должна быть симметричной и иметь
    положительные диагональные элементы.
    """
    if not sp.issparse(A) or A.shape[0] != A.shape[1]:
        raise ValueError("Этот предобуславливатель в данной реализации поддерживается для квадратных разреженных матриц")
    if not np.isfinite(relaxation) or not 0.0 < relaxation < 2.0:
        raise ValueError("параметр релаксации для симметричного предобуславливателя должен быть больше нуля и меньше двух")
    diagonal = np.asarray(A.diagonal(), dtype=np.float64)
    if not np.isfinite(A.data).all():
        raise ValueError("Предобуславливатель требует конечные элементы матрицы")
    if np.any(diagonal <= 0.0):
        raise ValueError("Для предобуславливателя требуется положительная диагональ")
    difference = A - A.T
    asymmetry = float(abs(difference).max()) if difference.nnz else 0.0
    scale = float(abs(A).max()) if A.nnz else 0.0
    symmetry_tol = 100.0 * np.finfo(float).eps * max(A.shape) * max(scale, np.finfo(float).tiny)
    if asymmetry > symmetry_tol:
        raise ValueError("Для предобуславливателя требуется симметричная матрица")
    lower = (sp.tril(A, k=-1) + sp.diags_array(diagonal / relaxation)).tocsr()
    upper = lower.T.tocsr()
    solve_lower = sparse_triangular_solver(lower)
    solve_upper = sparse_triangular_solver(upper)
    scale = (2.0 - relaxation) / relaxation

    def solve(r):
        rhs = np.asarray(r, dtype=np.float64)
        return solve_upper(scale * diagonal * solve_lower(rhs))
    return solve


def ilu_preconditioner(A, drop_tol: float = 1e-4, fill_factor: float = 10.0):
    """Предобуславливатель на основе неполного LU-разложения.

    Используется со стабилизированным методом бисопряжённых градиентов
    или обобщённым методом минимальной невязки. Не предназначен для
    метода сопряжённых градиентов."""
    if not sp.issparse(A) or A.shape[0] != A.shape[1]:
        raise ValueError("неполное LU-разложение требует квадратную разреженную матрицу")
    if drop_tol < 0 or not np.isfinite(drop_tol) or fill_factor < 1 or not np.isfinite(fill_factor):
        raise ValueError("некорректные параметры неполного LU-разложения")
    ilu = spla.spilu(A.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    return spla.LinearOperator(A.shape, matvec=ilu.solve, dtype=np.float64)


class _IterationLimit(Exception):
    pass


class _Callback:
    def __init__(self, time_limit=None):
        self.count = 0
        self.x = None
        self.start = time.perf_counter()
        self.time_limit = time_limit

    def __call__(self, xk=None, *_):
        self.count += 1
        if isinstance(xk, np.ndarray):
            self.x = np.asarray(xk, dtype=np.float64).copy()
        if self.time_limit is not None and time.perf_counter() - self.start >= self.time_limit:
            raise _IterationLimit


def _scipy_run(solver, A, b, tol, iteration_limit, time_limit,
               require_symmetric: bool = False, **kwargs):
    _check_control(tol, iteration_limit, time_limit)
    matrix, rhs = _check_system(A, b)
    if require_symmetric:
        _check_symmetric(matrix)
    counter = _Callback(time_limit)
    status = None
    code = None
    try:
        x, code = solver(matrix, rhs, callback=counter, **kwargs)
    except _IterationLimit:
        x = counter.x if counter.x is not None else np.zeros_like(rhs)
        status = "time limit"
    except (FloatingPointError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        x = counter.x if counter.x is not None else np.zeros_like(rhs)
        status = "breakdown"
        err = str(exc)
    if status == "time limit":
        return _final_info(matrix, rhs, x, counter.count, tol, status)
    if status == "breakdown":
        return _final_info(matrix, rhs, x, counter.count, tol, status, error=err)
    if not np.isfinite(x).all():
        return _final_info(matrix, rhs, np.zeros_like(rhs), counter.count, tol, "breakdown")
    _, info = _final_info(matrix, rhs, x, counter.count, tol,
                          "converged" if code == 0 else ("breakdown" if code is not None and code < 0 else "max iterations"))
    # Нулевой код SciPy сам по себе недостаточен: независимая проверка
    # фактической невязки имеет приоритет.
    if code == 0 and not info["converged"]:
        info["status"] = "verification_failed"
    return x, info


def scipy_cg(A, b, tol: float = 1e-8, maxiter: int = 10_000, M=None, time_limit=None):
    """Реализация метода сопряжённых градиентов из SciPy.

    Требуются симметричная положительно определённая матрица и
    симметричный положительно определённый предобуславливатель."""
    return _scipy_run(spla.cg, A, b, tol, maxiter, time_limit,
                      require_symmetric=True, rtol=tol, atol=0.0,
                      maxiter=maxiter, M=M)


def scipy_bicgstab(A, b, tol: float = 1e-8, maxiter: int = 10_000, M=None, time_limit=None):
    """Стабилизированный метод бисопряжённых градиентов из SciPy.

    Подходит для общих невырожденных разреженных систем."""
    return _scipy_run(spla.bicgstab, A, b, tol, maxiter, time_limit,
                      rtol=tol, atol=0.0, maxiter=maxiter, M=M)


def scipy_gmres(A, b, tol: float = 1e-8, maxiter: int = 10_000, M=None,
                restart: int = 50, time_limit=None):
    """Обобщённый метод минимальной невязки из SciPy. Параметр ``maxiter``
    задаёт число циклов перезапуска.

    Тайм-аут проверяется на границе цикла перезапуска, потому что callback с текущим
    приближением в SciPy вызывается именно на этой границе.
    """
    if not isinstance(restart, (int, np.integer)) or restart < 1:
        raise ValueError("число шагов между перезапусками должно быть положительным целым")
    return _scipy_run(spla.gmres, A, b, tol, maxiter, time_limit,
                      rtol=tol, atol=0.0, maxiter=maxiter, M=M,
                      restart=restart, callback_type="x")
