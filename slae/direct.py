"""Прямые методы решения систем линейных уравнений ``A x = b``.

Здесь намеренно разделены учебные реализации алгоритмов и решатели,
предоставляемые библиотеками NumPy и SciPy.

Функции не изменяют входные массивы. Для методов без численной устойчивости
(например, метода Томаса без выбора главного элемента) явно документированы
ограничения и возвращается понятная ошибка при вырожденном ведущем элементе.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class SingularMatrixError(ArithmeticError):
    """Матрица вырождена или численно неразрешима выбранным алгоритмом."""


def _dense_system(A, b) -> tuple[np.ndarray, np.ndarray]:
    """Проверить и скопировать квадратную вещественную систему."""
    a = np.asarray(A, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A должна быть квадратной двумерной матрицей")
    if rhs.ndim != 1 or rhs.shape[0] != a.shape[0]:
        raise ValueError("b должна быть вектором длины A.shape[0]")
    if a.shape[0] == 0:
        raise ValueError("система нулевого размера не поддерживается")
    if not np.isfinite(a).all() or not np.isfinite(rhs).all():
        raise ValueError("A и b должны содержать только конечные числа")
    return a.copy(), rhs.copy()


def _pivot_tolerance(A: np.ndarray) -> float:
    """Масштабно-инвариантный порог для обнаружения численно нулевого ведущего элемента."""
    scale = float(np.linalg.norm(A, ord=np.inf))
    return np.finfo(np.float64).eps * max(A.shape[0], 1) * scale


def gauss_pure(A: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Метод Гаусса с частичным выбором главного элемента (чистый Python).

    Сложность: ``O(n^3)`` операций и ``O(n^2)`` дополнительной памяти.
    """
    a, rhs = _dense_system(A, b)
    n = len(rhs)
    tol = _pivot_tolerance(a)
    m = [list(map(float, a[i])) + [float(rhs[i])] for i in range(n)]

    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(m[i][k]))
        if abs(m[p][k]) <= tol:
            raise SingularMatrixError(f"численно нулевой ведущий элемент на шаге {k}")
        if p != k:
            m[k], m[p] = m[p], m[k]
        pivot = m[k][k]
        for i in range(k + 1, n):
            factor = m[i][k] / pivot
            m[i][k] = 0.0
            for j in range(k + 1, n + 1):
                m[i][j] -= factor * m[k][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        value = m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))
        if abs(m[i][i]) <= tol:
            raise SingularMatrixError(f"численно нулевой диагональный элемент на шаге {i}")
        x[i] = value / m[i][i]
    return x


def gauss_numpy(A, b, pivot: bool = True) -> np.ndarray:
    """Метод Гаусса с векторизованным обновлением подматрицы.

    ``pivot=False`` предусмотрен только для отдельного исследования влияния
    выбора главного элемента. Для произвольных матриц его использовать не следует.
    """
    a, rhs = _dense_system(A, b)
    n = len(rhs)
    tol = _pivot_tolerance(a)

    for k in range(n - 1):
        if pivot:
            p = k + int(np.argmax(np.abs(a[k:, k])))
            if p != k:
                a[[k, p], :] = a[[p, k], :]
                rhs[[k, p]] = rhs[[p, k]]
        if abs(a[k, k]) <= tol:
            raise SingularMatrixError(f"численно нулевой ведущий элемент на шаге {k}")
        factors = a[k + 1:, k] / a[k, k]
        a[k + 1:, k + 1:] -= factors[:, None] * a[k, k + 1:]
        rhs[k + 1:] -= factors * rhs[k]
        a[k + 1:, k] = 0.0

    if abs(a[-1, -1]) <= tol:
        raise SingularMatrixError("численно нулевой последний диагональный элемент")
    x = np.empty(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (rhs[i] - a[i, i + 1:] @ x[i + 1:]) / a[i, i]
    if not np.isfinite(x).all():
        raise SingularMatrixError("решение содержит нечисловые значения или бесконечности")
    return x


def cholesky_pure(A, b) -> list[float]:
    """Решить систему методом Холецкого. Требуется симметричная
    положительно определённая матрица."""
    a, rhs = _dense_system(A, b)
    n = len(rhs)
    scale = max(float(np.linalg.norm(a, ord=np.inf)), np.finfo(float).tiny)
    sym_tol = 100 * np.finfo(float).eps * n * scale
    if np.linalg.norm(a - a.T, ord=np.inf) > sym_tol:
        raise ValueError("для метода Холецкого матрица должна быть симметричной")
    tol = np.finfo(float).eps * n * scale
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = float(a[i, j]) - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if value <= tol:
                    raise SingularMatrixError("матрица не является численно положительно определённой")
                L[i][j] = math.sqrt(value)
            else:
                L[i][j] = value / L[j][j]

    y = [0.0] * n
    for i in range(n):
        y[i] = (float(rhs[i]) - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / L[i][i]
    return x


def cholesky_numpy(A, b) -> np.ndarray:
    """Учебная реализация разложения Холецкого с операциями NumPy."""
    a, rhs = _dense_system(A, b)
    n = len(rhs)
    scale = max(float(np.linalg.norm(a, ord=np.inf)), np.finfo(float).tiny)
    sym_tol = 100 * np.finfo(float).eps * n * scale
    if np.linalg.norm(a - a.T, ord=np.inf) > sym_tol:
        raise ValueError("для метода Холецкого матрица должна быть симметричной")
    tol = np.finfo(float).eps * n * scale
    L = np.zeros_like(a)
    for j in range(n):
        value = a[j, j] - L[j, :j] @ L[j, :j]
        if value <= tol:
            raise SingularMatrixError("матрица не является численно положительно определённой")
        L[j, j] = math.sqrt(value)
        if j + 1 < n:
            L[j + 1:, j] = (a[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    y = np.empty(n)
    for i in range(n):
        y[i] = (rhs[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.empty(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L[i + 1:, i] @ x[i + 1:]) / L[i, i]
    return x


def _tridiagonal_inputs(lower, diag, upper, b):
    lo = np.asarray(lower, dtype=np.float64)
    di = np.asarray(diag, dtype=np.float64)
    up = np.asarray(upper, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    if di.ndim != 1 or len(di) == 0:
        raise ValueError("diag должен быть непустым вектором")
    n = len(di)
    if any(v.ndim != 1 or len(v) != n for v in (lo, up, rhs)):
        raise ValueError("lower, diag, upper и b должны иметь одинаковую длину")
    if not all(np.isfinite(v).all() for v in (lo, di, up, rhs)):
        raise ValueError("элементы трёх диагоналей и правая часть должны быть конечными числами")
    if abs(lo[0]) > 0 or abs(up[-1]) > 0:
        raise ValueError("lower[0] и upper[-1] не используются и должны быть равны нулю")
    return lo.copy(), di.copy(), up.copy(), rhs.copy()


def thomas_pure(lower, diag, upper, b) -> list[float]:
    """Алгоритм Томаса для трёхдиагональной матрицы без выбора главного элемента."""
    lo, di, up, rhs = _tridiagonal_inputs(lower, diag, upper, b)
    n = len(di)
    scale = max(float(np.max(np.abs(di)) + np.max(np.abs(lo)) + np.max(np.abs(up))),
                np.finfo(float).tiny)
    tol = np.finfo(float).eps * n * scale
    cp = [0.0] * n
    dp = [0.0] * n
    if abs(di[0]) <= tol:
        raise SingularMatrixError("нулевой ведущий элемент в алгоритме Томаса")
    if n > 1:
        cp[0] = up[0] / di[0]
    dp[0] = rhs[0] / di[0]
    for i in range(1, n):
        pivot = di[i] - lo[i] * cp[i - 1]
        if abs(pivot) <= tol:
            raise SingularMatrixError(f"нулевой ведущий элемент алгоритма Томаса на шаге {i}")
        if i < n - 1:
            cp[i] = up[i] / pivot
        dp[i] = (rhs[i] - lo[i] * dp[i - 1]) / pivot
    x = [0.0] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def thomas_numpy_loop(lower, diag, upper, b) -> np.ndarray:
    """Алгоритм Томаса на массивах NumPy с последовательной рекуррентной схемой."""
    lo, di, up, rhs = _tridiagonal_inputs(lower, diag, upper, b)
    n = len(di)
    scale = max(float(np.max(np.abs(di)) + np.max(np.abs(lo)) + np.max(np.abs(up))),
                np.finfo(float).tiny)
    tol = np.finfo(float).eps * n * scale
    cp = np.zeros(n)
    dp = np.empty(n)
    if abs(di[0]) <= tol:
        raise SingularMatrixError("нулевой ведущий элемент в алгоритме Томаса")
    if n > 1:
        cp[0] = up[0] / di[0]
    dp[0] = rhs[0] / di[0]
    for i in range(1, n):
        pivot = di[i] - lo[i] * cp[i - 1]
        if abs(pivot) <= tol:
            raise SingularMatrixError(f"нулевой ведущий элемент алгоритма Томаса на шаге {i}")
        if i < n - 1:
            cp[i] = up[i] / pivot
        dp[i] = (rhs[i] - lo[i] * dp[i - 1]) / pivot
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def solve_tridiagonal_scipy(diagonal_storage, b) -> np.ndarray:
    """Решить трёхдиагональную систему библиотечным решателем SciPy."""
    # В SciPy используется имя solve_banded, но сюда передаются ровно три
    # диагонали: одна нижняя, главная и одна верхняя.
    diagonal_storage = np.asarray(diagonal_storage, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    if (diagonal_storage.ndim != 2 or diagonal_storage.shape[0] != 3
            or rhs.ndim != 1 or diagonal_storage.shape[1] != len(rhs)):
        raise ValueError("ожидается массив трёх диагоналей формы (3, n) и вектор правой части формы (n,)")
    if len(rhs) == 0 or not np.isfinite(diagonal_storage).all() or not np.isfinite(rhs).all():
        raise ValueError("входные данные должны быть непустыми и конечными")
    return sla.solve_banded((1, 1), diagonal_storage.copy(), rhs.copy(), check_finite=True)


def solve_numpy(A, b) -> np.ndarray:
    """Решить плотную систему стандартным решателем библиотеки NumPy."""
    a, rhs = _dense_system(A, b)
    return np.linalg.solve(a, rhs)


def solve_lu_scipy(A, b) -> np.ndarray:
    """Решить систему с помощью LU-разложения из библиотеки SciPy."""
    a, rhs = _dense_system(A, b)
    lu, piv = sla.lu_factor(a, check_finite=False)
    if np.any(np.abs(np.diag(lu)) <= _pivot_tolerance(a)):
        raise SingularMatrixError("LU-разложение обнаружило численно вырожденную матрицу")
    return sla.lu_solve((lu, piv), rhs, check_finite=False)


def solve_cholesky_scipy(A, b) -> np.ndarray:
    """Решить систему с симметричной положительно определённой матрицей
    методом Холецкого из SciPy."""
    a, rhs = _dense_system(A, b)
    scale = max(float(np.linalg.norm(a, ord=np.inf)), np.finfo(float).tiny)
    if np.linalg.norm(a - a.T, ord=np.inf) > 100 * np.finfo(float).eps * len(rhs) * scale:
        raise ValueError("для метода Холецкого матрица должна быть симметричной")
    c, lower = sla.cho_factor(a, lower=True, check_finite=False)
    return sla.cho_solve((c, lower), rhs, check_finite=False)


def solve_qr_scipy(A, b) -> np.ndarray:
    """Решить квадратную систему QR-разложением Хаусхолдера."""
    a, rhs = _dense_system(A, b)
    q, r = sla.qr(a, mode="economic", check_finite=False)
    tol = _pivot_tolerance(a)
    if np.any(np.abs(np.diag(r)) <= tol):
        raise SingularMatrixError("QR-разложение обнаружило численно вырожденную матрицу")
    return sla.solve_triangular(r, q.T @ rhs, check_finite=False)


def solve_svd_lstsq(A, b) -> np.ndarray:
    """Решить задачу наименьших квадратов с помощью сингулярного разложения; допускается недостаточный ранг матрицы."""
    a, rhs = _dense_system(A, b)
    return np.linalg.lstsq(a, rhs, rcond=None)[0]


def solve_sparse_lu(A, b) -> np.ndarray:
    """Решить разреженную систему с помощью LU-разложения из библиотеки SciPy."""
    if not sp.issparse(A):
        raise TypeError("A должна быть разреженной матрицей SciPy")
    matrix = sp.csc_matrix(A, dtype=np.float64, copy=True)
    rhs = np.asarray(b, dtype=np.float64)
    if matrix.shape[0] != matrix.shape[1] or rhs.ndim != 1 or rhs.size != matrix.shape[0]:
        raise ValueError("несогласованные размеры разреженной системы")
    if matrix.shape[0] == 0 or not np.isfinite(matrix.data).all() or not np.isfinite(rhs).all():
        raise ValueError("разреженная система должна быть непустой и конечной")
    return spla.splu(matrix).solve(rhs.copy())
