"""Генераторы тестовых систем и преобразования матриц."""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp


def _positive_int(value, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} должен быть положительным целым")
    return int(value)


def right_hand_side_for(A, seed: int = 0):
    """Сгенерировать известное решение и согласованную правую часть ``b=A@x``."""
    rng = np.random.default_rng(seed)
    x_true = rng.standard_normal(A.shape[0])
    return x_true, np.asarray(A @ x_true, dtype=np.float64)


def dense_diag_dominant(n: int, seed: int = 0) -> np.ndarray:
    """Плотная строго диагонально преобладающая матрица (не универсальная выборка)."""
    n = _positive_int(n, "n")
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(n, n))
    np.fill_diagonal(A, 0.0)
    np.fill_diagonal(A, np.sum(np.abs(A), axis=1) + 1.0)
    return A


def dense_general(n: int, seed: int = 0) -> np.ndarray:
    """Плотная матрица с независимыми стандартными нормальными элементами."""
    n = _positive_int(n, "n")
    return np.random.default_rng(seed).standard_normal((n, n))


def dense_symmetric_positive_definite(n: int, seed: int = 0) -> np.ndarray:
    """Плотная симметричная положительно определённая матрица ``B B.T + n I``."""
    n = _positive_int(n, "n")
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    A = B @ B.T
    A[np.diag_indices(n)] += n
    return A


def with_condition(n: int, cond: float, seed: int = 0, positive_definite: bool = False) -> np.ndarray:
    """Сконструировать матрицу с заданным спектром сингулярных значений.

    Целевая 2-норма обусловленности равна ``cond``. В арифметике float64
    фактическая обусловленность может отличаться от целевой при ``cond``,
    близком к обратной машинной точности; фактическое значение нужно измерять.
    """
    n = _positive_int(n, "n")
    if not np.isfinite(cond) or cond < 1:
        raise ValueError("число обусловленности должно быть конечным и не меньше единицы")
    rng = np.random.default_rng(seed)
    singular_values = np.geomspace(1.0, 1.0 / float(cond), n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    if positive_definite:
        return (Q * singular_values) @ Q.T
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (Q * singular_values) @ V.T


def hilbert(n: int) -> np.ndarray:
    """Матрица Гильберта: классический пример плохой обусловленности."""
    n = _positive_int(n, "n")
    return sla.hilbert(n)


def tridiagonal_components(n: int, seed: int = 0):
    """Создать строго диагонально преобладающую трёхдиагональную матрицу.

    Матрица задаётся тремя векторами: нижней диагональю, главной диагональю
    и верхней диагональю. Неиспользуемые крайние элементы нижнего и верхнего
    векторов равны нулю.
    """
    n = _positive_int(n, "n")
    rng = np.random.default_rng(seed)
    lower = rng.uniform(-1.0, 1.0, n)
    upper = rng.uniform(-1.0, 1.0, n)
    lower[0], upper[-1] = 0.0, 0.0
    diag = np.abs(lower) + np.abs(upper) + 1.0
    return lower, diag, upper


def _check_tridiagonal_components(lower, diag, upper):
    lo, di, up = (np.asarray(v, dtype=np.float64) for v in (lower, diag, upper))
    if di.ndim != 1 or di.size == 0 or lo.shape != di.shape or up.shape != di.shape:
        raise ValueError("lower, diag и upper должны быть непустыми векторами одинаковой длины")
    if not all(np.isfinite(v).all() for v in (lo, di, up)):
        raise ValueError("элементы трёх диагоналей должны быть конечными числами")
    if lo[0] != 0 or up[-1] != 0:
        raise ValueError("lower[0] и upper[-1] должны быть равны нулю")
    return lo, di, up


def tridiagonal_to_dense(lower, diag, upper) -> np.ndarray:
    lo, di, up = _check_tridiagonal_components(lower, diag, upper)
    n = len(di)
    A = np.zeros((n, n), dtype=np.float64)
    idx = np.arange(n)
    A[idx, idx] = di
    if n > 1:
        A[idx[1:], idx[:-1]] = lo[1:]
        A[idx[:-1], idx[1:]] = up[:-1]
    return A


def tridiagonal_to_sparse(lower, diag, upper) -> sp.csr_array:
    lo, di, up = _check_tridiagonal_components(lower, diag, upper)
    return sp.diags_array([lo[1:], di, up[:-1]], offsets=[-1, 0, 1], format="csr")


def tridiagonal_to_scipy_storage(lower, diag, upper) -> np.ndarray:
    """Подготовить три диагонали в формате, который принимает решатель SciPy."""
    lo, di, up = _check_tridiagonal_components(lower, diag, upper)
    n = len(di)
    diagonal_storage = np.zeros((3, n), dtype=np.float64)
    if n > 1:
        diagonal_storage[0, 1:] = up[:-1]
        diagonal_storage[2, :-1] = lo[1:]
    diagonal_storage[1, :] = di
    return diagonal_storage


def tridiagonal_matvec(lower, diag, upper, x) -> np.ndarray:
    lo, di, up = _check_tridiagonal_components(lower, diag, upper)
    vector = np.asarray(x, dtype=np.float64)
    if vector.ndim != 1 or vector.size != len(di):
        raise ValueError("x должен быть вектором той же длины, что и diag")
    y = di * vector
    y[1:] += lo[1:] * vector[:-1]
    y[:-1] += up[:-1] * vector[1:]
    return y


def poisson2d(N: int) -> sp.csr_array:
    """Пятиточечный оператор Лапласа на квадратной внутренней сетке ``N x N``.

    Матрица имеет размер ``N**2`` и симметрична положительно определена.
    Масштаб ``1/h**2`` опущен: он не меняет число обусловленности и структуру
    решения линейной системы, но должен учитываться в физической интерпретации исходного дифференциального уравнения.
    """
    N = _positive_int(N, "N")
    T = sp.diags_array([-np.ones(N - 1), 2.0 * np.ones(N), -np.ones(N - 1)],
                       offsets=[-1, 0, 1], format="csr")
    identity = sp.eye_array(N, format="csr")
    return (sp.kron(identity, T) + sp.kron(T, identity)).tocsr()


def poisson2d_problem(N: int):
    """Сформировать дискретную систему Пуассона с симметричной положительно определённой матрицей с известным точным решением.

    Вектор ``b`` намеренно вычисляется как ``A @ x_true``. Это проверяет
    решатели на одной и той же дискретной системе, но не является заявлением о
    конкретной непрерывной правой части уравнения в частных производных.
    """
    A = poisson2d(N)
    grid = np.arange(1, N + 1, dtype=np.float64) / (N + 1)
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    x_true = (X * (1 - X) * Y * (1 - Y) * np.exp(X + Y)).ravel()
    return A, x_true, np.asarray(A @ x_true, dtype=np.float64)


def sparse_diag_dominant(n: int, nnz_per_row: int = 10, seed: int = 0) -> sp.csr_array:
    """Случайная разреженная строго диагонально преобладающая матрица.

    Дубли случайно выбранных столбцов допустимы: координатное представление при преобразовании в построчное разреженное хранение объединяет их,
    после чего диагональ устанавливается равной сумме модулей внедиагональных
    элементов плюс единица.
    """
    n = _positive_int(n, "n")
    if not isinstance(nnz_per_row, (int, np.integer)) or nnz_per_row < 1:
        raise ValueError("число ненулевых элементов в строке должно быть положительным целым")
    nnz_per_row = min(int(nnz_per_row), n)
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), nnz_per_row)
    cols = rng.integers(0, n, size=n * nnz_per_row)
    values = rng.uniform(-1.0, 1.0, size=n * nnz_per_row)
    A = sp.coo_array((values, (rows, cols)), shape=(n, n)).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    diagonal = np.asarray(abs(A).sum(axis=1)).ravel() + 1.0
    A = (A + sp.diags_array(diagonal, format="csr")).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    return A
