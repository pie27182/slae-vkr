"""Проверки математической корректности и правил вызова функций."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp

from experiments.real_matrices import load_matrix, methods, properties
from slae import bench, direct, iterative, matrices


RTOL_DIRECT = 1e-10
RTOL_ITER = 2e-7


@pytest.fixture(scope="module")
def general():
    A = matrices.dense_general(24, seed=12)
    x, b = matrices.right_hand_side_for(A, seed=31)
    return A, b, x


@pytest.fixture(scope="module")
def dd():
    A = matrices.dense_diag_dominant(30, seed=19)
    x, b = matrices.right_hand_side_for(A, seed=32)
    return A, b, x


@pytest.fixture(scope="module")
def symmetric_positive_definite():
    A = matrices.dense_symmetric_positive_definite(25, seed=20)
    x, b = matrices.right_hand_side_for(A, seed=33)
    return A, b, x


@pytest.fixture(scope="module")
def poisson():
    A = matrices.poisson2d(8)
    x, b = matrices.right_hand_side_for(A, seed=34)
    return A, b, x


@pytest.mark.parametrize("solver", [
    direct.gauss_numpy,
    direct.solve_numpy,
    direct.solve_lu_scipy,
    direct.solve_qr_scipy,
    direct.solve_svd_lstsq,
])
def test_general_direct_methods(solver, general):
    A, b, x = general
    result = solver(A, b)
    assert bench.rel_error(result, x) < RTOL_DIRECT
    assert bench.rel_residual(A, result, b) < RTOL_DIRECT


def test_gauss_pure(general):
    A, b, x = general
    result = direct.gauss_pure(A.tolist(), b.tolist())
    assert bench.rel_error(result, x) < RTOL_DIRECT


def test_gauss_does_not_mutate_inputs(general):
    A, b, _ = general
    A0, b0 = A.copy(), b.copy()
    direct.gauss_numpy(A, b)
    assert np.array_equal(A, A0)
    assert np.array_equal(b, b0)


def test_gauss_partial_pivoting_is_necessary():
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    assert np.allclose(direct.gauss_numpy(A, b), [1.0, 1.0])
    with pytest.raises(direct.SingularMatrixError):
        direct.gauss_numpy(A, b, pivot=False)


def test_gauss_rejects_singular_matrix():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(direct.SingularMatrixError):
        direct.gauss_numpy(A, np.ones(2))


@pytest.mark.parametrize("bad_A,b", [
    (np.ones((2, 3)), np.ones(2)),
    (np.ones((2, 2)), np.ones((2, 1))),
    (np.empty((0, 0)), np.empty(0)),
])
def test_direct_methods_validate_shapes(bad_A, b):
    with pytest.raises(ValueError):
        direct.solve_numpy(bad_A, b)


@pytest.mark.parametrize("solver", [direct.cholesky_pure, direct.cholesky_numpy,
                                     direct.solve_cholesky_scipy])
def test_cholesky_methods(solver, symmetric_positive_definite):
    A, b, x = symmetric_positive_definite
    result = solver(A.tolist(), b.tolist()) if solver is direct.cholesky_pure else solver(A, b)
    assert bench.rel_error(result, x) < RTOL_DIRECT


def test_cholesky_rejects_symmetric_indefinite_matrix():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(direct.SingularMatrixError):
        direct.cholesky_numpy(A, np.ones(2))


def test_cholesky_rejects_nonsymmetric_matrix():
    A = np.array([[2.0, 1.0], [0.0, 2.0]])
    with pytest.raises(ValueError):
        direct.cholesky_numpy(A, np.ones(2))


@pytest.mark.parametrize("n", [1, 2, 37])
def test_thomas_solvers(n):
    lower, diag, upper = matrices.tridiagonal_components(n, seed=40 + n)
    x = np.random.default_rng(41 + n).standard_normal(n)
    b = matrices.tridiagonal_matvec(lower, diag, upper, x)
    diagonal_storage = matrices.tridiagonal_to_scipy_storage(lower, diag, upper)
    sparse = matrices.tridiagonal_to_sparse(lower, diag, upper)
    solvers = [
        direct.thomas_pure(lower.tolist(), diag.tolist(), upper.tolist(), b.tolist()),
        direct.thomas_numpy_loop(lower, diag, upper, b),
        direct.solve_tridiagonal_scipy(diagonal_storage, b),
        direct.solve_sparse_lu(sparse, b),
    ]
    for result in solvers:
        assert bench.rel_error(result, x) < RTOL_DIRECT


def test_thomas_reports_zero_pivot():
    with pytest.raises(direct.SingularMatrixError):
        direct.thomas_pure([0.0, 1.0], [0.0, 2.0], [1.0, 0.0], [1.0, 1.0])


@pytest.mark.parametrize("solver", [iterative.jacobi, iterative.gauss_seidel])
def test_stationary_methods(solver, dd):
    A, b, x = dd
    result, info = solver(A, b, tol=1e-10, maxiter=10_000)
    assert info["converged"]
    assert info["status"] == "converged"
    assert info["resid"] <= 1e-10
    assert bench.rel_error(result, x) < RTOL_ITER


@pytest.mark.parametrize("pure,vectorized", [
    (iterative.jacobi_pure, iterative.jacobi),
    (iterative.gauss_seidel_pure, iterative.gauss_seidel),
])
def test_stationary_python_and_numpy_implementations(pure, vectorized, dd):
    A, b, x = dd
    x1, i1 = pure(A.tolist(), b.tolist(), tol=1e-9, maxiter=10_000)
    x2, i2 = vectorized(A, b, tol=1e-9, maxiter=10_000)
    assert i1["converged"] and i2["converged"]
    assert bench.rel_error(x1, x) < RTOL_ITER
    assert bench.rel_error(x2, x) < RTOL_ITER


def test_successive_over_relaxation_python_and_numpy(dd):
    A, b, x = dd
    x1, i1 = iterative.successive_over_relaxation_pure(A.tolist(), b.tolist(), relaxation=1.1, tol=1e-9)
    x2, i2 = iterative.successive_over_relaxation(A, b, relaxation=1.1, tol=1e-9)
    assert i1["converged"] and i2["converged"]
    assert bench.rel_error(x1, x) < RTOL_ITER
    assert bench.rel_error(x2, x) < RTOL_ITER


def test_stationary_methods_reject_zero_diagonal():
    A = np.array([[0.0, 1.0], [1.0, 2.0]])
    with pytest.raises(ZeroDivisionError):
        iterative.jacobi(A, np.ones(2))


def test_successive_over_relaxation_validates_parameter(dd):
    A, b, _ = dd
    with pytest.raises(ValueError):
        iterative.successive_over_relaxation(A, b, relaxation=2.0)


@pytest.mark.parametrize("solver", [iterative.cg, iterative.scipy_cg])
def test_cg_methods_on_symmetric_positive_definite_matrix(solver, symmetric_positive_definite):
    A, b, x = symmetric_positive_definite
    result, info = solver(A, b, tol=1e-10, maxiter=1000)
    assert info["converged"]
    assert info["resid"] <= 1e-10
    assert bench.rel_error(result, x) < RTOL_ITER


def test_cg_pure(symmetric_positive_definite):
    A, b, x = symmetric_positive_definite
    result, info = iterative.cg_pure(A.tolist(), b.tolist(), tol=1e-10, maxiter=1000)
    assert info["converged"]
    assert bench.rel_error(result, x) < RTOL_ITER


def test_preconditioned_cg_poisson():
    A, x, b = matrices.poisson2d_problem(10)
    preconditioner = iterative.symmetric_successive_over_relaxation_preconditioner(A, relaxation=1.0)
    result, info = iterative.cg(A, b, tol=1e-9, maxiter=5000, M_solve=preconditioner)
    assert info["converged"]
    assert info["resid"] <= 1e-9
    assert bench.rel_error(result, x) < RTOL_ITER


def test_cg_detects_indefinite_breakdown():
    A = np.diag([1.0, -1.0])
    result, info = iterative.cg(A, np.ones(2), maxiter=10)
    assert not info["converged"]
    assert info["status"] == "breakdown"


@pytest.mark.parametrize("solver", [iterative.scipy_bicgstab, iterative.scipy_gmres])
def test_general_krylov_methods(solver, general):
    A, b, x = general
    result, info = solver(A, b, tol=1e-10, maxiter=1000)
    assert info["converged"]
    assert info["resid"] <= 1e-10
    assert bench.rel_error(result, x) < RTOL_ITER


def test_krylov_methods_sparse(poisson):
    A, b, x = poisson
    for solver in (iterative.cg, iterative.scipy_cg, iterative.scipy_bicgstab, iterative.scipy_gmres):
        result, info = solver(A, b, tol=1e-9, maxiter=5000)
        assert info["converged"], (solver.__name__, info)
        assert bench.rel_error(result, x) < RTOL_ITER


def test_zero_rhs_is_immediately_solved():
    A = np.eye(4)
    x, info = iterative.cg(A, np.zeros(4))
    assert np.array_equal(x, np.zeros(4))
    assert info["converged"] and info["iters"] == 0 and info["resid"] == 0.0


def test_poisson_model_properties():
    A = matrices.poisson2d(5)
    assert A.shape == (25, 25)
    assert (A - A.T).nnz == 0
    assert np.all(np.linalg.eigvalsh(A.toarray()) > 0)


def test_matrix_generators_have_documented_properties():
    A = matrices.with_condition(20, 1e4, seed=8)
    assert np.isclose(np.linalg.cond(A), 1e4, rtol=1e-7)
    S = matrices.with_condition(20, 1e3, seed=9, positive_definite=True)
    assert np.allclose(S, S.T)
    assert np.all(np.linalg.eigvalsh(S) > 0)
    assert np.isclose(np.linalg.cond(S), 1e3, rtol=1e-7)
    D = matrices.dense_diag_dominant(20, seed=10)
    off = np.abs(D).sum(axis=1) - np.abs(np.diag(D))
    assert np.all(np.abs(np.diag(D)) > off)
    R = matrices.sparse_diag_dominant(100, 8, seed=11)
    off_sparse = np.asarray(abs(R).sum(axis=1)).ravel() - np.abs(R.diagonal())
    assert np.all(np.abs(R.diagonal()) > off_sparse)


def test_generator_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        matrices.poisson2d(0)
    with pytest.raises(ValueError):
        matrices.with_condition(10, 0.5)
    with pytest.raises(ValueError):
        matrices.sparse_diag_dominant(10, 0)


def test_matrix_market_loader_roundtrip(tmp_path):
    matrix = matrices.poisson2d(4)
    path = tmp_path / "poisson.mtx"
    sio.mmwrite(path, matrix)
    loaded = load_matrix(path)
    assert loaded.shape == matrix.shape
    assert (loaded - matrix).nnz == 0


def test_symmetry_does_not_prove_positive_definiteness():
    matrix = sp.csr_array([[1.0, 2.0], [2.0, 1.0]])
    props = properties("indefinite", matrix, {"posdef": "0"})
    assert props["symmetric"] is True
    assert props["positive_definite_metadata"] is False
    entries = methods(matrix, props, 1.0, max_dense_n=10, max_lu_nnz=100, max_pure_n=10)
    cg_entry = next(entry for entry in entries if entry[0] == "Метод сопряжённых градиентов")
    assert cg_entry[5] is False


def test_matrix_without_positive_definite_metadata_does_not_run_conjugate_gradients():
    matrix = matrices.poisson2d(3)
    props = properties("unknown_positive_definite", matrix, {})
    entries = methods(matrix, props, 1.0, max_dense_n=100, max_lu_nnz=1000, max_pure_n=20)
    assert not next(entry for entry in entries if entry[0] == "Метод сопряжённых градиентов")[5]


def test_benchmark_measurement_contract():
    measured = bench.measure(lambda x: x * 2, lambda: (3,), budget=0.05,
                             min_repeats=2, max_repeats=3, raw=True)
    assert measured["result"] == 6
    assert measured["repeats"] >= 2
    assert measured["time_median"] > 0
    assert len(measured["times"]) == measured["repeats"]


def test_benchmark_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        bench.measure(lambda: 1, lambda: (), budget=0)
    with pytest.raises(ValueError):
        bench.fit_power([1, 0], [0, 1])


def test_cg_zero_rhs_does_not_call_preconditioner():
    def must_not_run(_):
        raise AssertionError("предобуславливатель не должен вызываться для нулевой правой части")

    x, info = iterative.cg(np.eye(3), np.zeros(3), M_solve=must_not_run)
    assert np.array_equal(x, np.zeros(3))
    assert info["converged"] and info["iters"] == 0


def test_scipy_success_code_is_rechecked_by_true_residual(monkeypatch):
    from slae import iterative as module

    def fake_solver(A, b, callback=None, **kwargs):
        return np.ones_like(b), 0  # SciPy says success, but this is not a solution.

    x, info = module._scipy_run(fake_solver, np.eye(3), np.zeros(3),
                                tol=1e-12, iteration_limit=10, time_limit=None)
    assert np.array_equal(x, np.ones(3))
    assert not info["converged"]
    assert info["status"] == "verification_failed"


def test_backward_error_uses_consistent_infinity_norm():
    A = np.diag([100.0, 1.0])
    x = np.array([0.9, 100.0])
    b = np.array([100.0, 100.0])
    expected = 10.0 / (100.0 * 100.0 + 100.0)
    assert np.isclose(bench.backward_error(A, x, b), expected)


def test_matrix_index_cache_uses_custom_data_directory(tmp_path, monkeypatch):
    from experiments import download_matrices as downloader

    fields = downloader.FIELDS
    row = ["HB", "toy", "2", "2", "2", "1", "0", "0", "1", "1", "1", "test"]
    payload = (",".join(fields) + "\n" + ",".join(row) + "\n").encode("utf-8")
    monkeypatch.setattr(downloader, "get_bytes", lambda url, timeout=120.0: payload)
    parsed = downloader.read_index(data_dir=tmp_path, refresh=True)
    assert "toy" in parsed and parsed["toy"][0]["group"] == "HB"
    assert (tmp_path / "ssstats.csv").exists()
    assert not (downloader.ROOT / "data" / "matrices" / "ssstats.csv").exists()


def test_download_matrix_existing_file_checks_expected_dimensions(tmp_path):
    from experiments.download_matrices import download_one

    path = tmp_path / "toy.mtx"
    path.write_text("%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n2 2 1\n",
                    encoding="ascii")
    status = download_one("toy", "HB", tmp_path, {"nrows": "2", "ncols": "2"})
    assert "уже существует" in status
    with pytest.raises(ValueError, match="не совпадает"):
        download_one("toy", "HB", tmp_path, {"nrows": "3", "ncols": "3"})


def test_download_matrix_rejects_path_traversal(tmp_path):
    from experiments.download_matrices import download_one

    with pytest.raises(ValueError, match="недопустимое"):
        download_one("../escape", "HB", tmp_path)


def test_real_report_handles_only_matrix_load_errors(tmp_path):
    from experiments.real_matrices import report

    out = tmp_path / "results"
    figures = tmp_path / "figures"
    out.mkdir()
    (out / "real.csv").write_text(
        "matrix,n,nnz,method,family,seed,status,time\n", encoding="utf-8"
    )
    (out / "matrices.csv").write_text(
        "matrix,status,error\nbad,load_error,invalid file\n", encoding="utf-8"
    )
    report(out, figures)
    assert (out / "summary.md").exists()
    assert "Нет строк с результатами" in (out / "summary.md").read_text(encoding="utf-8")


def test_sparse_gauss_seidel_supports_csr_array():
    A = matrices.sparse_diag_dominant(80, nnz_per_row=8, seed=711)
    x_true, b = matrices.right_hand_side_for(A, seed=712)
    x, info = iterative.gauss_seidel(A, b, tol=1e-8, maxiter=5000, time_limit=5.0)
    assert info["converged"], info
    assert bench.rel_residual(A, x, b) <= 1e-8
    assert bench.rel_error(x, x_true) < 1e-6


def test_symmetric_successive_over_relaxation_rejects_nonsymmetric_matrix():
    matrix = sp.csr_array([[4.0, 1.0], [0.0, 3.0]])
    with pytest.raises(ValueError, match="симметричная матрица"):
        iterative.symmetric_successive_over_relaxation_preconditioner(matrix)


def test_symmetric_successive_over_relaxation_rejects_nonpositive_diagonal():
    matrix = sp.csr_array([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(ValueError, match="положительная диагональ"):
        iterative.symmetric_successive_over_relaxation_preconditioner(matrix)


def test_benchmark_rejects_nonfinite_batch_target():
    with pytest.raises(ValueError):
        bench.measure(lambda: 1, lambda: (), target_batch_seconds=float("nan"))


def test_benchmark_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        bench.measure(lambda: 1, lambda: (), max_batch_size=1.5)


def test_backward_error_sparse_matrix_matches_dense():
    dense = np.array([[3.0, -1.0], [-1.0, 2.0]])
    sparse = sp.csr_array(dense)
    x = np.array([0.5, 1.5])
    b = np.array([1.0, 2.0])
    assert np.isclose(bench.backward_error(dense, x, b),
                      bench.backward_error(sparse, x, b))


@pytest.mark.parametrize("solver", [iterative.cg, iterative.scipy_cg])
def test_cg_rejects_nonsymmetric_matrix(solver):
    matrix = np.array([[3.0, 2.0], [0.0, 2.0]])
    with pytest.raises(ValueError, match="симметричную матрицу"):
        solver(matrix, np.ones(2))


def test_cg_pure_rejects_nonsymmetric_matrix():
    matrix = [[3.0, 2.0], [0.0, 2.0]]
    with pytest.raises(ValueError, match="симметричную матрицу"):
        iterative.cg_pure(matrix, [1.0, 1.0])


def test_matrix_file_sha256_is_streamed_and_stable(tmp_path):
    import hashlib
    from experiments.download_matrices import sha256_file as download_hash
    from experiments.real_matrices import file_sha256 as result_hash

    path = tmp_path / "matrix.mtx"
    payload = b"%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 1\n2 2 1\n"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert download_hash(path) == expected
    assert result_hash(path) == expected


def test_real_matrix_right_hand_side_seed_is_stable_and_name_specific():
    from experiments.real_matrices import _right_hand_side_seed

    assert _right_hand_side_seed(12345, "matrix_a") == _right_hand_side_seed(12345, "matrix_a")
    assert _right_hand_side_seed(12345, "matrix_a") != _right_hand_side_seed(12345, "matrix_b")
    assert _right_hand_side_seed(12345, "matrix_a") != _right_hand_side_seed(54321, "matrix_a")
