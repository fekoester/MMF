import os
import shutil
import subprocess

import numpy as np
import pytest

from mmf import master_memory_function, modified_state_matrix, scan, simulate_memory_capacity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [(-0.5, 0.4, 0.5, 20, 27), (-1.0, -0.6, 0.3, 16, 16), (-0.2, 0.15, 1.0, 12, 19)]


@pytest.mark.parametrize("a,b,theta,N,nu", CASES)
def test_mmf_matches_simulation_of_the_linear_map(a, b, theta, N, nu):
    mask = np.random.default_rng(3).uniform(0, 1, N)
    C = master_memory_function(a, b, theta, N, nu, mask=mask)
    L = min(len(C), 60)
    sim = simulate_memory_capacity(a, b, theta, N, nu, mask=mask, n_cycles=80_000, max_recall=L)
    assert np.abs(C[:L] - sim).max() < 0.01
    assert abs(C.sum() - sim.sum()) < 0.02 * C.sum() + 0.05


def test_legacy_convention_is_the_memory_function_of_a_one_step_longer_delay():
    a, b, theta, N, nu = -0.5, 0.4, 0.5, 20, 27
    mask = np.random.default_rng(3).uniform(0, 1, N)
    legacy = master_memory_function(a, b, theta, N, nu, mask=mask, convention="legacy")
    # v0.1 used reversed node order, which is the paper convention with the mask read backwards
    longer = master_memory_function(a, b, theta, N, nu + 1, mask=mask[::-1])
    k = min(len(legacy), len(longer))
    np.testing.assert_allclose(legacy[:k], longer[:k], atol=1e-8)


def test_capacity_bounds_and_scan():
    C = master_memory_function(-0.5, 0.4, 0.5, 20, 27)
    assert np.all(C > -1e-9) and np.all(C < 1 + 1e-9) and C.sum() <= 20 + 1e-6
    S = modified_state_matrix(-0.5, 0.4, 0.5, 20, 27)
    assert S.shape[0] == 20 and S[:, 0].any()
    totals = scan("tau_in_theta", [10, 27], a=-0.5, b=0.4, theta=0.5, N=20)
    assert totals.shape == (2,) and totals[1] == pytest.approx(C.sum())


def test_input_validation():
    with pytest.raises(ValueError):
        master_memory_function(0.1, 0.4, 0.5, 20, 27)
    with pytest.raises(ValueError):
        master_memory_function(-0.5, 0.4, 0.5, 20, 27, mask=np.ones(3))
    with pytest.raises(ValueError):
        master_memory_function(-0.5, 0.4, 0.5, 20, 27, convention="other")


def _binary():
    path = os.path.join(ROOT, "MMF")
    if not os.path.exists(path):
        if shutil.which("make") is None or subprocess.run(["make", "-C", ROOT], capture_output=True).returncode:
            pytest.skip("C++ build not available (Armadillo / Boost missing?)")
    return path


@pytest.mark.parametrize("convention", ["paper", "legacy"])
def test_cpp_matches_python(tmp_path, convention):
    binary = _binary()
    a, b, theta, N, nu = -1.0, -0.6, 0.3, 16, 16
    mask = np.random.default_rng(5).uniform(0, 1, N)
    np.savetxt(tmp_path / "mask.txt", mask)
    subprocess.run([binary, "-a", str(a), "-b", str(b), "-theta", str(theta), "-N", str(N), "-tauInTheta", str(nu),
                    "-maskFile", "mask.txt", "-delayConvention", convention, "-filename", "out.txt"],
                   cwd=tmp_path, check=True, capture_output=True)
    cpp = np.loadtxt(tmp_path / "out.txt", skiprows=1)[:, 1]
    py = master_memory_function(a, b, theta, N, nu, mask=mask, convention=convention)
    assert len(cpp) == len(py)
    np.testing.assert_allclose(cpp, py, atol=1e-8)


def test_cpp_legacy_reproduces_v01(tmp_path):
    binary = _binary()
    try:
        first = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.split()[0]
        src = subprocess.run(["git", "show", f"{first}:MMF.cpp"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        pytest.skip("needs git history")
    (tmp_path / "old.cpp").write_text(src)
    if subprocess.run(["g++", "-std=c++17", "-O2", "old.cpp", "-o", "old", "-larmadillo", "-llapack", "-lblas"],
                      cwd=tmp_path, capture_output=True).returncode:
        pytest.skip("could not build v0.1")
    args = ["-a", "-0.3", "-b", "0.2", "-theta", "0.6", "-N", "30", "-tauInTheta", "41"]
    subprocess.run(["./old", *args, "-filename", "old.txt"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([binary, *args, "-delayConvention", "legacy", "-filename", "new.txt"], cwd=tmp_path, check=True,
                   capture_output=True)
    old = np.loadtxt(tmp_path / "old.txt", skiprows=1)[:, 1]
    new = np.loadtxt(tmp_path / "new.txt", skiprows=1)[:, 1]
    k = min(len(old), len(new))
    np.testing.assert_allclose(new[:k], old[:k], atol=1e-4)  # v0.1 printed 4 decimals


def test_cpp_flag_order_and_errors(tmp_path):
    binary = _binary()
    outs = []
    for args in (["-N", "120", "-seed", "2"], ["-seed", "2", "-N", "120"]):
        r = subprocess.run([binary, *args, "-tauInTheta", "130", "-filename", "o.txt"], cwd=tmp_path,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs.append(np.loadtxt(tmp_path / "o.txt", skiprows=1)[:, 1])
    np.testing.assert_array_equal(outs[0], outs[1])  # v0.1 crashed for N > 100 unless -seed came last
    r = subprocess.run([binary, "-a", "0.2"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 1 and "a must be negative" in r.stderr
    r = subprocess.run([binary, "-N"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 1 and "missing value" in r.stderr
