from __future__ import annotations

import numpy as np
from scipy.signal import lfilter
from scipy.stats import binom


def _mask(N: int, mask=None, seed: int = 0) -> np.ndarray:
    if mask is not None:
        w = np.asarray(mask, dtype=float).ravel()
        if len(w) != N:
            raise ValueError(f"mask has {len(w)} values, expected N = {N}")
        return w
    return np.random.default_rng(seed).uniform(0.0, 1.0, N)


def _check(a, theta, N, tau_in_theta, convention):
    if a >= 0:
        raise ValueError("a must be negative (p = exp(a theta) < 1)")
    if theta <= 0 or N <= 0 or tau_in_theta <= 0:
        raise ValueError("theta, N and tau_in_theta must be positive")
    if convention not in ("paper", "legacy"):
        raise ValueError("convention must be 'paper' or 'legacy'")


def modified_state_matrix(a, b, theta, N, tau_in_theta, mask=None, seed=0, convention="paper",
                          relative_threshold=1e-6, max_iterations=20000) -> np.ndarray:
    """
    S~ (N x L): S~[n, l] = sum over (i, k) with recall step l of C(i+k, i) p^i m^k w_idx, where
    p = exp(a theta), m = -(b/a)(1 - p), the input offset is i + k nu (nu = tau_in_theta), node n of a clock
    cycle sees the input ceil((off - n)/N) cycles back and mask index (n - off) mod N.
    convention="legacy" reproduces the v0.1 C++ program (offset i + k(nu + 1), reversed node order).
    """
    _check(a, theta, N, tau_in_theta, convention)
    w = _mask(N, mask, seed)
    p = np.exp(a * theta)
    ratio = -b / a
    n = np.arange(N)
    cols: dict[int, np.ndarray] = {}
    converged = False
    for total in range(max_iterations):
        k = np.arange(total + 1)
        t = ratio**k * binom.pmf(k, total, 1.0 - p)  # = C(total, k) p^(total-k) m^k
        if convention == "legacy":
            off = total + k * tau_in_theta
            l = (n[None, :] + off[:, None]) // N
            idx = (n[None, :] + off[:, None]) % N
        else:
            off = (total - k) + k * tau_in_theta
            l = -((n[None, :] - off[:, None]) // N)  # ceil((off - n)/N)
            idx = (n[None, :] - off[:, None]) % N
        contrib = t[:, None] * w[idx]
        for li in np.unique(l):
            if li < 0:
                continue
            sel = l == li
            col = cols.setdefault(int(li), np.zeros(N))
            np.add.at(col, np.nonzero(sel)[1], contrib[sel])
        if np.all(np.abs(t) < relative_threshold):
            converged = True
            break
    if not converged:
        import warnings

        warnings.warn("terms still above relative_threshold after max_iterations; result truncated", stacklevel=2)
    S = np.zeros((N, max(cols) + 1))
    for li, v in cols.items():
        S[:, li] = v
    return S


def _capacities(S, ridge, ridge_relative):
    G = S @ S.T
    lam = ridge * G.max() if ridge_relative else ridge
    Ginv = np.linalg.pinv(G + lam * np.eye(len(G)))
    return np.einsum("nl,nm,ml->l", S, Ginv, S)


def master_memory_function(a, b, theta, N, tau_in_theta, mask=None, seed=0, convention="paper", ridge=1e-6,
                           ridge_relative=True, relative_threshold=1e-6, max_iterations=20000) -> np.ndarray:
    """Linear memory capacity C_l for every recall step l (sum = total MC)."""
    S = modified_state_matrix(a, b, theta, N, tau_in_theta, mask, seed, convention, relative_threshold,
                              max_iterations)
    return _capacities(S, ridge, ridge_relative)


def simulate_memory_capacity(a, b, theta, N, tau_in_theta, mask=None, seed=0, n_cycles=50_000, washout=2_000,
                             max_recall=None, ridge=1e-6, ridge_relative=True, input_seed=1) -> np.ndarray:
    """
    Ground truth: run the map s_j = I_j + p s_{j-1} + m s_{j-nu} with I_j = u_{floor(j/N)} w_{j mod N}, u uniform in
    [-1, 1], and measure C_l = y_l^T S (S^T S + lambda I)^+ S^T y_l / |y_l|^2 from the N node states of each cycle.
    """
    _check(a, theta, N, tau_in_theta, "paper")
    w = _mask(N, mask, seed)
    p = np.exp(a * theta)
    m = -(b / a) * (1 - p)
    u = np.random.default_rng(input_seed).uniform(-1, 1, n_cycles)
    I = (u[:, None] * w[None, :]).ravel()
    den = np.zeros(tau_in_theta + 1)
    den[0] = 1.0
    den[1] -= p
    den[tau_in_theta] -= m
    s = lfilter([1.0], den, I)
    S = s.reshape(n_cycles, N)[washout:]
    S = S - S.mean(0)
    G = S.T @ S
    lam = ridge * G.max() if ridge_relative else ridge
    Ginv = np.linalg.pinv(G + lam * np.eye(N))
    L = max_recall or 3 * (tau_in_theta // N + 2) * 10
    out = np.empty(L)
    T = n_cycles
    for l in range(L):
        y = u[washout - l: T - l]
        y = y - y.mean()
        v = S.T @ y
        out[l] = v @ Ginv @ v / (y @ y)
    return out


def scan(param: str, values, **fixed) -> np.ndarray:
    """Total MMF capacity over a grid of one parameter, e.g. scan('tau_in_theta', range(50, 300), a=-0.5, ...)."""
    totals = []
    for v in values:
        kw = dict(fixed)
        kw[param] = v
        totals.append(master_memory_function(**kw).sum())
    return np.array(totals)
