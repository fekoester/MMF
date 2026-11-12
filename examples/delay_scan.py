"""
Total linear memory capacity vs delay, from the MMF, with direct simulations of the linear map as checks.

    python examples/delay_scan.py      # writes docs/delay_scan.png
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmf import master_memory_function, simulate_memory_capacity  # noqa: E402

a, b, theta, N = -0.5, 0.4, 0.5, 50
mask = np.random.default_rng(0).uniform(0, 1, N)
taus = np.arange(10, 161, 2)
mmf = [master_memory_function(a, b, theta, N, nu, mask=mask).sum() for nu in taus]
legacy = [master_memory_function(a, b, theta, N, nu, mask=mask, convention="legacy").sum() for nu in taus]
check = taus[::8]
sim = [simulate_memory_capacity(a, b, theta, N, nu, mask=mask, n_cycles=30_000, max_recall=200).sum() for nu in check]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(taus, mmf, label="MMF", lw=2)
ax.plot(taus, legacy, ls="--", lw=1, color="gray", label="v0.1 (models tau + theta)")
ax.plot(check, sim, "o", color="black", ms=5, label="simulation of the linear map")
for r in range(1, 4):
    ax.axvline(r * N, color="tab:red", ls=":", lw=1)
ax.set_xlabel(r"delay $\tau/\theta$")
ax.set_ylabel("total linear memory capacity")
ax.set_title(f"a={a}, b={b}, theta={theta}, N={N} (dotted: tau = multiples of N theta)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/delay_scan.png", dpi=130)
print("max |MMF - simulation|:", max(abs(m - s) for m, s in zip(np.array(mmf)[::8], sim)))
