"""Performance evaluation: speedup, efisiensi, dan fraksi serial (Karp-Flatt).

  speedup    S(p) = T(1) / T(p)
  efisiensi  E(p) = S(p) / p
  Karp-Flatt e(p) = (1/S - 1/p) / (1 - 1/p)   -> perkiraan fraksi serial/overhead
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .parallel import make_tasks, run_parallel, run_serial, timed
from .terrain import Region


def default_worker_counts(max_workers: int | None = None) -> list[int]:
    cpus = max_workers or os.cpu_count() or 1
    counts, w = [], 1
    while w < cpus:
        counts.append(w)
        w *= 2
    counts.append(cpus)
    return sorted(set(counts))


def strong_scaling(region: Region, n_scenarios: int, worker_counts: list[int],
                   repeats: int = 2, seed: int = 7) -> pd.DataFrame:
    """Beban kerja tetap, jumlah worker bertambah (strong scaling)."""
    tasks = make_tasks(region, n_scenarios, seed)
    reference, _ = timed(run_serial, tasks[:2])  # pemanasan (import, cache)
    rows, base_affected = [], None
    for p in worker_counts:
        times = []
        for _ in range(repeats):
            res, dt = timed(run_parallel, tasks, p)
            times.append(dt)
        affected = np.array([r["affected"] for r in res])
        if base_affected is None:
            base_affected = affected
        rows.append({"workers": p, "time_s": min(times),
                     "identical_to_serial": bool(np.array_equal(affected, base_affected))})
    df = pd.DataFrame(rows)
    t1 = df.loc[df["workers"] == 1, "time_s"].iloc[0]
    df["speedup"] = t1 / df["time_s"]
    df["efficiency"] = df["speedup"] / df["workers"]
    p = df["workers"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kf = (1 / df["speedup"] - 1 / p) / (1 - 1 / p)
    df["karp_flatt_serial_fraction"] = np.where(p > 1, kf, np.nan)
    df["scenarios_per_s"] = n_scenarios / df["time_s"]
    return df.round(4)
