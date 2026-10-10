"""Eksekusi Monte Carlo serial vs paralel dengan hasil yang dapat direproduksi.

Kunci reproducibility: tiap skenario memiliki seed sendiri dari SeedSequence.spawn(),
sehingga hasil TIDAK bergantung pada jumlah worker atau urutan eksekusi.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from .simulation import run_scenario
from .terrain import Region


def make_tasks(region: Region, n_scenarios: int, seed: int = 7) -> list[tuple]:
    seeds = np.random.SeedSequence(seed).spawn(n_scenarios)
    return [(region, s) for s in seeds]


def run_serial(tasks: list[tuple]) -> list[dict]:
    return [run_scenario(t) for t in tasks]


def run_parallel(tasks: list[tuple], workers: int, chunksize: int = 1) -> list[dict]:
    if workers <= 1:
        return run_serial(tasks)
    with mp.Pool(processes=workers) as pool:
        return pool.map(run_scenario, tasks, chunksize=chunksize)  # urutan hasil terjaga


def timed(fn, *args, **kwargs) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def aggregate(results: list[dict]) -> dict:
    masks = np.stack([r["_flooded_mask"] for r in results])
    affected = np.array([r["affected"] for r in results], dtype=float)
    return {
        "n_scenarios": len(results),
        "flood_probability": masks.mean(axis=0),
        "mean_affected": float(affected.mean()),
        "p95_affected": float(np.percentile(affected, 95)),
        "prob_any_shelter_overflow": float(np.mean([r["overflow_shelters"] > 0 for r in results])),
        "mean_evac_p90_min": float(np.mean([r["evac_p90_min"] for r in results])),
        "affected": affected,
    }
