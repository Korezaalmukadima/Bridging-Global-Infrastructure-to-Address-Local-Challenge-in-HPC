"""Tes: reproducibility lintas jumlah worker, perilaku fisik dasar, metrik performa."""
import numpy as np
import pandas as pd
import pytest

from src import benchmark, parallel, simulation, terrain


@pytest.fixture(scope="module")
def region():
    return terrain.make_region(size=60, n_shelters=4, seed=1)


def test_results_identical_for_any_worker_count(region):
    tasks = parallel.make_tasks(region, 12, seed=3)
    a = [r["affected"] for r in parallel.run_parallel(tasks, 1)]
    b = [r["affected"] for r in parallel.run_parallel(tasks, 2)]
    c = [r["affected"] for r in parallel.run_parallel(tasks, 3, chunksize=2)]
    assert a == b == c


def test_same_seed_same_result(region):
    t = parallel.make_tasks(region, 1, seed=5)
    assert simulation.run_scenario(t[0])["affected"] == simulation.run_scenario(t[0])["affected"]


def test_water_is_nonnegative_and_finite(region):
    d = simulation.flood_depth(region.elevation, rain_mm=200, inflow=0.005, steps=60)
    assert np.isfinite(d).all() and (d >= 0).all()


def test_more_rain_floods_more(region):
    low = simulation.flood_depth(region.elevation, 50, 0.0, steps=80)
    high = simulation.flood_depth(region.elevation, 400, 0.0, steps=80)
    assert (high > 0.3).sum() >= (low > 0.3).sum()


def test_water_flows_downhill():
    elev = np.tile(np.linspace(10, 0, 20)[:, None], (1, 20))  # miring ke bawah
    d = simulation.flood_depth(elev, 100, 0.0, steps=100)
    assert d[:10].sum() < d[10:].sum()  # air berpindah ke bagian hilir (baris besar = lebih rendah)


def test_aggregate_probability_in_unit_interval(region):
    res = parallel.run_serial(parallel.make_tasks(region, 6, seed=2))
    agg = parallel.aggregate(res)
    p = agg["flood_probability"]
    assert p.min() >= 0 and p.max() <= 1
    assert 0 <= agg["prob_any_shelter_overflow"] <= 1


def test_strong_scaling_metrics(region):
    df = benchmark.strong_scaling(region, 8, [1, 2], repeats=1)
    assert isinstance(df, pd.DataFrame)
    assert df["identical_to_serial"].all()
    assert df.loc[df.workers == 1, "speedup"].iloc[0] == pytest.approx(1.0)
    assert (df["efficiency"] > 0).all()
