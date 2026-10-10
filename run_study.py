"""Studi kasus: risiko banjir dan evakuasi dengan simulasi Monte Carlo paralel.

Contoh:
    python run_study.py --scenarios 500 --workers 4          # simulasi + peta risiko
    python run_study.py --benchmark --scenarios 240          # + evaluasi performa
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import benchmark, parallel, terrain

OUT = Path(__file__).resolve().parent / "outputs"


def plot_risk_map(region: terrain.Region, agg: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(agg["flood_probability"], cmap="Blues", vmin=0, vmax=1)
    ax.contour(region.elevation, levels=8, colors="k", linewidths=0.25, alpha=0.4)
    ax.scatter(region.shelters[:, 1], region.shelters[:, 0], marker="^", s=90,
               c="#D62728", edgecolor="white", label="Tempat evakuasi")
    ax.set_title(f"Peluang tergenang (>0,3 m) dari {agg['n_scenarios']} skenario")
    ax.legend(loc="lower right")
    fig.colorbar(im, ax=ax, label="Peluang tergenang")
    fig.tight_layout()
    fig.savefig(OUT / "flood_probability_map.png", dpi=150)
    plt.close(fig)


def plot_affected(agg: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(agg["affected"], bins=30, color="#4C78A8")
    ax.axvline(agg["mean_affected"], color="k", ls="--", label=f"rata-rata {agg['mean_affected']:.0f}")
    ax.axvline(agg["p95_affected"], color="#D62728", ls="--", label=f"persentil-95 {agg['p95_affected']:.0f}")
    ax.set_xlabel("Jiwa terdampak per skenario")
    ax.set_ylabel("Jumlah skenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "affected_distribution.png", dpi=150)
    plt.close(fig)


def plot_benchmark(df) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["workers"], df["speedup"], "o-", label="terukur")
    axes[0].plot(df["workers"], df["workers"], "k--", label="ideal")
    axes[0].set_xlabel("Jumlah worker")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Strong scaling")
    axes[0].legend()
    axes[1].plot(df["workers"], df["efficiency"] * 100, "o-", color="#E45756")
    axes[1].set_ylim(0, 110)
    axes[1].set_xlabel("Jumlah worker")
    axes[1].set_ylabel("Efisiensi (%)")
    axes[1].set_title("Efisiensi paralel")
    for ax in axes:
        ax.set_xticks(df["workers"])
    fig.tight_layout()
    fig.savefig(OUT / "benchmark_scaling.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=500)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--grid", type=int, default=120, help="ukuran grid wilayah (sel)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--benchmark", action="store_true", help="jalankan juga strong-scaling benchmark")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    region = terrain.make_region(size=a.grid)
    tasks = parallel.make_tasks(region, a.scenarios, a.seed)
    results, dt = parallel.timed(parallel.run_parallel, tasks, a.workers)
    agg = parallel.aggregate(results)
    print(f"{a.scenarios} skenario, {a.workers} worker: {dt:.1f} detik "
          f"({a.scenarios / dt:.1f} skenario/detik)")

    summary = {k: v for k, v in agg.items() if k not in ("flood_probability", "affected")}
    summary.update(total_population=int(region.population.sum()), workers=a.workers,
                   wall_time_s=round(dt, 2), seed=a.seed, grid=a.grid)
    (OUT / "risk_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(OUT / "flood_probability.npy", agg["flood_probability"])
    plot_risk_map(region, agg)
    plot_affected(agg)
    print(json.dumps(summary, indent=2))

    if a.benchmark:
        counts = benchmark.default_worker_counts(a.workers)
        df = benchmark.strong_scaling(region, a.scenarios, counts, seed=a.seed)
        df.to_csv(OUT / "benchmark_strong_scaling.csv", index=False)
        plot_benchmark(df)
        print("\n== Strong scaling ==\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
