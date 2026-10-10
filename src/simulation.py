"""Satu skenario banjir + estimasi evakuasi (unit pekerjaan yang diparalelkan).

CATATAN PENTING: model ini disederhanakan untuk tujuan belajar komputasi paralel.
Ia BUKAN model hidrologi tervalidasi dan tidak boleh dipakai untuk keputusan nyata.
Gantikan `flood_depth` dengan model yang tervalidasi bila dipakai untuk penelitian.
"""
from __future__ import annotations

import numpy as np

from .terrain import Region

STEPS = 250          # langkah waktu automaton
FLOW_K = 0.22        # koefisien aliran antar sel (stabil jika <= 0.25)


def flood_depth(elevation: np.ndarray, rain_mm: float, inflow: float, steps: int = STEPS) -> np.ndarray:
    """Automaton seluler sederhana: air mengalir dari kepala hidrolik tinggi ke rendah.

    rain_mm : hujan merata per sel (diubah ke meter air)
    inflow  : tambahan debit di hulu (baris pertama) tiap langkah
    Tepi peta berfungsi sebagai saluran keluar (air hilang).
    """
    h, w = elevation.shape
    water = np.full((h, w), rain_mm / 1000.0)
    for _ in range(steps):
        water[0, :] += inflow
        head = elevation + water
        padded = np.pad(head, 1, mode="constant", constant_values=-1e3)  # tepi = sink
        outflow = np.zeros_like(water)
        flows = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            f = FLOW_K * np.maximum(head - nb, 0.0)
            flows.append((dy, dx, f))
            outflow += f
        # jangan keluarkan air lebih dari yang ada
        scale = np.minimum(1.0, water / np.maximum(outflow, 1e-12))
        inflow_acc = np.zeros_like(water)
        for dy, dx, f in flows:
            moved = f * scale
            shifted = np.zeros_like(water)
            ys = slice(max(dy, 0), h + min(dy, 0))
            yd = slice(max(-dy, 0), h + min(-dy, 0))
            xs = slice(max(dx, 0), w + min(dx, 0))
            xd = slice(max(-dx, 0), w + min(-dx, 0))
            shifted[ys, xs] = moved[yd, xd]
            inflow_acc += shifted
        water = water - outflow * scale + inflow_acc
        # buang galat pembulatan floating-point (~1e-18 m) agar kedalaman tidak negatif
        np.maximum(water, 0.0, out=water)
    return water


def evacuation(region: Region, depth: np.ndarray, threshold: float = 0.3) -> dict:
    """Estimasi evakuasi sederhana untuk penduduk di sel yang tergenang > threshold (m)."""
    flooded = depth > threshold
    affected = int(region.population[flooded].sum())
    if affected == 0:
        return {"affected": 0, "evac_p90_min": 0.0, "overflow_shelters": 0}

    cells = np.argwhere(flooded)
    pop = region.population[flooded]
    d = np.linalg.norm(cells[:, None, :] - region.shelters[None, :, :], axis=2)  # (n, S)
    nearest = d.argmin(axis=1)
    # jarak ke tempat evakuasi (1 sel = 100 m), kecepatan jalan 4 km/jam
    minutes = d[np.arange(len(cells)), nearest] * 0.1 / 4.0 * 60.0
    load = np.bincount(nearest, weights=pop, minlength=len(region.shelters))
    overflow = int((load > region.shelter_capacity).sum())
    # persentil-90 tertimbang populasi
    order = np.argsort(minutes)
    cum = np.cumsum(pop[order]) / pop.sum()
    p90 = float(minutes[order][np.searchsorted(cum, 0.9)])
    return {"affected": affected, "evac_p90_min": p90, "overflow_shelters": overflow}


def run_scenario(args: tuple) -> dict:
    """Jalankan 1 skenario. `args` = (region, seed_seq). Hasil hanya bergantung pada seed."""
    region, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    rain = float(rng.gamma(shape=4.0, scale=45.0))      # mm, rerata ~180
    inflow = float(rng.gamma(shape=2.0, scale=0.0025))  # m/langkah di hulu
    depth = flood_depth(region.elevation, rain, inflow)
    out = evacuation(region, depth)
    out.update(rain_mm=rain, inflow=inflow, flooded_cells=int((depth > 0.3).sum()),
               _flooded_mask=(depth > 0.3))
    return out
