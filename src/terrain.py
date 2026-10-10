"""Wilayah sintetis: lembah sungai, permukiman, dan tempat evakuasi.

Tahap "Masalah lokal -> Data" pada rantai integrator
(Masalah lokal -> Data -> Komputasi -> Model -> HPC -> Hasil -> Aplikasi).
Untuk penelitian nyata, ganti dengan DEM (data elevasi) dan data penduduk wilayah
yang diteliti. Fungsi ini hanya menyediakan data uji yang dapat direproduksi.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    elevation: np.ndarray      # (H, W) meter
    population: np.ndarray     # (H, W) jiwa per sel
    shelters: np.ndarray       # (S, 2) koordinat (baris, kolom)
    shelter_capacity: np.ndarray  # (S,) kapasitas jiwa

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape


def make_region(size: int = 120, n_shelters: int = 6, seed: int = 2026) -> Region:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]

    # Lembah sungai di tengah (kolom ~ size/2) yang meliuk, tebing naik ke tepi peta.
    river_x = size / 2 + 10 * np.sin(yy / size * 4 * np.pi)
    valley = 0.045 * np.abs(xx - river_x)
    slope = 0.012 * (size - yy)                    # hulu (baris 0) lebih tinggi
    noise = rng.normal(0, 0.35, (size, size))
    # haluskan derau agar medan tidak terlalu kasar
    for _ in range(3):
        noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
                 + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5
    elevation = 2.0 + valley + slope * 0.5 + noise * 2.0

    # Penduduk: padat di dataran rendah dekat sungai.
    dist_river = np.abs(xx - river_x)
    density = np.exp(-dist_river / 18.0) * rng.uniform(0.5, 1.5, (size, size))
    population = np.round(density * 12).astype(int)

    # Tempat evakuasi: di area lebih tinggi (30% elevasi teratas), tersebar.
    high = np.argwhere(elevation > np.quantile(elevation, 0.7))
    idx = rng.choice(len(high), n_shelters, replace=False)
    shelters = high[idx]
    capacity = rng.integers(1500, 4000, n_shelters)
    return Region(elevation, population, shelters, capacity)
