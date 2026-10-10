# HPC Flood Evacuation: Monte Carlo paralel untuk masalah lokal

Studi kasus yang menerapkan materi **"Bridging Global Infrastructure to Address Local Challenge in High Performance Computing"** (Informatics Practitioner Lecture Series, Thammasat University). Pesan intinya: HPC bermakna ketika kemampuan komputasi besar dihubungkan dengan masalah nyata di masyarakat, dan mahasiswa informatika dapat berperan sebagai **integrator**.

Masalah lokal yang dipilih adalah **risiko banjir dan evakuasi**, salah satu contoh dari materi. Ribuan skenario hujan disimulasikan untuk memperkirakan peluang sebuah wilayah tergenang, jumlah penduduk terdampak, dan apakah tempat evakuasi cukup.

> **Batasan penting.** Wilayah dan datanya **sintetis**, dan model alirannya adalah automaton seluler yang sangat disederhanakan. Ini bukan model hidrologi tervalidasi dan tidak boleh dipakai untuk keputusan nyata. Tujuan repositori ini adalah mempraktikkan alur HPC (simulasi, paralelisasi, benchmark, reproducibility). Untuk penelitian, ganti `terrain.py` dengan data elevasi (DEM) dan penduduk wilayah nyata serta model yang tervalidasi.

## Konsep materi → implementasi

| Konsep dari materi | Di mana dalam kode |
|---|---|
| Rantai integrator: masalah lokal → data → komputasi → model → HPC → hasil → aplikasi | `terrain.py` (data) → `simulation.py` (model) → `parallel.py` (HPC) → `run_study.py` (hasil, peta risiko) |
| HPC untuk **simulasi** banyak skenario | Monte Carlo ribuan skenario hujan |
| Parallel computing | `multiprocessing.Pool` di `parallel.py` |
| **Performance evaluation** dan benchmarking | `benchmark.py`: speedup, efisiensi, Karp-Flatt |
| Akses supercomputer lewat Linux, SSH, **job script** | `jobs/run_study.sbatch` (template SLURM) |
| **Artifact description / evaluation** (dapat dijelaskan, dipakai ulang, direplikasi) | seed tetap, tes reproducibility, README ini |
| Efisiensi energi dan keberlanjutan | metrik `scenarios_per_s` dan efisiensi paralel sebagai proksi (energi tidak diukur langsung) |
| Kolaborasi lintas bidang | catatan di bawah: apa yang perlu divalidasi oleh ahli hidrologi dan kebencanaan |

## Struktur

```
src/terrain.py      wilayah sintetis: elevasi, penduduk, tempat evakuasi
src/simulation.py   satu skenario: banjir (automaton seluler) + estimasi evakuasi
src/parallel.py     eksekusi serial/paralel, seed per skenario, agregasi
src/benchmark.py    strong scaling: speedup, efisiensi, Karp-Flatt
run_study.py        program utama (CLI) + grafik
jobs/               template job script SLURM
tests/              tes reproducibility, perilaku fisik dasar, metrik
outputs/            hasil run (peta risiko, benchmark, ringkasan)
```

## Cara menjalankan

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
python run_study.py --scenarios 500 --workers 4
python run_study.py --scenarios 240 --benchmark      # sekaligus evaluasi performa
```

Pada cluster dengan SLURM, lihat `jobs/run_study.sbatch` (template; sesuaikan partisi dan modul dengan institusi Anda).

## Reproducibility

Setiap skenario memakai seed turunan dari `SeedSequence.spawn()`, sehingga **hasil identik untuk jumlah worker berapa pun**. Ini diuji di `tests/test_study.py` (1, 2, dan 3 worker menghasilkan angka yang sama) dan dicek ulang pada setiap benchmark (kolom `identical_to_serial`).

## Hasil (seed 7, 240 skenario, grid 120×120, mesin uji 2 core)

**Performa (strong scaling, beban tetap):**

| Worker | Waktu (detik) | Speedup | Efisiensi | Fraksi serial (Karp-Flatt) |
|---|---|---|---|---|
| 1 | 28,59 | 1,00 | 100% | n/a |
| 2 | 15,12 | 1,89 | 94,5% | 0,058 |

**Keluaran simulasi (hanya ilustrasi, data sintetis):** rata-rata 16.930 dari 49.596 jiwa terdampak per skenario (persentil-95: 30.443), dan pada 90% skenario setidaknya satu tempat evakuasi melebihi kapasitas. Peta peluang tergenang ada di `outputs/flood_probability_map.png`.

**Keterbatasan hasil benchmark:** mesin uji hanya memiliki 2 core, sehingga skalabilitas ke puluhan atau ratusan core belum terbukti. Pekerjaan ini juga "embarrassingly parallel" (skenario independen), jadi efisiensi tinggi adalah hal yang wajar; beban kerja yang lebih saling bergantung (misalnya simulasi satu domain besar dengan MPI) akan lebih sulit diskalakan. Jalankan `--benchmark` di mesin atau cluster Anda untuk angka yang relevan.

Catatan lain: waktu evakuasi persentil-90 hampir konstan (±79 menit) karena penduduk padat berada di dekat sungai dengan jarak tetap ke tempat evakuasi terdekat pada wilayah sintetis ini; pada wilayah nyata nilainya akan lebih bervariasi.

## Yang perlu divalidasi bersama ahli (prinsip kolaborasi)

- Model aliran air terhadap kejadian banjir historis dan model hidrologi/hidrolika yang diakui
- Distribusi hujan (di sini gamma sembarang) berdasarkan data curah hujan lokal
- Kecepatan dan jalur evakuasi nyata, bukan jarak garis lurus
- Kapasitas dan lokasi tempat evakuasi sesungguhnya

## Pengembangan lanjutan (ide skripsi)

- Ganti automaton dengan model banjir tervalidasi dan data DEM nyata
- Paralelisasi domain dengan MPI (`mpi4py`) dan bandingkan dengan multiprocessing
- Akselerasi GPU (CuPy/PyTorch) untuk langkah aliran air
- Weak scaling, pengukuran konsumsi energi, dan studi sensitivitas
- Antarmuka web sederhana (tahap "aplikasi" pada rantai integrator)

## Lisensi

MIT. Lihat `LICENSE`.
