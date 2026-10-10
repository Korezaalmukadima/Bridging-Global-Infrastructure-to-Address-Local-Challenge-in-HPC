#!/bin/bash
# TEMPLATE job script SLURM. Belum diuji pada cluster tertentu: sesuaikan nama partisi,
# modul, dan batas sumber daya dengan dokumentasi HPC institusi Anda.
#
# Kirim job :  sbatch jobs/run_study.sbatch
# Pantau    :  squeue -u $USER
# Hasil     :  folder outputs/ dan file slurm-<jobid>.out

#SBATCH --job-name=flood-mc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # jumlah core untuk multiprocessing
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out

# module load python/3.11         # sesuaikan dengan modul yang tersedia di cluster
python -m venv .venv && source .venv/bin/activate
pip install -q -r requirements.txt

# gunakan jumlah core yang dialokasikan scheduler, bukan seluruh core node
python run_study.py --scenarios 2000 --workers "$SLURM_CPUS_PER_TASK" --benchmark
