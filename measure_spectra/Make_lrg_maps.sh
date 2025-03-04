#!/bin/bash -l
#SBATCH -J make_maps
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -o output_logs/debug_lrgmaps%j.out
#SBATCH -e output_logs/debug_lrgmaps%j.err
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -A desi

date
#

module load python
conda activate NaMaster_mpi

export OMP_NUM_THREADS=32

srun -n 32 -c 32 python make_y1footprint_mask.py