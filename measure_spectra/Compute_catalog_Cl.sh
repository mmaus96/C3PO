#!/bin/bash -l
#SBATCH -J comp_Cl
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -o output_logs/debug_Cl%j.out
#SBATCH -e output_logs/debug_Cl%j.err
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -A desi

date
#

module load python
conda activate NaMaster

export OMP_NUM_THREADS=128

srun -n 1 python -u compute_catalog_based_Cl.py 3