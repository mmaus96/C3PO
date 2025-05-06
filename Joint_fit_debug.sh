#!/bin/bash -l
#SBATCH -J Fit_debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -o output_logs/debug_fit%j.out
#SBATCH -e output_logs/debug_fit%j.err
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -A desi

date
#

module load python
conda activate C3PO_env


export PYTHONPATH=${PYTHONPATH}:./
export PYTHONPATH=${PYTHONPATH}:./likelihoods
export OMP_NUM_THREADS=8

echo "Setup done.  Starting to run code ..."

srun -n 16 -c 8 cobaya-run ./configs/yamls/<yaml file>



