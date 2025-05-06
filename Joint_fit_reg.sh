#!/bin/bash -l
#SBATCH -J Fit_reg
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -o output_logs/job%j.out
#SBATCH -e output_logs/job%j.err
#SBATCH -q regular
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
# srun -n 16 -c 8 cobaya-run ./configs/yamls/fit_LCDM_RSDBAO_BGSLRG1LRG2LRG3_Cell_specz_BGSLRG1LRG2LRG3_PR4DR6.yaml -r