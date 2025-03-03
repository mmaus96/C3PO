#!/bin/bash -l
#SBATCH -J Clcov
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -o output_logs/Clcov%j.out
#SBATCH -e output_logs/Clcov%j.err
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A desi

date
#

module load python
# conda activate NaMaster
conda activate NaMaster_old

export OMP_NUM_THREADS=128

# srun -n 1 python -u make_specz_lrg_map.py 3
# srun -n 1 python -u calc_cl_cov_pixel_photoz.py 128
# srun -n 1 python -u calc_cl_cov_pixel_photoz.py 1024
# srun -n 1 python -u calc_cl_cov_pixel_photoz.py -1
# srun -n 1 python -u calc_cl_cov_pixel.py

srun -n 1 python -u lrg_cross_pr4+dr6_lmax-1900.py