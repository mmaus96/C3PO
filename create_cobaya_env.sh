#!/bin/bash
#
# Need to run this from an interactive shell, not a Jupyter "shell"
# since those don't have the PrgEnv installed. (Not sure if this is still true in perlmutter)
#
# I needed to use PrgEnv-gnu in order to get the Planck lensing
# likelihood to compile.
# 
module load python
conda create --name C3PO_env
conda activate C3PO_env

# Install some basic stuff
conda install -c conda-forge namaster
# Use the "Cobaya" version of CLASS (installed with cobaya-install below)
# Set up the environment for Jupyter.
conda install ipykernel ipython jupyter # Should already be there.
python -m ipykernel install --user --name C3PO_env --display-name C3PO-env
#
python -m pip install cobaya  --upgrade
#
cobaya-install cosmo -p /global/homes/m/mmaus/Cobaya_new/C3PO/packages # replace with desired local path
#
# Install velocileptors. 
 python3 -m pip install -v git+https://github.com/sfschen/velocileptors
# and Aemulusv
python3 -m pip install -v git+https://github.com/AemulusProject/aemulus_heft
# mpi
python -m pip install mpi4py
# finding
pip install --upgrade findiff
# pyfftw
pip install pyFFTW
