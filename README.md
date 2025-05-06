# C3PO
A measurement and analysis pipeline for 3D Clustering and Cross-Correlations between DESI galaxies and CMB lensing

## Dependencies
There are two conda environments necessary for running scripts in  this pipeline. For nearly everything (measuring angular spectra, computing their covariances, training Taylor series emulators, and running chains) the environment created with `create_cobaya_env.sh` is sufficient. However, for measuring the 3D power spectra and window matrices we use the cosmodesi environment that can be sourced with:

`source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main`

## Measuring Spectra:
### Measuring 3D 
### Measuring 2D angular spectra

## Running chains
In order to run MCMC fits with Cobaya we first create config files using the `setup_fits.py` script. By running `python setup_fits.py --[settings]` a yaml file will be created in `./configs/yamls/`. The yaml file contains everything Cobaya needs for running the chain with desired settings specified by parameters in `--[settings]`. The filename (and relative path) can simply be copied into the `Joint_fit_debug.sh` or `Joint_fit_reg.sh` job submission scripts in the line: 

`srun -n 16 -c 8 cobaya-run ./configs/yamls/[yaml file]`. 

For example, running:

`python setup_fits.py --likelihoods RSD BAO Cell_specz --tracers_3d BGS LRG1 LRG2 LRG3 --s_tracers BGS LRG1 LRG2 LRG3 --kappa_maps PR4 DR6  --chain_outpath ./chains/test/`

will setup a $P_{\ell}(k) + \xi_{\ell}^{\rm post}(s) + C_{\ell}^{\kappa g}$ fit using the BGS and three LRG spectroscopic galaxy samples and Planck PR4 + ACT DR6 CMB lensing maps by creating the file: `fit_LCDM_RSDBAO_BGSLRG1LRG2LRG3_Cell_specz_BGSLRG1LRG2LRG3_PR4DR6.yaml` and then you just begin the fit with 

`srun -n 16 -c 8 cobaya-run ./configs/yamls/fit_LCDM_RSDBAO_BGSLRG1LRG2LRG3_Cell_specz_BGSLRG1LRG2LRG3_PR4DR6.yaml`

This runs an MCMC with 16 parallel chains using 8 threads per chain. One can modify -n and -c as desired. The following options can be passed to `setup_fits.py`:

* `--cosmo_model`    specifies the cosmological model. The current options are just 'LCDM'(default) and 'w0wa'.
* `--likelihoods`    specifies the likelihoods/data combos for fitting. Options are 'RSD', 'BAO', 'Cell_specz', 'Cell_photoz', 'SNe'. At the moment the 'SNe' option (for including supernova) is only compatible with w0wa fits. 
* `--tracers_3d`     specifies the spectroscopic galaxy samples to use for FS (+BAO) fits. The naming convention is 'BGS', 'LRG1', 'LRG2', 'LRG3','ELG', and 'QSO'.
* `--s_tracers`      specifies the spectroscopic galaxy samples to use for fitting Ckg. The options are a subset of tracers_3d but currently Ckg is not available for ELG and QSO tracers. If this parameter is not specified then it defaults all available samples (ie BGS, LRG1, LRG2, LRG3)
* `--p_tracers`      specifies the photometric galaxy samples to use for fitting Cgg+Ckg. Options are 'pBGS1','pBGS2','pLRG1','pLRG2','pLRG3','pLRG4'
* `--chain_outpath`  path to directory where MCMC chains will be stored. Will create directory if it doesn't already exist. Will give a warning if specified directory is not empty
* `--kappa_maps`     specifies which CMB lensing data sets to use for cross-correlations. Options are 'PR4' and 'DR6'
* `--SNe_sample`     specifies which SNe data set to combine with if 'SNe' is in likelihoods and cosmo_model='w0wa'. Default is 'union3' but any sample available in Cobaya can be used. 
* `--ns_prior`       prior width on n_s (as a factor of Planck 2018). Default is 1.0
* `--rescale_cov`    factor by which to rescale covariance (C' = C/rescale_cov) to test things like projection effects by artificially constraining power of data (changing effective survey volume)
* `--covmat`         specifies path to parameter covmat from a different fit in order to speed up convergence
* `--debug`          turns on debug: True for cobaya-run for more verbose outputs during fitting
* `--optimize`       sets up minimizer (bobyqa) instead of MCMC
* `--ignore_lin_priors` turns off priors on linear nuisance parameters (equivalent to setting infinite flat priors). When running minimizer (with --optimize) this will also set ignore_priors=True in the Cobaya minimizer setting to obtain the true best-fit (ML) point. The saved yaml file will have "optimize_ML" in the name. If using --optimize without --ignore_lin_priors then this will give the Maximum A Posteriori (MAP) so the yaml file will have "optimize_MAP" in the name.
* `--min_rhoend`     parameter specifying convergence criterion for bobyqa minimizer. Cobaya's default is 0.05 but the default here is 0.01. The smaller the value the longer it will take. 

