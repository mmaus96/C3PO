import argparse
import sys,os
import numpy as np
from cobaya.yaml          import yaml_load_file # type: ignore
import yaml

parser = argparse.ArgumentParser()

### What data and likelihoods to include in fits
parser.add_argument('--likelihoods', nargs='+',   required=True)
parser.add_argument('--chain_outpath', type=str,   required=True)
parser.add_argument('--tracers_3d',     nargs='+',   required=True) #Tracers to use for 3D clustering (FS+BAO)
parser.add_argument('--s_tracers',     nargs='+',   required=False,default='all') #Tracers to use for spec-z cross-correlations
parser.add_argument('--p_tracers',     nargs='+',   required=False,default='all')
parser.add_argument('--kappa_maps',  nargs='*',   required=False, default = None)
parser.add_argument('--SNe_sample', type=str,   required=False, default = 'union3')

### What cosmological model to use
parser.add_argument('--cosmo_model', type=str,   required=False, default = 'LCDM')
parser.add_argument('--slip',   action='store_true')
parser.add_argument('--ns_prior', type=float,   required=False, default = 1.0)

### Minimizer settings (if trying to find ML or MAP instead of running MCMC)
parser.add_argument('--optimize',    action='store_true')
parser.add_argument('--min_rhoend', type=float, required=False, default=0.01)
parser.add_argument('--min_seed', type=int, required=False,default=None)
parser.add_argument('--ignore_lin_priors',    action='store_true')

### Other fit options
parser.add_argument('--rescale_cov', type=float,   required=False, default = 1.0) #rescale covariance by 1/rescale_cov
parser.add_argument('--covmat', type=str,   required=False, default = None) #chain.covmat filepath from another fit to speed up convergence
parser.add_argument('--jeffrey',    action='store_true')  #Partial jeffrey's prior on linear params
parser.add_argument('--debug',    action='store_true') #Turns on 'debug' setting in cobaya-run for more verbose outputs
args = parser.parse_args()
args.include_lin_priors = not args.ignore_lin_priors

lcdm_emu_basedir = '/pscratch/sd/m/mmaus/DESI_RSD_kxg/emulator/LCDM/emu/' #Replace these with local once emulators have been created
w0wa_emu_basedir = '/pscratch/sd/m/mmaus/DESI_RSD_kxg/emulator/dHwp/emu/'
data_basedir = '/pscratch/sd/m/mmaus/DESI_RSD_kxg/data/'
w0wa_x0s = ['-1.00_0.00','-0.80_-0.80','-0.70_-1.20'] #Center pts of taylor series emulators


Cell_info = yaml_load_file('./configs/sample_info/Cell_lik.yaml')
Cell_specz_available = ['BGS','LRG1','LRG2','LRG3'] #tracers for which we can do cross correlations
Cell_photoz_available = ['pBGS1','pBGS2','pLRG1','pLRG2','pLRG3','pLRG4']


print('Making Config file...')
print('Using spectroscopic galaxies: ', args.tracers_3d)
print('with likelihood combination: ', args.likelihoods)
if args.kappa_maps:
    if ('Cell_specz' in args.likelihoods) or ('Cell_photoz' in args.likelihoods):
        print('Using CMB kappa maps from: ',args.kappa_maps)
    else:
        print(f'WARNING: You specified kappa maps {args.kappa_maps} but no Cell is specified in likelihoods')
if ('Cell_specz' in args.likelihoods):
    if args.s_tracers == 'all':
        args.s_tracers = Cell_specz_available
    else:
        args.s_tracers = [tr for tr in args.s_tracers if tr in Cell_specz_available]
    print(f'Using {args.s_tracers} for spectroscopic Cells')
if ('Cell_photoz' in args.likelihoods):
    if args.p_tracers == 'all':
        args.p_tracers = Cell_photoz_available
    else:
        args.p_tracers = [tr for tr in args.p_tracers if tr in Cell_photoz_available]
    print(f'Using {args.p_tracers} for photometric Cells')
        
info = {}
tracer_zfids = {'BGS':0.30,'LRG1': 0.51,'LRG2': 0.71,'LRG3': 0.92,'ELG': 1.32,'QSO': 1.49}
Rsmooth = {'BGS':15.,'LRG1': 15.,'LRG2': 15.,'LRG3': 15.,'ELG': 15.,'QSO': 30.}


##################### Make Theory Info #######################
def setup_theory(args,info):
    info['theory'] = {}
    
    if args.cosmo_model == 'LCDM':
        
        if ('RSD' in args.likelihoods) & ('BAO' in args.likelihoods):
            thy_nm = 'FSBAO_likelihood_gc_PkXiemu_am_tmp.Taylor_pk_theory_zs'
            thy_block = {}
            thy_block['zfids'] = [tracer_zfids[tr] for tr in args.tracers_3d]
            thy_block['bao_sample_names'] = args.tracers_3d
            thy_block['Rsmooth'] = [Rsmooth[tr] for tr in args.tracers_3d]
            thy_block['basedir'] = lcdm_emu_basedir
            thy_block['s8_filenames'] = [f'abacus_s8_ns_z{zfid:.2f}_desilike.json' for zfid in thy_block['zfids']] #My emulator naming convention is all over the place
            thy_block['xi_filenames'] = [f'desi_z_{zfid:.2f}_xiells.json'for zfid in thy_block['zfids']]
            thy_block['pk_filenames'] = [f'abacus_z_{zfid:.2f}_pkells_desilike.json'for zfid in thy_block['zfids']]
            thy_block['omega_nu'] = 0.0006442
            thy_block['stop_at_error'] = True 
            info['theory'][thy_nm] = thy_block
            
        elif ('RSD' in args.likelihoods) & ('BAO' not in args.likelihoods):
            thy_nm = 'pk_likelihood_emu_fs_am_tmp2.Taylor_pk_theory_zs'
            thy_block = {}
            thy_block['zfids'] = [tracer_zfids[tr] for tr in args.tracers_3d]
            thy_block['basedir'] = lcdm_emu_basedir
            thy_block['s8_filenames'] = [f'abacus_s8_ns_z{zfid:.2f}_desilike.json' for zfid in thy_block['zfids']]
            thy_block['pk_filenames'] = [f'abacus_z_{zfid:.2f}_pkells_desilike.json'for zfid in thy_block['zfids']]
            thy_block['omega_nu'] = 0.0006442
            thy_block['stop_at_error'] = True
            info['theory'][thy_nm] = thy_block 

        if 'Cell_specz' in args.likelihoods:
            if args.slip:
                thy_nm = 'CggCkg_specz_likelihood_am_slip.Taylor_Cells'
            else:
                thy_nm = 'CggCkg_specz_likelihood_am.Taylor_Cells'
            thy_block = {}
            thy_block['zeffs'] = [tracer_zfids[tr] for tr in args.s_tracers]
            thy_block['basedir'] = './data/spec_z_dat/'
            thy_block['gal_sample_names']= args.s_tracers
            thy_block['dndzfns'] = [Cell_info['specz'][tr]['dndz_fn'] for tr in args.s_tracers]
            thy_block['do_auto'] = False 
            thy_block['stop_at_error'] = True
            info['theory'][thy_nm] = thy_block 
        
        if 'Cell_photoz' in args.likelihoods:
            if args.slip:
                thy_nm = 'CggCkg_photoz_likelihood_am_slip.Taylor_Cells'
            else:
                thy_nm = 'CggCkg_photoz_likelihood_am.Taylor_Cells'
            thy_block = {}
            thy_block['basedir'] = './data/photo_z_dat/'
            thy_block['gal_sample_names'] = args.p_tracers
            thy_block['dndzfns'] = [Cell_info['photoz'][tr]['dndz_fn'] for tr in args.p_tracers]
            thy_block['stop_at_error'] = True
            info['theory'][thy_nm] = thy_block 

    if args.cosmo_model == 'w0wa':
        if 'SNe' in args.likelihoods:
            thy_nm = 'classy'
            thy_block = {}
            thy_block['extra_args'] = {'output':"",'N_ncdm': 1,'N_ur': 2.0328,'Omega_Lambda': 0}
            thy_block['stop_at_error'] = False
            thy_block['input_params'] = ['As','omegam','w','wa','100*theta_s','ns','omega_b']
            info['theory'][thy_nm] = thy_block
        if ('RSD' in args.likelihoods) & ('BAO' in args.likelihoods):
            thy_nm = 'FSBAO_likelihood_gc_PkXiemu_am_w0wa_multiemu.Taylor_pk_theory_zs'
            thy_block = {}
            thy_block['zfids'] = [tracer_zfids[tr] for tr in args.tracers_3d]
            thy_block['bao_sample_names'] = args.tracers_3d
            thy_block['Rsmooth'] = [Rsmooth[tr] for tr in args.tracers_3d]
            thy_block['basedir'] = w0wa_emu_basedir
            thy_block['s8_filenames'] = [[f'abacus_s8_w0wa_z_{zfid:.2f}_{x0}.json' for x0 in w0wa_x0s] for zfid in thy_block['zfids']] #Chain multiple Taylor Series emulators together
            thy_block['xi_filenames'] = [[f'desi_z{zfid:.2f}_xiells_w0wa_{x0}.json' for x0 in w0wa_x0s] for zfid in thy_block['zfids']]
            thy_block['pk_filenames'] = [[f'abacus_z_{zfid:.2f}_pkells_w0wa_{x0}.json' for x0 in w0wa_x0s] for zfid in thy_block['zfids']]
            thy_block['omega_nu'] = 0.0006442
            thy_block['stop_at_error'] = True 
            info['theory'][thy_nm] = thy_block

        if 'Cell_specz' in args.likelihoods:
            thy_nm = 'CggCkg_specz_likelihood_am_cleft_w0wa_multiemu.Taylor_Cells'
            thy_block = {}
            thy_block['zeffs'] = [tracer_zfids[tr] for tr in args.s_tracers]
            thy_block['basedir'] = './data/spec_z_dat/'
            thy_block['gal_sample_names']= args.s_tracers
            thy_block['dndzfns'] = [Cell_info['specz'][tr]['dndz_fn'] for tr in args.s_tracers]
            thy_block['do_auto'] = False 
            thy_block['Pmm_filenames'] = [w0wa_emu_basedir + f'Phalofit_w0wa_{x0}.json' for x0 in w0wa_x0s]
            thy_block['Pgm_filenames'] = [[w0wa_emu_basedir + f'abacus_z_{zfid:.2f}_PggPgm_real_{x0}.json' for x0 in w0wa_x0s] for zfid in thy_block['zfids']]

    return info

########################### Make Likelihood Info ##########################
def setup_likelihood(args,info):
    info['likelihood'] = {}
    
    if args.cosmo_model == 'LCDM':
        if 'Cell_specz' in args.likelihoods:
            lik_nm = 'Ckg_pr4_dr6_desi_specz'
            lik_block = {}
            if args.slip:
                lik_block['class'] = 'CggCkg_specz_likelihood_am_slip.CellLikelihood'
            else:
                lik_block['class'] = 'CggCkg_specz_likelihood_am.CellLikelihood'
            lik_block['basedir'] = Cell_info['specz']['basedir']
            lik_block['linear_param_dict_fn'] = Cell_info['specz']['linear_param_dict_fn']
            lik_block['zfids'] = [Cell_info['specz'][tr]['zfid'] for tr in args.s_tracers]
            lik_block['gal_sample_names']= args.s_tracers
            lik_block['kappa_sample_names'] = args.kappa_maps
            lik_block['dndzfns'] = [Cell_info['specz'][tr]['dndz_fn'] for tr in args.s_tracers]
            lik_block['datfns'] = [[Cell_info['specz'][tr]['data_fn'][knm] for knm in args.kappa_maps] for tr in args.s_tracers]
            lik_block['covfn'] = Cell_info['specz']['covfn']
            lik_block['amin'] = [Cell_info['specz'][tr]['amin'] for tr in args.s_tracers]
            lik_block['amax'] = [Cell_info['specz'][tr]['amax'] for tr in args.s_tracers]
            lik_block['xmin'] = [[Cell_info['specz'][tr]['xmin'][knm] for tr in args.s_tracers] for knm in args.kappa_maps]
            lik_block['xmax'] = [[Cell_info['specz'][tr]['xmax'][knm] for tr in args.s_tracers] for knm in args.kappa_maps]
            lik_block['include_priors'] = args.include_lin_priors

            lik_block['optimize'] = args.optimize
            lik_block['stop_at_error'] = True
            lik_block['jeff'] = args.jeffrey
            lik_block['cov_fac'] = args.rescale_cov
            info['likelihood'][lik_nm] = lik_block       
        if 'Cell_photoz' in args.likelihoods:
            lik_nm = 'CggCkg_pr4_dr6_desi_photoz'
            lik_block = {}
            if args.slip:
                lik_block['class'] = 'CggCkg_photoz_likelihood_am_slip.CellLikelihood'
            else:
                lik_block['class'] = 'CggCkg_photoz_likelihood_am.CellLikelihood'
            lik_block['basedir'] = Cell_info['photoz']['basedir']
            lik_block['linear_param_dict_fn'] = Cell_info['photoz']['linear_param_dict_fn']
            lik_block['chenprior'] = True
            lik_block['gal_sample_names']= args.p_tracers
            lik_block['alt_nms'] = [Cell_info['photoz'][tr]['alt_nm'] for tr in args.p_tracers] #if data json files have different naming convention
            lik_block['kappa_sample_names'] = args.kappa_maps
            lik_block['dndzfns'] = [Cell_info['photoz'][tr]['dndz_fn'] for tr in args.p_tracers]
            lik_block['gal_datfns'] = None #Currently using pixel based spectra in photo-z bins
            lik_block['cl_pix'] = True
            lik_block['covfn'] = Cell_info['photoz']['covfn']
            lik_block['amin'] = [Cell_info['photoz'][tr]['amin'] for tr in args.p_tracers]
            lik_block['amax'] = [Cell_info['photoz'][tr]['amax'] for tr in args.p_tracers]
            lik_block['xmin'] = [[Cell_info['photoz'][tr]['xmin'][knm] for tr in args.p_tracers] for knm in args.kappa_maps]
            lik_block['xmax'] = [[Cell_info['photoz'][tr]['xmax'][knm] for tr in args.p_tracers] for knm in args.kappa_maps]
            lik_block['fidSN'] = [Cell_info['photoz'][tr]['fid_SN'] for tr in args.p_tracers]
            lik_block['include_priors'] = args.include_lin_priors
            lik_block['optimize'] = args.optimize
            lik_block['stop_at_error'] = True
            lik_block['jeff'] = args.jeffrey
            lik_block['cov_fac'] = args.rescale_cov
            info['likelihood'][lik_nm] = lik_block
        if ('RSD' in args.likelihoods) & ('BAO' in args.likelihoods):
            for tr in args.tracers_3d:
                lik_nm = f'DESI_RSDBAO_{tr}'
                lik_block = yaml_load_file(f'./configs/sample_info/DESI_RSDBAO_{tr}_lik.yaml')
                # overwrite settings:
                lik_block['basedir'] = data_basedir + 'spec_z_dat/'
                lik_block['include_priors'] = args.include_lin_priors
                lik_block['optimize'] = args.optimize
                lik_block['jeff'] = args.jeffrey
                lik_block['cov_fac'] = args.rescale_cov
                info['likelihood'][lik_nm] = lik_block
        elif ('RSD' in args.likelihoods) & ('BAO' not in args.likelihoods):
            for tr in args.tracers_3d:
                lik_nm = f'DESI_RSD_{tr}'
                lik_block = yaml_load_file(f'./configs/sample_info/DESI_RSD_{tr}_lik.yaml')
                # overwrite settings:
                lik_block['basedir'] = data_basedir + 'spec_z_dat/'
                lik_block['include_priors'] = args.include_lin_priors
                lik_block['optimize'] = args.optimize
                lik_block['jeff'] = args.jeffrey
                lik_block['cov_fac'] = args.rescale_cov
                info['likelihood'][lik_nm] = lik_block

    if args.cosmo_model == 'w0wa':
        if 'SNe' in args.likelihoods:
            lik_nm = f'sn.{args.SNe_sample}'
            info['likelihood'][lik_nm] = None #None if using default, otherwise specify settings to overwrite
        if 'Cell_specz' in args.likelihoods:
            lik_nm = 'Ckg_pr4_dr6_desi_specz'
            lik_block = {}
            lik_block['class'] = 'CggCkg_specz_likelihood_am_cleft_w0wa_multiemu.CellLikelihood'
            lik_block['basedir'] = Cell_info['specz']['basedir']
            lik_block['linear_param_dict_fn'] = Cell_info['specz']['linear_param_dict_fn']
            lik_block['zfids'] = [Cell_info['specz'][tr]['zfid'] for tr in args.s_tracers]
            lik_block['gal_sample_names']= args.s_tracers
            lik_block['kappa_sample_names'] = args.kappa_maps
            lik_block['dndzfns'] = [Cell_info['specz'][tr]['dndz_fn'] for tr in args.s_tracers]
            lik_block['datfns'] = [[Cell_info['specz'][tr]['data_fn'][knm] for knm in args.kappa_maps] for tr in args.s_tracers]
            lik_block['covfn'] = Cell_info['specz']['covfn']
            lik_block['amin'] = [Cell_info['specz'][tr]['amin'] for tr in args.s_tracers]
            lik_block['amax'] = [Cell_info['specz'][tr]['amax'] for tr in args.s_tracers]
            lik_block['xmin'] = [[Cell_info['specz'][tr]['xmin'][knm] for tr in args.s_tracers] for knm in args.kappa_maps]
            lik_block['xmax'] = [[Cell_info['specz'][tr]['xmax'][knm] for tr in args.s_tracers] for knm in args.kappa_maps]
            lik_block['include_priors'] = args.include_lin_priors
            lik_block['optimize'] = args.optimize
            lik_block['stop_at_error'] = True
            lik_block['jeff'] = args.jeffrey
            lik_block['cov_fac'] = args.rescale_cov
            info['likelihood'][lik_nm] = lik_block
        if ('RSD' in args.likelihoods) & ('BAO' in args.likelihoods):
            for tr in args.tracers_3d:
                lik_nm = f'DESI_RSDBAO_{tr}'
                lik_block = yaml_load_file(f'./configs/sample_info/DESI_RSDBAO_{tr}_lik.yaml')
                # overwrite settings:
                lik_block['class'] = 'FSBAO_likelihood_gc_PkXiemu_am_w0wa_multiemu.JointLikelihood'
                lik_block['basedir'] = data_basedir + 'spec_z_dat/'
                lik_block['include_priors'] = args.include_lin_priors
                lik_block['optimize'] = args.optimize
                lik_block['jeff'] = args.jeffrey
                lik_block['cov_fac'] = args.rescale_cov
                info['likelihood'][lik_nm] = lik_block
    return info

########################## Add Params Block #################################
gamma = {'prior': {'dist': 'uniform', 'min': 0.2, 'max': 1.8}, 'ref': {'dist': 'norm', 'loc': 1.0, 'scale': 0.05},'latex': '\gamma'}
def setup_params(args,info):
    info['params'] = {}
    blocking = []
    
    if args.cosmo_model == 'LCDM':
        cosmo_pars = yaml_load_file('./configs/params/params_cosmo_LCDM.yaml')
        cosmo_pars['ns']['prior']['scale'] = cosmo_pars['ns']['prior']['scale']*args.ns_prior
        info['params'] = cosmo_pars
        cosmo_pars_sampled = []
        for p in cosmo_pars.keys():
            if 'prior' in cosmo_pars[p]:
                cosmo_pars_sampled.append(p)
        if args.slip:
            cosmo_pars['gamma'] = gamma
            cosmo_pars_sampled.append('gamma')
        blocking.append([1,cosmo_pars_sampled])
    if args.cosmo_model == 'w0wa':
        cosmo_pars = yaml_load_file('./configs/params/params_cosmo_w0wa.yaml')
        info['params'] = cosmo_pars
        cosmo_pars_sampled = []
        for p in cosmo_pars.keys():
            if 'prior' in cosmo_pars[p]:
                cosmo_pars_sampled.append(p)
        blocking.append([1,cosmo_pars_sampled])

    for tr in args.tracers_3d:
        nuisance_params = yaml_load_file(f'./configs/params/params_nuisance_{tr}_specz.yaml')
        if 'RSD' in args.likelihoods:
            rsd_pnames = [f'bsig8_{tr}',f'b2sig8_{tr}',f'bssig8_{tr}',f'b3sig8_{tr}']
            rsd_pnames_sampled = []
            for p in rsd_pnames:
                info['params'][p] = nuisance_params[p]
                if 'prior' in nuisance_params[p]:
                    rsd_pnames_sampled.append(p)
            if ('Cell_specz' not in args.likelihoods) or (tr not in args.s_tracers): 
                blocking.append([4,rsd_pnames_sampled])
        if 'BAO' in args.likelihoods:
            bao_pnames = [f'B1_{tr}',f'F_{tr}',f'Sigpar_{tr}',f'Sigperp_{tr}',f'Sigs_{tr}']
            bao_pnames_sampled = []
            for p in bao_pnames:
                info['params'][p] = nuisance_params[p]
                if 'prior' in nuisance_params[p]:
                    bao_pnames_sampled.append(p)
            blocking.append([4,bao_pnames_sampled])
        if ('Cell_specz' in args.likelihoods) & (tr in args.s_tracers):
            info['params'][f'smag_{tr}'] = nuisance_params[f'smag_{tr}']
            blocking.append([2,rsd_pnames_sampled+[f'smag_{tr}']])


    if 'Cell_photoz' in args.likelihoods:
        for tr in args.p_tracers:
            nuisance_params = yaml_load_file(f'./configs/params/params_nuisance_{tr}.yaml')
            for p in list(nuisance_params.keys()):
                info['params'][p] = nuisance_params[p]
            blocking.append([2,nuisance_params])

    return info,blocking

######################### Add Sampler Block ##################################

def setup_mcmc(args,info,blocking):
    info['sampler'] = {'mcmc': {}}    
    samp_block = {}
    if args.covmat != None:
        samp_block['covmat'] = args.covmat
    samp_block['learn_proposal'] = True
    samp_block['learn_proposal_Rminus1_max'] = 30.0
    samp_block['output_every'] = '60s'
    samp_block['measure_speeds'] = False
    samp_block['Rminus1_stop'] = 0.01
    samp_block['blocking'] = blocking
    info['sampler']['mcmc'] = samp_block

    return info


def setup_minimizer(args,info):
    info['sampler'] = {'minimize': {}}
    info['sampler']['minimize']['seed'] = args.min_seed
    info['sampler']['minimize']['override_bobyqa'] = {'rhoend': args.min_rhoend}
    if os.path.exists(args.chain_outpath + 'chain.covmat'):
        info['sampler']['minimize']['covmat'] = args.chain_outpath + 'chain.covmat'
    if not args.include_lin_priors:
        info['sampler']['minimize']['ignore_prior'] = True
    return info

######################### Create Yaml filename ####################################

def create_filename(args):
    yaml_nm = f'fit_{args.cosmo_model}'
    for lik in args.likelihoods:
        # if (lik == 'RSD') or (lik == 'BAO'):
        if lik == 'RSD':
            if 'BAO' not in args.likelihoods:
                yaml_nm += '_' + lik  
            else:
                yaml_nm += '_RSDBAO'
            yaml_nm += '_' + ''.join(args.tracers_3d)
        if lik == 'BAO':
            if 'RSD' not in args.likelihoods:
                yaml_nm += '_' + lik 
                yaml_nm += '_' + ''.join(args.tracers_3d)
        elif lik == 'Cell_specz':
            yaml_nm += '_' + lik 
            yaml_nm += '_' + ''.join(args.s_tracers)
        elif lik == 'Cell_photoz':
            yaml_nm += '_' + lik 
            yaml_nm += '_' + ''.join(args.p_tracers) 
        elif lik == 'SNe':
            yaml_nm += '_' + lik 
            yaml_nm += '_' + args.SNe_sample 
    if args.kappa_maps:
        yaml_nm += '_' + ''.join(args.kappa_maps)
    if args.ns_prior != 1.0:
        yaml_nm += '_' + f'ns{args.ns_prior}'
    if args.rescale_cov != 1.0:
        yaml_nm += '_' + f'Covresc{args.rescale_cov}'
    if args.jeffrey:
        yaml_nm += '_' + 'jeffrey'
    if args.optimize:
        if args.include_lin_priors:
            yaml_nm += '_' + f'optimize_MAP_Rhoend{args.min_rhoend}'
        else:
            yaml_nm += '_' + f'optimize_ML_Rhoend{args.min_rhoend}'
    yaml_nm += '.yaml'
    return yaml_nm

    
    

info = setup_theory(args,info)
info = setup_likelihood(args,info)
info,blocking = setup_params(args,info)
if args.optimize:
    info = setup_minimizer(args,info)
else:
    info = setup_mcmc(args,info,blocking)

if os.path.exists(args.chain_outpath):
    if not os.listdir(args.chain_outpath):
        print("Found empty destination directory: ",args.chain_outpath)
    else:
        print("WARNING: Destination directory is NOT empty!!!")
else:
    os.mkdir(args.chain_outpath)

info['output'] = args.chain_outpath + 'chain'
if args.debug:
    info['debug'] = True
    info['debug_file'] = args.chain_outpath + 'chain'
info['timing'] = True
info['stop_on_error'] = True

yaml_nm = create_filename(args)
print(f'Creating Yaml config file: ./configs/yamls/{yaml_nm}')
print(f'To run fit use:')
print(f'srun -n 16 -c 8 cobaya-run ./configs/yamls/{yaml_nm}')
print(f'Resume existing chain with -r at end')
print(f'Force restart (USE CAUTIOUSLY) with -f at end')

with open(f'./configs/yamls/{yaml_nm}', 'w') as outfile:
    yaml.dump(info, outfile, default_flow_style=False, sort_keys=False)