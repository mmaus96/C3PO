import sys
import numpy as np
import json

from copy import deepcopy
from pathlib import Path
# from astropy.table import Table, hstack,vstack,join
# import fitsio

import pymaster as nmt
import healpy as hp

from calc_cl import*

def get_bins(ledges):#,nside):
    """
    Takes an input set of ledges and sets up a NaMaster bin object.
    
    ledges : list of ell-bin edges
    nside  : healpix nside
    """
    # set up ell-bins
    Nbin = len(ledges)-1
    ells = np.arange(ledges[-1]+1,dtype='int32')
    bpws = np.zeros_like(ells) - 1
    for i in range(Nbin): bpws[ledges[i]:ledges[i+1]] = i
    # bins = nmt.NmtBin(nside,bpws=bpws,ells=ells,weights=np.ones_like(ells))
    bins = nmt.NmtBin(bpws=bpws,ells=ells,weights=np.ones_like(ells))
    return bins


nlb = None #50
nside = 2048
NSIDE = nside
lmax = 3*nside -1

if nlb != None:
    bins = nmt.NmtBin.from_lmax_linear(lmax, nlb=nlb)
else:
    bins = None
weight_type = 'FKP'
cat_root = 'v1.5'


ledges = [10,20,44,79,124,178,243,317,401,495,600,713,837]
ledges = ledges + (np.linspace(971**0.5,(3*nside)**0.5,20,endpoint=True)**2).astype(int).tolist()

# bins = get_bins(ledges)

# fnout = 'cls_cov_specz_pixel_kg1g2g3_default_ledges.json'
fnout = './data/spec_z_dat/PR4_DR6_DESI_cov/cls_cov_specz_pixel_PR4_DR6_LRG_{}_nlb{}_{}.json'.format(weight_type,nlb,cat_root)
#load maps
#Load cmb map:
print('loading CMB lensing maps')
# kappa = hp.read_map('./data/planck_pr4/PR4_lens_kap_filt.hpx2048.fits')
# lens_mask = hp.read_map('./data/planck_pr4/PR4_lens_mask.fits')
# get ACT alms and filter them
klm    = np.nan_to_num(hp.read_alm('./data/ACT_DR6/kappa_alm_data_act_dr6_lensing_v1_baseline.fits'))
filt   = np.ones(3*NSIDE) ; filt[3000:] = 0. 
klm    = hp.almxfl(klm,filt)
pr4_map   = [hp.read_map(f'./data/planck_pr4/PR4_lens_kap_filt.hpx2048.fits')]
pr4_mask  = [hp.read_map(f'./data/planck_pr4/PR4_lens_mask.fits')]
dr6_map   = [hp.alm2map(klm,NSIDE)]
dr6_mask  = [hp.read_map(f'./data/ACT_DR6/DR6_lens_mask.fits')]

pr4_nkk   = np.loadtxt(f'./data/planck_pr4/PR4_lens_nlkk_filt.txt')
dr6_nkk   = np.loadtxt(f'./data/ACT_DR6/DR6_lens_nlkk.txt')
pr4dr6_nkk= np.loadtxt('./data/PR4xDR6_nkk.txt')

# f_lens = nmt.NmtField(lens_mask, [kappa])
# maps = [kappa]
# masks = [lens_mask]
print('done')

#load LRG maps:
print('loading LRG maps')
lrg_maps = [hp.read_map(f'./data/spec_z_dat/maps/specz_pixel_desi_LRG_z{i+1}_{weight_type}_{cat_root}.hpx2048_map.fits') for i in range(3)]
lrg_masks = [hp.read_map(f'./data/spec_z_dat/maps/specz_pixel_desi_LRG_z{i+1}_{weight_type}_{cat_root}.hpx2048_msk.fits') for i in range(3)]
print('done')

# give our maps & masks some names
kapNames = ['PR4','DR6']
galNames = ['LRG1','LRG2','LRG3']
names    = kapNames + galNames
masks     = pr4_mask + dr6_mask + lrg_masks
maps     = pr4_map  + dr6_map  + lrg_maps

# curves for covariance estimation
cij  = np.zeros((5,5,3*2048))
cij[1:,1:,:] = np.loadtxt(f'./data/spec_z_dat/fiducial/cls_LRGxPR4_bestFit.txt').reshape((4,4,3*2048))
cij[0,1:,:] = cij[1,1:,:]
cij[1:,0,:] = cij[1,1:,:]
ells = np.arange(cij.shape[-1])
cij[0,0,:] = np.interp(ells,pr4_nkk[:,0],pr4_nkk[:,2],right=0)
cij[1,1,:] = np.interp(ells,dr6_nkk[:,0],dr6_nkk[:,2],right=0)
cij[0,1,:] = np.interp(ells,pr4dr6_nkk[:,0],pr4dr6_nkk[:,1],right=0)
cij[1,0,:] = np.interp(ells,pr4dr6_nkk[:,0],pr4dr6_nkk[:,1],right=0)

print('computing Cls and covariances')
pairs = [[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,2],[3,3],[4,4]]
full_master(ledges, maps, masks, names, fnout, lmin=20, lmax=1900, do_cov=True, cij=cij,
                only_auto=False, pairs=None, overwrite=True, overwrite_cov=True,bins=bins)