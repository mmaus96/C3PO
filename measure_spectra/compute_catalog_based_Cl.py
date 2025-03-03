import sys
import numpy as np
import json

from copy import deepcopy
from pathlib import Path
from astropy.table import Table, hstack,vstack,join
import fitsio

import pymaster as nmt
import healpy as hp

spec_z_bin = int(sys.argv[1])
# weight_type = 'default'
weight_type = 'FKP'
nlb = 50
# bin_type = 'linear'
bin_type = 'ledges'
cat_root = 'v1.5'

knm = 'DR6' #'PR4'
gnm = f'LRG{spec_z_bin}'

nside = 2048
lmax = 3*nside -1
ledges = [10,20,44,79,124,178,243,317,401,495,600,713,837]
ledges = ledges + (np.linspace(971**0.5,(3*nside)**0.5,20,endpoint=True)**2).astype(int).tolist()


# b = nmt.NmtBin.from_lmax_linear(lmax, nlb=nlb)
# leff = b.get_effective_ells()

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

def get_positions_weights(catalog,weight_type = 'FKP'):
        if weight_type=='FKP':
            return [np.radians(90-catalog['DEC']),np.radians(catalog['RA'])], catalog['WEIGHT'] * catalog['WEIGHT_FKP']
        elif weight_type =='default':
            return [np.radians(90-catalog['DEC']),np.radians(catalog['RA'])], catalog['WEIGHT']
        
def get_cl(f1, f2,b):
    w = nmt.NmtWorkspace.from_fields(f1, f2, b)
    cl = nmt.compute_coupled_cell(f1, f2)
    cl_dec = w.decouple_cell(cl)
    return cl,cl_dec,w        

#load Lensing Maps:

if knm == 'PR4':
    kappa = hp.read_map('./data/planck_pr4/PR4_lens_kap_filt.hpx2048.fits')
    lens_mask = hp.read_map('./data/planck_pr4/PR4_lens_mask.fits')
if knm == 'DR6':
    klm    = np.nan_to_num(hp.read_alm('./data/ACT_DR6/kappa_alm_data_act_dr6_lensing_v1_baseline.fits'))
    filt   = np.ones(3*nside) ; filt[3000:] = 0. 
    klm    = hp.almxfl(klm,filt)
    kappa   = hp.alm2map(klm,nside)
    lens_mask  = hp.read_map(f'./data/ACT_DR6/DR6_lens_mask.fits')

f_pix = nmt.NmtField(lens_mask, [kappa])

print('lens field computed')

#Load clustering catalogs:
# cat_path = './data/full_cats/'
catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{}/'.format(cat_root)
randoms_ngc_fns = [catalog_dir + 'LRG_NGC_{}_clustering.ran.fits'.format(i) for i in range(18)]
randoms_sgc_fns = [catalog_dir + 'LRG_SGC_{}_clustering.ran.fits'.format(i) for i in range(18)]

data_ngc_fn = catalog_dir + 'LRG_NGC_clustering.dat.fits'
data_sgc_fn = catalog_dir + 'LRG_SGC_clustering.dat.fits'
data_ngc = Table(fitsio.read(data_ngc_fn))
data_sgc = Table(fitsio.read(data_sgc_fn))
data = vstack([data_ngc,data_sgc])
# randoms = Table(fitsio.read(cat_path + 'randoms_0_full_v1_desiweights_joined.fits'))
randoms_ngc = vstack([Table(fitsio.read(fn)) for fn in randoms_ngc_fns])
randoms_sgc = vstack([Table(fitsio.read(fn)) for fn in randoms_sgc_fns])
randoms = vstack([randoms_ngc,randoms_sgc])
print('catalog data and randoms loaded')

if spec_z_bin == 1:
    z_min,z_max = 0.4, 0.6
elif spec_z_bin == 2:
    z_min,z_max = 0.6, 0.8
elif spec_z_bin == 3:
    z_min,z_max = 0.8, 1.1

zcut_rand = (randoms['Z'] <= z_max) & (randoms['Z'] >= z_min)
zcut_dat = (data['Z'] <= z_max) & (data['Z'] >= z_min)
    
data_positions, data_weights = get_positions_weights(data[zcut_dat],weight_type)
random_positions, random_weights = get_positions_weights(randoms[zcut_rand],weight_type)

print('acquired positions and weights')

f_cat = nmt.NmtFieldCatalogClustering(data_positions, data_weights, random_positions, random_weights, lmax=lmax, lonlat=False)

print('prepared catalog field')

if bin_type == 'linear':
    bins = nmt.NmtBin.from_lmax_linear(lmax, nlb=nlb) #
    fnout = f'./data/spec_z_dat/kxg_cl/cl_ggkg_{knm}_LRG_z{spec_z_bin}_{weight_type}_nlb{nlb}_{cat_root}.json'
else:
    bins = get_bins(ledges)
    fnout = f'./data/spec_z_dat/kxg_cl/cl_ggkg_{knm}_LRG_z{spec_z_bin}_{weight_type}_{cat_root}.json'

cl_kg,cl_kg_dec,w_kg = get_cl(f_cat, f_pix,bins)

print('computed cross spectra')

wmat_kg = w_kg.get_bandpower_windows()
mmat_kg = w_kg.get_coupling_matrix()

print('got window and coupling matrix')

cl_gg,cl_gg_dec,w_gg = get_cl(f_cat, f_cat,bins)

print('computed auto spectra')

wmat_gg = w_gg.get_bandpower_windows()
mmat_gg = w_gg.get_coupling_matrix()

print('got window and coupling matrix')

leffs = bins.get_effective_ells()

outdict = {'nside':nside, 'lmax': lmax, 'ledges': ledges, 'ell': leffs.tolist(),'weight_type': weight_type,\
          f'cl_coupled_{knm}_{gnm}': cl_kg.tolist(), f'cl_{knm}_{gnm}': cl_kg_dec.tolist(),\
           f'w_{knm}_{gnm}': wmat_kg.tolist(), f'm_{knm}_{gnm}': mmat_kg.tolist(),\
           f'cl_coupled_{gnm}_{gnm}': cl_gg.tolist(), f'cl_{gnm}_{gnm}': cl_gg_dec.tolist(),\
           f'w_{gnm}_{gnm}': wmat_gg.tolist(), f'm_{gnm}_{gnm}': mmat_gg.tolist()
          }


with open(fnout, "w") as outfile:
    json.dump(outdict, outfile)

print('data saved')
