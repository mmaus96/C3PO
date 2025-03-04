import sys
import numpy as np
import json

from copy import deepcopy
from pathlib import Path
from astropy.table import Table, hstack,vstack,join
import fitsio

import pymaster as nmt
import healpy as hp

# spec_z_bin = int(sys.argv[1])
# weight_type = 'default'
# weight_type = 'FKP'
# nlb = 50
# bin_type = 'linear'
# bin_type = 'ledges'
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc= comm.Get_size()

cat_root = 'v1.5'

# Define pixelization parameters
nside = 2048  
npix = hp.nside2npix(nside)

#load clustering catalogs:
print('loading LRG catalogs...')
catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{}/'.format(cat_root)
randoms_ngc_fns = [catalog_dir + 'LRG_NGC_{}_clustering.ran.fits'.format(i) for i in range(18)]
randoms_sgc_fns = [catalog_dir + 'LRG_SGC_{}_clustering.ran.fits'.format(i) for i in range(18)]
randoms_fns = np.concatenate((randoms_ngc_fns,randoms_sgc_fns))
data_ngc_fn = catalog_dir + 'LRG_NGC_clustering.dat.fits'
data_sgc_fn = catalog_dir + 'LRG_SGC_clustering.dat.fits'
# data_ngc = Table(fitsio.read(data_ngc_fn))
# data_sgc = Table(fitsio.read(data_sgc_fn))
# data = vstack([data_ngc,data_sgc])
# randoms = Table(fitsio.read(cat_path + 'randoms_0_full_v1_desiweights_joined.fits'))


#create Y1 footprint mask with several resolutions:
def radec_to_pix(ra, dec, nside):
    theta = np.radians(90 - dec)  # Convert Dec to colatitude
    phi = np.radians(ra)  # Convert RA to longitude
    return hp.ang2pix(nside, theta, phi)

for n in range(5,12):
    nside_y1 = 2**n
    npix_y1 = hp.nside2npix(nside_y1)
    mask = np.ones(npix_y1, dtype=bool)
    wmap = np.zeros(npix_y1,dtype='f8')
    rmap = np.zeros(npix_y1,dtype='f8')
    for i,fn in enumerate(randoms_fns):
        if i%nproc==rank:
            randoms = Table.read(fn)
            randoms_pixels = radec_to_pix(randoms['RA'], randoms['DEC'], nside_y1)
            tmp, _ = np.histogram(randoms_pixels,weights=randoms['WEIGHT'] * randoms['WEIGHT_FKP'],bins=np.arange(npix_y1+1)-0.5)
            rmap += tmp
            tmp , _ = np.histogram(randoms_pixels,bins=np.arange(npix_y1+1)-0.5)
            wmap += tmp
            mask[randoms_pixels] = False
    
    rtot = np.zeros(npix_y1,dtype='f8')
    comm.Reduce(rmap,rtot,op=MPI.SUM,root=0)
    rmap = rtot
    
    wtot = np.zeros(npix_y1,dtype='f8')
    comm.Reduce(wmap,wtot,op=MPI.SUM,root=0)
    wmap = wtot
    msk  = np.nonzero(rmap>0)[0]
    avg  = np.mean(rmap[msk])
    msk  = np.nonzero(rmap>0.20*avg)[0]
    
    if rank==0:
        print("Have {:8.2f} randoms  per masked pixel for nside={}.".format(np.mean(wmap[msk]),nside_y1))


        mask = hp.ud_grade(mask,nside)
        fnout = './data/spec_z_dat/maps/y1_footprint_mask_{}_{}.fits'.format(nside_y1,cat_root)
        hp.write_map(fnout,mask,dtype='f4',\
                         nest=False,coord='C',overwrite=True)