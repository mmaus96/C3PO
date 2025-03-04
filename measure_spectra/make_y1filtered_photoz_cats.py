import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

from astropy.table import Table, hstack,vstack,join
import fitsio
import healpy as hp
import glob

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc= comm.Get_size()

nside = 128
npix = hp.nside2npix(nside)
isnest= False

# Convert RA, Dec to radians for HEALPix
def radec_to_pix(ra, dec, nside):
    theta = np.radians(90 - dec)  # Convert Dec to colatitude
    phi = np.radians(ra)  # Convert RA to longitude
    return hp.ang2pix(nside, theta, phi)

outdir  = './data/photo_z_dat/'
db = '/global/cfs/cdirs/desi/users/rongpu/data/lrg_xcorr/catalogs/'
fn = 'dr9_lrg_pzbins_20230509.fits'
rb  = '/global/cfs/cdirs/desi/public/ets/target/catalogs/dr9/0.49.0/randoms/resolve/'
catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/'

if rank==0:
    print('Reading photo_z galaxy catalog')
    photo_gal = Table.read(db+fn)
else:
    photo_gal = None
photo_gal = comm.bcast(photo_gal,root=0)
if rank==0: 
    print('done')

    flist = glob.glob(rb+r'randoms-[0-9]-[0-9].fits')
    flist+= glob.glob(rb+r'randoms-[0-9]-1[0-9].fits')
    flist = sorted(flist)
else:
    flist = None
flist = comm.bcast(flist,root=0)

if rank==0:
    print('Reading spec_z randoms')
    randoms_ngc_fns = [catalog_dir + 'LRG_NGC_{}_clustering.ran.fits'.format(i) for i in range(18)]
    randoms_sgc_fns = [catalog_dir + 'LRG_SGC_{}_clustering.ran.fits'.format(i) for i in range(18)]
    randoms_ngc = vstack([Table.read(fn) for fn in randoms_ngc_fns])
    randoms_sgc = vstack([Table.read(fn) for fn in randoms_sgc_fns])
    randoms = vstack([randoms_ngc,randoms_sgc])
    print('making spec_rand_map')
    spec_rand_pixels = radec_to_pix(randoms['RA'], randoms['DEC'], nside)
    spec_rand_map, _ = np.histogram(spec_rand_pixels,bins=np.arange(npix+1)-0.5)
    print('making spec_mask')
    # Create a mask for spec_rand footprint
    spec_mask = np.zeros(npix, dtype=bool)
    spec_mask[spec_rand_pixels] = True
    hpex = ".hpx{:04d}.fits".format(nside)
    nmap = hp.ud_grade(spec_rand_map,nside)
    hp.write_map(outdir+'maps/specz_random_map'+hpex,nmap,dtype='f4',\
             nest=isnest,coord='G',overwrite=True)
    del randoms,randoms_ngc,randoms_sgc
else:
    spec_rand_pixels,spec_rand_map,spec_mask = None,None,None   
spec_rand_pixels = comm.bcast(spec_rand_pixels,root=0)
spec_rand_map = comm.bcast(spec_rand_map,root=0)
spec_mask = comm.bcast(spec_mask,root=0)

# if rank==0: 
#     print('done')
#     print('making spec_rand_map')
#     spec_rand_pixels = radec_to_pix(randoms['RA'], randoms['DEC'], nside)
#     spec_rand_map, _ = np.histogram(spec_rand_pixels,bins=np.arange(npix+1)-0.5)
#     print('making spec_mask')
#     # Create a mask for spec_rand footprint
#     spec_mask = np.zeros(npix, dtype=bool)
#     spec_mask[spec_rand_pixels] = True
# else:
#     spec_rand_pixels,spec_rand_map,spec_mask = None,None,None   
# spec_rand_pixels = comm.bcast(spec_rand_pixels,root=0)
# spec_rand_map = comm.bcast(spec_rand_map,root=0)
# spec_mask = comm.bcast(spec_mask,root=0)

if rank==0: 
    print('done')
    print('making photo galaxy map and mask')
    photo_gal_pixels = radec_to_pix(photo_gal['RA'], photo_gal['DEC'], nside)
    photo_gal_map, _ = np.histogram(photo_gal_pixels,bins=np.arange(npix+1)-0.5)
    gal_mask = ~spec_mask[photo_gal_pixels]
    print('making filtered photo galaxy catalog and saving')
    filtered_photo_gal = photo_gal[gal_mask]
    filtered_photo_gal.write(outdir + 'catalogs/photoz_galaxies_Y1filtered_{}'.format(nside), format='fits',overwrite=True)
    
    print('making filtered photo galaxy map') 
    filtered_photo_gal_pixels = radec_to_pix(filtered_photo_gal['RA'], filtered_photo_gal['DEC'], nside)
    filtered_photo_gal_map, _ = np.histogram(filtered_photo_gal_pixels,bins=np.arange(npix+1)-0.5)
else:
    photo_gal_map,filtered_photo_gal_map = None,None
    
photo_gal_map = comm.bcast(photo_gal_map,root=0)
filtered_photo_gal_map = comm.bcast(filtered_photo_gal_map,root=0)

photo_rand_map = np.zeros(npix,dtype='f8')
filtered_photo_rand_map = np.zeros(npix,dtype='f8')

if rank==0:
    print('done')
    print('loading photo randoms and filtering')

for i,fn in enumerate(flist):
    if i%nproc==rank:
        photo_rand = Table.read(fn)
        photo_rand_pixels = radec_to_pix(photo_rand['RA'], photo_rand['DEC'], nside)
        tmp, _ = np.histogram(photo_rand_pixels,bins=np.arange(npix+1)-0.5)
        photo_rand_map += tmp
        rand_mask = ~spec_mask[photo_rand_pixels]
        filtered_photo_rand = photo_rand[rand_mask]
        
        del photo_rand
        
        filtered_photo_rand.write(outdir + 'catalogs/photoz_randoms_{}_Y1filtered_{}'.format(i,nside), format='fits',overwrite=True)
        filtered_photo_rand_pixels = radec_to_pix(filtered_photo_rand['RA'], filtered_photo_rand['DEC'], nside)
        tmp2, _ = np.histogram(filtered_photo_rand_pixels,bins=np.arange(npix+1)-0.5)
        filtered_photo_rand_map += tmp2
        del filtered_photo_rand
        
# rtot = np.zeros(npix,dtype='f8')
# comm.Reduce(photo_rand_map,rtot,op=MPI.SUM,root=0)
# photo_rand_map = rtot

# filt_rtot = np.zeros(npix,dtype='f8')
# comm.Reduce(filtered_photo_rand_map,filt_rtot,op=MPI.SUM,root=0)
# filtered_photo_rand_map = filt_rtot

# if rank ==0:
#     print('done')
    
# hpex = ".hpx{:04d}.fits".format(nside)
# nmap = hp.ud_grade(photo_gal_map,nside)
# hp.write_map(outdir+'maps/photoz_galaxy_map'+hpex,nmap,dtype='f4',\
#          nest=isnest,coord='G',overwrite=True)
# nmap = hp.ud_grade(filtered_photo_gal_map,nside)
# hp.write_map(outdir+'maps/photoz_galaxy_map_Y1filtered'+hpex,nmap,dtype='f4',\
#          nest=isnest,coord='G',overwrite=True)

# nmap = hp.ud_grade(photo_rand_map,nside)
# hp.write_map(outdir+'maps/photoz_random_map'+hpex,nmap,dtype='f4',\
#          nest=isnest,coord='G',overwrite=True)
# nmap = hp.ud_grade(filtered_photo_rand_map,nside)
# hp.write_map(outdir+'maps/photoz_random_map_Y1filtered'+hpex,nmap,dtype='f4',\
#          nest=isnest,coord='G',overwrite=True)

# hpex = ".hpx{:04d}.fits".format(nside)
# nmap = hp.ud_grade(spec_rand_map,nside)
# hp.write_map(outdir+'maps/specz_random_map'+hpex,nmap,dtype='f4',\
#          nest=isnest,coord='G',overwrite=True)