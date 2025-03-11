import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from assign_randoms_weights import get_randoms_weights

from astropy.table import Table, hstack,vstack,join
import fitsio
import healpy as hp
import glob

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc= comm.Get_size()

nside = 1024
npix = hp.nside2npix(nside)
isnest= False

# Convert RA, Dec to radians for HEALPix
def radec_to_pix(ra, dec, nside):
    theta = np.radians(90 - dec)  # Convert Dec to colatitude
    phi = np.radians(ra)  # Convert RA to longitude
    return hp.ang2pix(nside, theta, phi)

def toMag(depth,ext,ebv):
    """A 'safe' conversion of depth to magnitude."""
    dd = np.sqrt( depth.clip(1e-30,1e30) )
    mag= -2.5*(np.log10(5/dd)-9) - ext*ebv
    return(mag)

outdir  = './data/photo_z_dat/'
db = '/global/cfs/cdirs/desi/public/papers/c3/lrg_xcorr_2023/v1/catalogs/'
fn = 'dr9_lrg_pzbins.fits'
# rb  = '/global/cfs/cdirs/desi/public/ets/target/catalogs/dr9/0.49.0/randoms/resolve/'
rb  = '/dvs_ro/cfs/cdirs/desi/public/ets/target/catalogs/dr9/0.49.0/randoms/resolve/'
catalog_dir = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/'

weights_path = db + 'imaging_weights/main_lrg_linear_coeffs_pz.yaml'


sfn  = '/global/cfs/cdirs/desi/public/papers/c3/lrg_xcorr_2023/v1/misc/pixweight-dr7.1-0.22.0_stardens_64_ring.fits'
pxw_nside = 256
pxw_nest  = True
# Want to only read this once if possible.
if rank==0:
    stardens = fitsio.read(sfn)
else:
    stardens = None
stardens = comm.bcast(stardens,root=0)
# Downgrade the stellar density map to remove the many holes
# and islands that exist at Nside=256.
stardens_nside = 64
ebv_cut=0.15
star_cut=2500.

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
    spec_mask = hp.read_map(f'/pscratch/sd/m/mmaus/DESI_RSD_kxg/data/spec_z_dat/maps/y1_footprint_mask_{nside}_v1.5.fits',dtype=bool)
else:
    spec_mask = None
spec_mask = comm.bcast(spec_mask,root=0)



if rank==0: 
    print('done')
    print('making photo galaxy map and mask')
    photo_gal_pixels = radec_to_pix(photo_gal['RA'], photo_gal['DEC'], nside)
    # photo_gal_map, _ = np.histogram(photo_gal_pixels,bins=np.arange(npix+1)-0.5)
    gal_mask = spec_mask[photo_gal_pixels]
    print('making filtered photo galaxy catalog and saving')
    filtered_photo_gal = photo_gal[gal_mask]
    filtered_photo_gal.write(outdir + 'catalogs/photoz_galaxies_Y1filtered_{}'.format(nside), format='fits',overwrite=True)
    
    # print('making filtered photo galaxy map') 
    # filtered_photo_gal_pixels = radec_to_pix(filtered_photo_gal['RA'], filtered_photo_gal['DEC'], nside)
    # filtered_photo_gal_map, _ = np.histogram(filtered_photo_gal_pixels,bins=np.arange(npix+1)-0.5)
else:
    photo_gal_map,filtered_photo_gal_map = None,None
    
# photo_gal_map = comm.bcast(photo_gal_map,root=0)
# filtered_photo_gal_map = comm.bcast(filtered_photo_gal_map,root=0)

photo_rand_map = np.zeros(npix,dtype='f8')
filtered_photo_rand_map = np.zeros(npix,dtype='f8')

mb  = '/global/cfs/cdirs/desi/public/papers/c3/lrg_xcorr_2023/v1/catalogs/lrgmask_v1.1/'
mb + fn[len(rb):-5] + '-lrgmask_v1.1.fits.gz'
columns = ['TARGETID','RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'MASKBITS', 'EBV','GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z','PHOTSYS']

if rank==0:
    print('done')
    print('loading photo randoms and filtering')

for i,fn in enumerate(flist[128:]):
    i+=128
    if i%nproc==rank:
        print(f'Starting Randoms {i}')
        # tt = Table.read(fn)

        
        randoms = Table(fitsio.read(fn, columns=columns))
        mf = mb + fn[len(rb):-5] + '-lrgmask_v1.1.fits.gz'
        lrgmask = Table(fitsio.read(mf))
        randoms = hstack([randoms, lrgmask])

        target_min_nobs = 1
        target_maskbits = sorted([1, 12, 13])
        
        mask = (randoms['NOBS_G']>=target_min_nobs) & (randoms['NOBS_R']>=target_min_nobs) & (randoms['NOBS_Z']>=target_min_nobs)
        randoms = randoms[mask]
        
        mask = np.ones(len(randoms), dtype=bool)
        for bit in target_maskbits:
            mask &= (randoms['MASKBITS'] & 2**bit)==0
        randoms = randoms[mask]

        ###################### Create the same survey geometry as the LRG catalogs ######################

        min_nobs = 2
        max_ebv = 0.15
        max_stardens = 2500
        
        # Remove "islands" in the NGC
        mask = ~((randoms['DEC']<-10.5) & (randoms['RA']>120) & (randoms['RA']<260))
        # print('Remove islands', np.sum(mask), np.sum(~mask), np.sum(mask)/len(mask))
        randoms = randoms[mask]
        
        # NOBS cut
        mask = (randoms['NOBS_G']>=min_nobs) & (randoms['NOBS_R']>=min_nobs) & (randoms['NOBS_Z']>=min_nobs)
        # print('NOBS', np.sum(mask), np.sum(~mask), np.sum(mask)/len(mask))
        randoms = randoms[mask]
        
        # Apply LRG mask
        mask = randoms['lrg_mask']==0
        # print('LRG mask', np.sum(mask), np.sum(~mask), np.sum(~mask)/len(mask))
        randoms = randoms[mask]
        
        # EBV cut
        mask = randoms['EBV']<max_ebv
        # print('EBV', np.sum(mask), np.sum(~mask), np.sum(~mask)/len(mask))
        randoms = randoms[mask]

        #Stellar density cut
        mask = stardens['STARDENS']>=star_cut
        bad_hp_idx = stardens['HPXPIXEL'][mask]
        cat_hp_idx = hp.pixelfunc.ang2pix(stardens_nside, randoms['RA'], randoms['DEC'], lonlat=True, nest=False)
        mask_bad = np.in1d(cat_hp_idx, bad_hp_idx)
        # print('STARDENS', np.sum(~mask_bad), np.sum(mask_bad), np.sum(mask_bad)/len(mask_bad))
        randoms = randoms[~mask_bad]
        
        # Random weights want these fields.
        randoms['galdepth_gmag_ebv'] = toMag(randoms['GALDEPTH_G' ],3.214,randoms['EBV'])
        randoms['galdepth_rmag_ebv'] = toMag(randoms['GALDEPTH_R' ],2.165,randoms['EBV'])
        randoms['galdepth_zmag_ebv'] = toMag(randoms['GALDEPTH_Z' ],1.211,randoms['EBV'])
        # randoms['psfdepth_w1mag_ebv']= toMag(randoms['PSFDEPTH_W1'],0.184,randoms['EBV'])
        # randoms['psfdepth_w2mag_ebv']= toMag(randoms['PSFDEPTH_W2'],0.113,randoms['EBV'])

        # filter out Y1 footprint
        #spec_mask has 0s in Y1 footprint and 1s outside
        photo_rand_pixels = radec_to_pix(randoms['RA'], randoms['DEC'], nside)
        rand_mask = spec_mask[photo_rand_pixels]
        randoms = randoms[rand_mask]
        print(f'Applied cuts to Randoms {i}, computing weights')
        # Now weight the randoms (or not).
        # if weights_path is not None:
        save_cols = ['RA','DEC']
        for pz_bin in range(1,5):
            wt = get_randoms_weights(randoms,weights_path,pz_bin)
            randoms[f'weight_{pz_bin}'] = wt
            save_cols.append(f'weight_{pz_bin}')
        randoms = randoms[save_cols]        
        randoms.write(outdir + 'catalogs/photoz_randoms_{}_Y1filtered_{}'.format(i,nside), format='fits',overwrite=True)
        # filtered_photo_rand_pixels = radec_to_pix(filtered_photo_rand['RA'], filtered_photo_rand['DEC'], nside)
        # tmp2, _ = np.histogram(filtered_photo_rand_pixels,bins=np.arange(npix+1)-0.5)
        # filtered_photo_rand_map += tmp2
        del randoms
        print(f'Finished Randoms {i}')
