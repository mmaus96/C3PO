import sys
import numpy as np
import json

from copy import deepcopy
from pathlib import Path
from astropy.table import Table, hstack,vstack,join
import fitsio

import pymaster as nmt
import healpy as hp

COORD = 'C'
spec_z_bin = int(sys.argv[1])
# weight_type = 'default'
weight_type = 'FKP'
nlb = 50
bin_type = 'linear'
# bin_type = 'ledges'
cat_root = 'v1.5'

# Define pixelization parameters
nside = 2048  
npix = hp.nside2npix(nside)
isnest= False

lmax = 3*nside -1
ledges = [10,20,44,79,124,178,243,317,401,495,600,713,837]
ledges = ledges + (np.linspace(971**0.5,(3*nside)**0.5,20,endpoint=True)**2).astype(int).tolist()

rot = hp.rotator.Rotator(coord=['c',COORD])

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
        return np.array([np.radians(90-catalog['DEC']),np.radians(catalog['RA'])]), catalog['WEIGHT'] * catalog['WEIGHT_FKP']
    elif weight_type=='default':
        return np.array([np.radians(90-catalog['DEC']),np.radians(catalog['RA'])]), catalog['WEIGHT']
    
def get_cl(f1, f2,b):
    w = nmt.NmtWorkspace.from_fields(f1, f2, b)
    cl = nmt.compute_coupled_cell(f1, f2)
    cl_dec = w.decouple_cell(cl)
    return cl,cl_dec,w    

def approx_cl(cl,lmax=3000,expfit=True):
    lval  = np.arange(lmax+1)
    clog = np.log(np.abs(cl[0,:]))
    coeff = np.polyfit(ell,clog,14,w=1/clog**2)
    cl_fit = np.exp(np.poly1d(coeff)(lval))
    return cl_fit

def radec_to_pix(ra, dec, nside):
    theta = np.radians(90 - dec)  # Convert Dec to colatitude
    phi = np.radians(ra)  # Convert RA to longitude
    return hp.ang2pix(nside, theta, phi)
    
        
#Load cmb map:
print('loading CMB lensing field and creating field object...')
kappa = hp.read_map('./data/planck_pr4/PR4_lens_kap_filt.hpx2048.fits')
lens_mask = hp.read_map('./data/planck_pr4/PR4_lens_mask.fits')
f_lens = nmt.NmtField(lens_mask, [kappa])
print('done')

#load clustering catalogs:
print('loading LRG catalogs...')
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
print('n_galaxies: ', len(data))
print('n_randoms: ', len(randoms))
#get positions and weights
print('getting positions and weights for zbin: ', spec_z_bin)
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
print('done')

print('creating galaxy map...')
# Initialize maps
# galaxy_map = np.zeros(npix)
# random_map = np.zeros(npix)
# w_map = np.zeros(npix)

theta,phi = data_positions[0],data_positions[1]
theta,phi = rot(theta,phi) # C->G coordinates
pixnum    = hp.ang2pix(nside,theta,phi)
# print(len(pixnum),len(data_weights))
# gal_pix = radec_to_pix(data['RA'], data['DEC'], nside)
galaxy_map, _ = np.histogram(pixnum,weights=data_weights,bins=np.arange(npix+1)-0.5)
print('done')

print('creating random map...')
theta,phi = random_positions[0],random_positions[1]
theta,phi = rot(theta,phi) # C->G coordinates
pixnum    = hp.ang2pix(nside,theta,phi,nest=isnest)
# rand_pix = radec_to_pix(randoms['RA'], randoms['DEC'], nside)
random_map, _ = np.histogram(pixnum,weights=random_weights,bins=np.arange(npix+1)-0.5)

print('done')

print('creating  wmap...')

wmap, _ = np.histogram(pixnum,bins=np.arange(npix+1)-0.5)
wmap = random_map/(wmap+1e-30)

print('done')

# # Fill galaxy map
# for ii,pos in enumerate(data_positions):
#     # print(pos)
#     ipix = hp.ang2pix(nside, pos[0], pos[1])  # Ensure proper ang/deg conversion
#     galaxy_map[ipix] += data_weights[ii] # Add weights
    
# for ii,pos in enumerate(random_positions):
#     # print(pos)
#     ipix = hp.ang2pix(nside, pos[0], pos[1])
#     random_map[ipix] += random_weights[ii] 
#     wmap[ipix] += 1.


    
# Mask where there are at least N random points (threshold can be adjusted)
msk  = np.nonzero(random_map>0)[0]
avg  = np.mean(random_map[msk])

msk  = np.nonzero(random_map>0.20*avg)[0]

# Now fill in the masked region.
omap      = np.zeros(npix,dtype='f8')
omap[msk] = galaxy_map[msk]/random_map[msk]
omap[msk] = omap[msk]/np.mean(omap[msk]) - 1
# omap      = omap.astype('f4')   # Don't need full precision.
mask      = np.zeros(npix,dtype='f4')
mask[msk] = 1.0

# Print some useful numbers.
shot = np.sum(galaxy_map[msk]/wmap[msk])**2/np.sum(galaxy_map[msk]/wmap[msk]**2)
shot = np.sum(mask)*hp.nside2pixarea(nside,False)/shot
ninv = np.sum(mask)*hp.nside2pixarea(nside,False)/np.sum(galaxy_map[msk])
nbar = 1.0/ninv * (np.pi/180.)**2 # Per sq.deg.
print("Mean of omap is {:e}.".format(np.mean(omap)))
print("nbar         is {:f}/deg2".format(nbar))
print("1/nbar       is {:e}".format(ninv))
print("Shot noise   is {:e}".format(shot))
print("Sky fraction is {:f}.".format(np.sum(mask)/mask.size))

print('done')

print('saving galaxy mask and overdensity map')
fnout = './data/spec_z_dat/maps/specz_pixel_desi_LRG_z{}_{}_{}.hpx{:04d}_msk.fits'.format(spec_z_bin,weight_type,cat_root,nside)
nmap = hp.ud_grade(mask,nside)
hp.write_map(fnout,nmap,dtype='f4',\
                     nest=False,coord='C',overwrite=True)

fnout = './data/spec_z_dat/maps/specz_pixel_desi_LRG_z{}_{}_{}.hpx{:04d}_map.fits'.format(spec_z_bin,weight_type,cat_root,nside)
nmap = hp.ud_grade(omap,nside)
hp.write_map(fnout,nmap,dtype='f4',\
                     nest=False,coord='C',overwrite=True)

print('data saved')

#create Y1 footprint mask with several resolutions:


# for n in range(5,12):
#     nside_y1 = 2**n
#     npix_y1 = hp.nside2npix(nside_y1)
#     randoms_pixels = radec_to_pix(randoms['RA'], randoms['DEC'], nside_y1)
#     rmap, _ = np.histogram(randoms_pixels,weights=random_weights,bins=np.arange(npix_y1+1)-0.5)
#     wmap , _ = np.histogram(randoms_pixels,bins=np.arange(npix_y1+1)-0.5)
    
#     msk  = np.nonzero(rmap>0)[0]
#     avg  = np.mean(rmap[msk])
#     msk  = np.nonzero(rmap>0.20*avg)[0]
#     print("Have {:8.2f} randoms  per masked pixel for nside={}.".format(np.mean(wmap[msk]),nside_y1))
    
#     mask = np.ones(npix_y1, dtype=bool)
#     mask[randoms_pixels] = False
#     mask = hp.ud_grade(omap,nside)
#     fnout = './data/spec_z_dat/maps/y1_footprint_mask_{}.fits'.format(nside)
# print('data saved')