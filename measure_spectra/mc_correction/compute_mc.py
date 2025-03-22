from do_mc_corr import *
import sys
sys.path.append('../')
from globe import NSIDE

job = int(sys.argv[2])
CMB = str(sys.argv[1])

# load some masks (which may or may not be used depending on the job)
PR4mask  = hp.read_map(f'/pscratch/sd/m/mmaus/DESI_RSD_kxg/data/planck_pr4/PR4_lens_mask.fits')
DR6mask = hp.read_map(f'/pscratch/sd/m/mmaus/DESI_RSD_kxg/data/ACT_DR6/DR6_lens_mask.fits')
kmasks = {'PR4': PR4mask, 'DR6': DR6mask}

bdir     = '/pscratch/sd/m/mmaus/DESI_RSD_kxg/data/'
if job <5: 
    isamp    = job
    photo_lrg_mask = hp.read_map(bdir+f'photo_z_dat/maps/lrg_s0{isamp}_msk.hpx2048.fits')
    make_mc_cls(f'photoz/full_{CMB}/photo-lrg-full-z{isamp}'  ,photo_lrg_mask, kmasks[CMB],'c',lensmap=CMB)
    
if (job >=5)&(job <9):
    y1nside = 1024
    isamp = job - 4
    photo_lrg_mask = hp.read_map(bdir+f'photo_z_dat/maps/lrg_s0{isamp}_y1removed{y1nside}_msk.hpx2048.fits')
    make_mc_cls(f'photoz/y1filt{y1nside}_{CMB}/photo-lrg-y1filt{y1nside}-z{isamp}'  ,\
                photo_lrg_mask, kmasks[CMB],'c',lensmap=CMB)
    
if (job >=9)&(job < 12):
    isamp = job - 8
    spec_lrg_mask = hp.read_map(f'../data/spec_z_dat/maps/specz_pixel_desi_LRG_z{isamp}_FKP_v1.5.hpx2048_msk.fits')
    make_mc_cls(f'specz/{CMB}/specz-lrg-full-z{isamp}'  ,spec_lrg_mask,kmasks[CMB],'c',lensmap=CMB)

if job == 12:
    spec_bgs_mask = hp.read_map(f'../data/spec_z_dat/maps/specz_pixel_desi_BGS_FKP_v1.5.hpx2048_msk.fits')
    make_mc_cls(f'specz/{CMB}/specz-bgs-full'  ,spec_bgs_mask,kmasks[CMB],'c',lensmap=CMB)