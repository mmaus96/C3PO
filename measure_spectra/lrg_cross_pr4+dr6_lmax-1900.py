from calc_cl import *
# sys.path.append('../')
sys.path.append('./mc_correction/')
from globe      import LEDGES,NSIDE
# from do_mc_corr import apply_mc_corr

y1_nside = None #1024

# get ACT alms and filter them
klm    = np.nan_to_num(hp.read_alm('./data/ACT_DR6/kappa_alm_data_act_dr6_lensing_v1_baseline.fits'))
filt   = np.ones(3*NSIDE) ; filt[3000:] = 0. 
klm    = hp.almxfl(klm,filt)

# load the other maps, masks and CMB lensing noise curves
if y1_nside == None:
    lrg_maps = [hp.read_map(f'./data/photo_z_dat/maps/lrg_s0{isamp}_del.hpx2048.fits.gz') for isamp in range(1,5)]
    lrg_masks = [hp.read_map(f'./data/photo_z_dat/maps/lrg_s0{isamp}_msk.hpx2048.fits.gz') for isamp in range(1,5)]
else:
    lrg_maps  = [hp.read_map(f'./data/photo_z_dat/maps/lrg_s0{isamp}_y1removed{y1_nside}_del.hpx2048.fits.gz') for isamp in range(1,5)]
    lrg_masks = [hp.read_map(f'./data/photo_z_dat/maps/lrg_s0{isamp}_y1removed{y1_nside}_msk.hpx2048.fits.gz') for isamp in range(1,5)]
pr4_map   = [hp.read_map(f'./data/planck_pr4/PR4_lens_kap_filt.hpx2048.fits')]
pr4_mask  = [hp.read_map(f'./data/planck_pr4/PR4_lens_mask.fits')]
dr6_map   = [hp.alm2map(klm,NSIDE)]
dr6_mask  = [hp.read_map(f'./data/ACT_DR6/DR6_lens_mask.fits')]
pr4_nkk   = np.loadtxt(f'./data/planck_pr4/PR4_lens_nlkk_filt.txt')
dr6_nkk   = np.loadtxt(f'./data/ACT_DR6/DR6_lens_nlkk.txt')
pr4dr6_nkk= np.loadtxt('./data/PR4xDR6_nkk.txt')

if y1_nside == None:
    fnout     = f'./data/photo_z_dat/Planck_ACT_DESI/lrg_cross_pr4+dr6_lmax-1900.json'
else:
    fnout     = f'./data/photo_z_dat/Planck_ACT_DESI/lrg_cross_pr4+dr6_lmax-1900_y1removed{y1_nside}.json'
    

# give our maps & masks some names
kapNames = ['PR4','DR6']
galNames = ['pLRG1','pLRG2','pLRG3','pLRG4']
names    = kapNames + galNames
msks     = pr4_mask + dr6_mask + lrg_masks
maps     = pr4_map  + dr6_map  + lrg_maps

# curves for covariance estimation
cij  = np.zeros((6,6,3*2048))
cij[1:,1:,:] = np.loadtxt(f'./data/photo_z_dat/fiducial/cls_LRGxPR4_bestFit.txt').reshape((5,5,3*2048))
cij[0,1:,:] = cij[1,1:,:]
cij[1:,0,:] = cij[1,1:,:]
ells = np.arange(cij.shape[-1])
cij[0,0,:] = np.interp(ells,pr4_nkk[:,0],pr4_nkk[:,2],right=0)
cij[1,1,:] = np.interp(ells,dr6_nkk[:,0],dr6_nkk[:,2],right=0)
cij[0,1,:] = np.interp(ells,pr4dr6_nkk[:,0],pr4dr6_nkk[:,1],right=0)
cij[1,0,:] = np.interp(ells,pr4dr6_nkk[:,0],pr4dr6_nkk[:,1],right=0)

# compute power spectra and window functions, save to json file
pairs = [[0,2],[0,3],[0,4],[0,5],[1,2],[1,3],[1,4],[1,5],[2,2],[3,3],[4,4],[5,5]]
full_master(LEDGES,maps,msks,names,fnout,cij=cij,do_cov=True,pairs=pairs,lmax=1900)

# apply MC correction
bdir = './mc_correction/sims/'
if y1_nside == None:
    apply_mc_corr(fnout,fnout,kapNames[0],galNames,[bdir+f'photoz/full_PR4/photo-lrg-full-z{i}_PR4-baseline' for i in range(1,5)])
    apply_mc_corr(fnout,fnout,kapNames[1],galNames,[bdir+f'photoz/full_DR6/photo-lrg-full-z{i}_DR6-baseline' for i in range(1,5)])
else:
    apply_mc_corr(fnout,fnout,kapNames[0],galNames,[bdir+f'photoz/y1filt{y1_nside}_PR4/photo-lrg-y1filt{y1_nside}-z{i}_PR4-baseline' for i in range(1,5)])
    apply_mc_corr(fnout,fnout,kapNames[1],galNames,[bdir+f'photoz/y1filt{y1_nside}_DR6/photo-lrg-y1filt{y1_nside}-z{i}_DR6-baseline' for i in range(1,5)])
