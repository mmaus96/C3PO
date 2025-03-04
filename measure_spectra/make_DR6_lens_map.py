import numpy    as np
import healpy   as hp
import urllib.request
import os
import sys
sys.path.append('../')
from globe import NSIDE

website  = 'https://portal.nersc.gov/project/act/dr6_lensing_v1/'
website2 = 'https://phy-act1.princeton.edu/public/data/dr6_lensing_v1/'

# download the mask and update the resolution 
# save to masks/nmaskfn and delete original file
maskfn  = 'mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits'
nmaskfn = 'DR6_lens_mask.fits'
urllib.request.urlretrieve(website+'maps/baseline/'+maskfn, maskfn)
mask = hp.ud_grade(hp.read_map(maskfn),NSIDE)
hp.write_map(f'./data/ACT_DR6/{nmaskfn}',mask,dtype='f4',overwrite=True)
os.system(f'rm {maskfn}')

# download Ckk and Nkk, save to ../data/DR6_lens_nlkk.txt
ckkfn = 'clkk.txt'
nkkfn = 'N_L_kk_act_dr6_lensing_v1_baseline.txt'
urllib.request.urlretrieve(website2+'misc/'         +ckkfn, ckkfn)
urllib.request.urlretrieve(website +'maps/baseline/'+nkkfn, nkkfn)
ckk = np.loadtxt(ckkfn)
nkk = np.loadtxt(nkkfn)[:,1]
N   = min(len(ckk),len(nkk))
dat = np.array([np.arange(N),nkk[:N],ckk[:N]+nkk[:N]]).T
with open("./data/ACT_DR6/DR6_lens_nlkk.txt","w") as fout:
    fout.write("# ACT DR6 lensing noise curves.\n")
    fout.write("# {:>6s} {:>15s} {:>15s}\n".format("ell","Noise","Sig+Noise"))
    for i in range(dat.shape[0]):
        fout.write("{:8.0f} {:15.5e} {:15.5e}\n".format(dat[i,0],dat[i,1],dat[i,2]))
os.system(f'rm {ckkfn}')
os.system(f'rm {nkkfn}')

# download the CMB lensing map
klmfn = 'kappa_alm_data_act_dr6_lensing_v1_baseline.fits'
urllib.request.urlretrieve(website+'maps/baseline/'+klmfn, klmfn)