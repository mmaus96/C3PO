import numpy as np

from classy import Class
from compute_xiell_tables_recsym import compute_xiells, compute_bao_pkmu, kint,sphr
# from linear_theory import f_of_a
# from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
# from shapefit import shapefit_factor

# k vector to use:

from scipy.special import spherical_jn
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
from linear_theory import*
from pnw_dst import pnw_dst



class direct_fit_theory():
    
    def __init__(self, zs, pars = None):
        if pars != None:
            w, omega_b,omega_cdm, h, logA, ns = pars
        else:
            w, omega_b,omega_cdm, h, logA, ns = [-1., 0.02237, 0.12, 0.6736, np.log(1e10 * 2.0830e-9), 0.9649]

        nnu = 1
        nur = 2.0328
        # mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106
        
        As =  np.exp(logA)*1e-10
        w0 = w
        wa = 0.

        # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2

        pkparams = {
            'output': 'mPk',
            'P_k_max_h/Mpc': 20.,
            'z_pk': '0.0,10',
            'A_s': As,
            'n_s': ns,
            'h': h,
            'N_ur': nur,
            'N_ncdm': nnu,
            'omega_ncdm': omega_nu,
            # 'm_ncdm': mnu,
            'tau_reio': 0.0568,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            'Omega_Lambda': 0.,
            'w0_fld': w0,
            'wa_fld': wa}

        fid_class = Class()
        fid_class.set(pkparams)
        fid_class.compute()
        
        self.theta_star = fid_class.theta_star_100()
        
        self.fid_class = fid_class
        
        self.kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )

        self.zs = zs
        
        self.fid_dists = {}

        for z in zs:
            zstr = "%.2f" %(z)
            self.fid_dists[zstr] = self.get_fid_dists(z)
        
    def get_fid_dists(self,z):
        
        speed_of_light = 2.99792458e5
        
        fid_class = self.fid_class
        h = fid_class.h()
        
        Hz_fid = fid_class.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
        chiz_fid = fid_class.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
        fid_dists = (Hz_fid, chiz_fid)
        
        return fid_dists

    def compute_pkclass(self,pars):
    
        w0,wa,Omega_m, theta_star, logA = pars
        # speed_of_light = 2.99792458e5
    
        omega_b = 0.02237
    
        As =  np.exp(logA)*1e-10 #2.0830e-9
        ns = 0.9649
    
        nnu = 1
        nur = 2.0328
        # mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106
            
        # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2
        # OmegaM = (omega_cdm + omega_b + omega_nu) / h**2
    
        # w0,wa = w0wa_from_wpdH(wp,dH,h*100,OmegaM)
        # print('w0,wa=', (w0,wa) )
        pkparams = {
            'output': 'mPk',
            'P_k_max_h/Mpc': 20.,
            'z_pk': '0.0,10',
            'A_s': As,
            'n_s': ns,
            # 'h': h,
            '100*theta_s': theta_star,
            'N_ur': nur,
            'N_ncdm': nnu,
            'omega_ncdm': omega_nu,
            # 'm_ncdm': mnu,
            # 'tau_reio': 0.0568,
            'z_reio': 7.,
            'omega_b': omega_b,
            'Omega_m': Omega_m,
            'Omega_Lambda': 0.,
            'w0_fld': w0,
            'wa_fld': wa}
    
        pkclass = Class()
        pkclass.set(pkparams)
        pkclass.compute()

        self.pkclass = pkclass
        self.h = pkclass.h()
        self.sigma8_0 = pkclass.sigma8()
        
        return pkclass,self.h,self.sigma8_0


    def compute_xiell_tables(self,z,  R=15., rmin=50, rmax=160, dr=0.5, sigs = (0.,0.,0.)):
    
        # w0,wa,Omega_m, theta_star, logA = pars
        sig_s,sig_par,sig_perp = sigs

        zstr = "%.2f" %(z)
        Hzfid, chizfid = self.fid_dists[zstr]

        h = self.h
        pkclass = self.pkclass
        
        speed_of_light = 2.99792458e5
    
        # Caluclate AP parameters
        Hz = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
        chiz = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
        apar, aperp = Hzfid / Hz, chiz / chizfid
        
        # Calculate growth rate
        # fnu = pkclass.Omega_nu / pkclass.Omega_m()
        # f0   = f_of_a(1/(1.+z), OmegaM=OmegaM) * (1 - 0.6 * fnu)
        f0 = pkclass.scale_independent_growth_factor_f(z)
    
        # Calculate and renormalize power spectrum
        ki = np.logspace(-3.0,1.0,200)
        pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in ki] )
        # pi = (sigma8/pkclass.sigma8())**2 * pi
    
        # Do the Zeldovich reconstruction predictions
    
        knw, pnw = pnw_dst(ki, pi)
        pw = pi - pnw
                
        qbao   = pkclass.rs_drag() * h # want this in Mpc/h units
    
        j0 = spherical_jn(0,ki*qbao)
        Sk = np.exp(-0.5*(ki*R)**2)
    
        sigmads_dd = simps( 2./3 * pi * (1-Sk)**2, x = ki) / (2*np.pi**2)
        # sigmads_ss = simps( 2./3 * pi * (-Sk)**2, x = ki) / (2*np.pi**2)
        # sigmads_ds = -simps( 2./3 * pi * (1-Sk)*(-Sk)*j0, x = ki) / (2*np.pi**2) # this minus sign is because we subtract the cross term
    
        sigmas = (sigmads_dd, sigs[0], sigs[1], sigs[2])
        
        # Now make the multipoles!
        klin, plin = ki, pi
        routs = np.arange(rmin, rmax, dr)
        
        # this is 1
        xi0_00,xi2_00  = compute_xiells(routs, 0, 0, klin, plin, pnw, f0, apar, aperp, R, sigmas)
    
        # this is 1 + B1 + B1^2
        # and 1 + 2 B1 + 4 B1^2
        xi0_10,xi2_10  = compute_xiells(routs,1, 0, klin, plin, pnw, f0, apar, aperp, R, sigmas)
        xi0_20,xi2_20  = compute_xiells(routs,2, 0, klin, plin, pnw, f0, apar, aperp, R, sigmas)
        
        # this is 1 + F + F^2
        # and 1 + 2 F + 4 F^2
        xi0_01,xi2_01  = compute_xiells(routs,0, 1, klin, plin, pnw, f0, apar, aperp, R, sigmas)
        xi0_02,xi2_02  = compute_xiells(routs,0, 2, klin, plin, pnw, f0, apar, aperp, R, sigmas)
        
        # and 1 + B1 + F + B1^2 + F^2 + BF
        xi0_11,xi2_11 = compute_xiells(routs,1, 1, klin, plin, pnw, f0, apar, aperp, R, sigmas)
        
        xi0table, xi2table = np.zeros( (len(routs),6) ), np.zeros( (len(routs),6) )
        
        # Form combinations:
        xi0_B1 = 0.5 * (4 * xi0_10 - xi0_20 - 3*xi0_00)
        xi0_B1sq = xi0_10 - xi0_B1 - xi0_00
    
        xi0_F = 0.5 * (4 * xi0_01 - xi0_02 - 3*xi0_00)
        xi0_Fsq = xi0_01 - xi0_F - xi0_00
    
        xi0_BF = xi0_11 - xi0_B1 - xi0_F - xi0_B1sq - xi0_Fsq - xi0_00
        
        xi2_B1 = 0.5 * (4 * xi2_10 - xi2_20 - 3*xi2_00)
        xi2_B1sq = xi2_10 - xi2_B1 - xi2_00
    
        xi2_F = 0.5 * (4 * xi2_01 - xi2_02 - 3*xi2_00)
        xi2_Fsq = xi2_01 - xi2_F - xi2_00
    
        xi2_BF = xi2_11 - xi2_B1 - xi2_F - xi2_B1sq - xi2_Fsq - xi2_00
        
        # Load
        xi0table[:,0] = xi0_00
        
        xi0table[:,1] = xi0_B1
        xi0table[:,2] = xi0_F
    
        xi0table[:,3] = xi0_B1sq
        xi0table[:,4]= xi0_Fsq
    
        xi0table[:,5] = xi0_BF
        
        xi2table[:,0] = xi2_00
        
        xi2table[:,1] = xi2_B1
        xi2table[:,2] = xi2_F
    
        xi2table[:,3] = xi2_B1sq
        xi2table[:,4]= xi2_Fsq
    
        xi2table[:,5] = xi2_BF
        
    
        return routs,xi0table, xi2table, pkclass.scale_independent_growth_factor(z),f0



    def compute_pell_tables_wcdm(self, pars, z, fid_dists= (None,None), ap_off=False, w0wa = False ):

        if w0wa:
            w0,wa, omega_b,omega_cdm, h, logA = pars
        else:
            w, omega_b,omega_cdm, h, logA = pars
            w0 = w
            wa = 0.
        Hzfid, chizfid = fid_dists
        speed_of_light = 2.99792458e5

        # omega_b = 0.02242
        
        As =  np.exp(logA)*1e-10#2.0830e-9
        ns = 0.9649

        nnu = 1
        nur = 2.0328
        # mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106

        # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2
        OmegaM = (omega_cdm + omega_b + omega_nu) / h**2

        pkparams = {
            'output': 'mPk',
            'P_k_max_h/Mpc': 20.,
            'z_pk': '0.0,10',
            'A_s': As,
            'n_s': ns,
            'h': h,
            'N_ur': nur,
            'N_ncdm': nnu,
            'omega_ncdm': omega_nu,
            # 'tau_reio': 0.0544,
            'z_reio': 7.,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            'Omega_Lambda': 0.,
            'w0_fld': w0,
            'wa_fld': wa}

        pkclass = Class()
        pkclass.set(pkparams)
        pkclass.compute()

        # Caluclate AP parameters
        Hz = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
        chiz = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
        apar, aperp = Hzfid / Hz, chiz / chizfid

        if ap_off:
            apar, aperp = 1.0, 1.0

        # Calculate growth rate
        # fnu = pkclass.Omega_nu / pkclass.Omega_m()
        # f   = f_of_a(1/(1.+z), OmegaM=OmegaM) * (1 - 0.6 * fnu)
        f = pkclass.scale_independent_growth_factor_f(z)

        # Calculate and renormalize power spectrum
        ki = np.logspace(-3.0,1.0,200)
        pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in ki] )
        # pi = (sigma8/pkclass.sigma8())**2 * pi

        # Now do the RSD
        modPT = LPT_RSD(ki, pi, kIR=0.2,use_Pzel = False,\
                    cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
        modPT.make_pltable(f, kv=self.kvec, apar=apar, aperp=aperp, ngauss=3)

        return modPT.p0ktable, modPT.p2ktable, modPT.p4ktable, pkclass.sigma8(), pkclass.scale_independent_growth_factor(z),f
    
    def compute_pell_tables_w0wacdm(self, pars, z, fid_dists= (None,None), ap_off=False ):

        w0,wa,ns, omega_b,omega_cdm, h, logA = pars
        Hzfid, chizfid = fid_dists
        speed_of_light = 2.99792458e5

        # omega_b = 0.02242
        # w0 = w
        # wa = 0.
        As =  np.exp(logA)*1e-10#2.0830e-9
        ns = 0.9649

        nnu = 1
        nur = 2.0328
        # mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106

        # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2
        OmegaM = (omega_cdm + omega_b + omega_nu) / h**2

        pkparams = {
            'output': 'mPk',
            'P_k_max_h/Mpc': 20.,
            'z_pk': '0.0,10',
            'A_s': As,
            'n_s': ns,
            'h': h,
            'N_ur': nur,
            'N_ncdm': nnu,
            'omega_ncdm': omega_nu,
            # 'm_ncdm': mnu,
            # 'tau_reio': 0.0568,
            'z_reio': 7.,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            'Omega_Lambda': 0.,
            'w0_fld': w0,
            'wa_fld': wa}

        pkclass = Class()
        pkclass.set(pkparams)
        pkclass.compute()

        # Caluclate AP parameters
        Hz = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
        chiz = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
        apar, aperp = Hzfid / Hz, chiz / chizfid

        if ap_off:
            apar, aperp = 1.0, 1.0

        # Calculate growth rate
        # fnu = pkclass.Omega_nu / pkclass.Omega_m()
        # f   = f_of_a(1/(1.+z), OmegaM=OmegaM) * (1 - 0.6 * fnu)
        f = pkclass.scale_independent_growth_factor_f(z)

        # Calculate and renormalize power spectrum
        ki = np.logspace(-3.0,1.0,200)
        pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in ki] )
        # pi = (sigma8/pkclass.sigma8())**2 * pi

        # Now do the RSD
        modPT = LPT_RSD(ki, pi, kIR=0.2,use_Pzel = False,\
                    cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
        modPT.make_pltable(f, kv=self.kvec, apar=apar, aperp=aperp, ngauss=3)

        return modPT.p0ktable, modPT.p2ktable, modPT.p4ktable, pkclass.sigma8(), pkclass.scale_independent_growth_factor(z),f
    
    def compute_pell_tables_SF(self,pars, z ):
    
        # OmegaM, h, sigma8 = pars
        # Hzfid, chizfid = fid_dists
        f_sig8,apar,aperp,m = pars
        
        pkclass = self.fid_class
        h = self.fid_class.h()

        sig8_z = pkclass.sigma(8,z,h_units=True)
        f = f_sig8 / sig8_z

        # Calculate and renormalize power spectrum
        ki = np.logspace(-3.0,1.0,200)
        pi = np.array( [pkclass.pk_cb(k*h, z ) * h**3 for k in ki] ) * np.exp( shapefit_factor(ki,m) )
        # pi = (sig8_z/pkclass.sigma8())**2 * pi

        # Now do the RSD
        modPT = LPT_RSD(ki, pi, kIR=0.2,use_Pzel = False,\
                    cutoff=10, extrap_min = -4, extrap_max = 3, N = 1000, threads=8, jn=5)
        modPT.make_pltable(f, kv=self.kvec, apar=apar, aperp=aperp, ngauss=3)

        return modPT.p0ktable, modPT.p2ktable, modPT.p4ktable
    
    def compute_fsigma_s8(self,pars,z):
        
        f_sig8,_,_,m = pars
        
        h_fid = self.fid_class.h()
        rd_fid = self.fid_class.rs_drag()
        
        return f_sig8*np.exp(m/(1.2) * np.tanh(0.6*np.log((rd_fid*h_fid)/(8.0*h_fid)) ))
    
    def compute_theta_star_w0waCDM(self,pars):
        
        
        w0,wa, omega_b,omega_cdm, h, logA = pars
        
        As =  np.exp(logA)*1e-10#2.0830e-9
        ns = 0.9649

        nnu = 1
        nur = 2.0328
        # mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106

        # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2
        OmegaM = (omega_cdm + omega_b + omega_nu) / h**2

        pkparams = {
            'A_s': As,
            'n_s': ns,
            'h': h,
            'N_ur': nur,
            'N_ncdm': nnu,
            'omega_ncdm': omega_nu,
            # 'tau_reio': 0.0544,
            'z_reio': 7.,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            'Omega_Lambda': 0.,
            'w0_fld': w0,
            'wa_fld': wa}

        pkclass = Class()
        pkclass.set(pkparams)
        pkclass.compute()

        return pkclass.theta_star_100()