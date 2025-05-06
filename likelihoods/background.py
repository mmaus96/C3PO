# This is a wrapper around various codes for computing background 
# quantities (OmM, chistar, Ez, chi) relevant for Limber integrals.
# Each method should have thy_args and zs as its first two arguments.
#
# Currently wrapped:
# - background from CLASS

# ingredients
import numpy as np
from classy import Class

def classyBackground(thy_args, zs):
   """
   Computes background quantities relevant for Limber integrals
   using CLASS. Returns OmM (~0.3), chistar (comoving dist [h/Mpc] 
   to the surface of last scatter), Ez (H(z)/H0 evaluated on zs), 
   and chi (comoving distance [h/Mpc] evaluated on zs).
   
   Parameters
   ----------
   thy_args: list or ndarray
      omb,omc,ns,ln10As,H0,Mnu = thy_args[:6]
   zs: list OR ndarray
      redshifts to evaluate chi(z) and E(z) 
   """
   omb,omc,ns,ln10As,H0,Mnu = thy_args[:6]
             
   params = {'A_s': 1e-10*np.exp(ln10As),'n_s': ns,'h': H0/100., 
             'N_ur': 2.0328,'N_ncdm': 1,'m_ncdm': Mnu,'tau_reio': 0.0568,
             'omega_b': omb,'omega_cdm': omc}
   
   cosmo = Class()
   cosmo.set(params)
   cosmo.compute()
   
   OmM     = cosmo.Omega0_m()
   zstar   = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
   chistar = cosmo.comoving_distance(zstar)*cosmo.h()
   Ez      = np.vectorize(cosmo.Hubble)(zs)/cosmo.Hubble(0.)
   chi     = np.vectorize(cosmo.comoving_distance)(zs)*cosmo.h()
   
   return OmM,chistar,Ez,chi

def classyBackground_taylor(thy_args, zs):
   """
   Computes background quantities relevant for Limber integrals
   using CLASS. Returns OmM (~0.3), chistar (comoving dist [h/Mpc] 
   to the surface of last scatter), Ez (H(z)/H0 evaluated on zs), 
   and chi (comoving distance [h/Mpc] evaluated on zs).
   
   Parameters
   ----------
   thy_args: list or ndarray
      ns,omb,omc,h,ln10As = thy_args[:5]
   zs: list OR ndarray
      redshifts to evaluate chi(z) and E(z) 
   """
   ns,omb,omc,h,ln10As = thy_args[:5]
   Mnu = 0.06
             
   params = {'A_s': 1e-10*np.exp(ln10As),'n_s': ns,'h': h, 
             'N_ur': 2.0328,'N_ncdm': 1,'m_ncdm': Mnu,'tau_reio': 0.0568,
             'omega_b': omb,'omega_cdm': omc}
   
   cosmo = Class()
   cosmo.set(params)
   cosmo.compute()
   
   OmM     = cosmo.Omega0_m()
   zstar   = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
   chistar = cosmo.comoving_distance(zstar)*cosmo.h()
   Ez      = np.vectorize(cosmo.Hubble)(zs)/cosmo.Hubble(0.)
   chi     = np.vectorize(cosmo.comoving_distance)(zs)*cosmo.h()
   
   return OmM,chistar,Ez,chi

def classyBackground_w0wa(thy_args, zs):
   """
   Computes background quantities relevant for Limber integrals
   using CLASS. Returns OmM (~0.3), chistar (comoving dist [h/Mpc] 
   to the surface of last scatter), Ez (H(z)/H0 evaluated on zs), 
   and chi (comoving distance [h/Mpc] evaluated on zs).
   
   Parameters
   ----------
   thy_args: list or ndarray
      ns,omb,omc,h,ln10As = thy_args[:5]
   zs: list OR ndarray
      redshifts to evaluate chi(z) and E(z) 
   """
   w0,wa,Om,ln10As = thy_args[:4]
   theta_star = 1.0411
   Mnu = 0.06
   ns = 0.9649
   omega_b = 0.02237
             
   params = {'A_s': 1e-10*np.exp(ln10As),'n_s': ns,'100*theta_s':1.0411, 
             'N_ur': 2.0328,'N_ncdm': 1,'m_ncdm': Mnu,'tau_reio': 0.0568,
             'Omega_Lambda': 0.,'w0_fld': w0,'wa_fld': wa,'Omega_m': Om,'omega_b': omega_b}
   
   cosmo = Class()
   cosmo.set(params)
   cosmo.compute()
   
   OmM     = cosmo.Omega0_m()
   zstar   = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
   chistar = cosmo.comoving_distance(zstar)*cosmo.h()
   Ez      = np.vectorize(cosmo.Hubble)(zs)/cosmo.Hubble(0.)
   chi     = np.vectorize(cosmo.comoving_distance)(zs)*cosmo.h()
   
   return OmM,chistar,Ez,chi