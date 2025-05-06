import numpy as np
import json
from taylor_approximation import taylor_approximate
from scipy.interpolate import RectBivariateSpline

class Compute_cleft():
    '''Compute Pgg, Pgm, and Pmm using a Taylor series emulator'''

    def __init__(self, Pmm_filename, Pcleft_filenames,order,zint,zeffs):

        self.order = order
        # self.kout = kout
        self.zint = zint
    
        json_file = open(Pmm_filename, 'r')
        emu = json.load( json_file )
        json_file.close()

        self.derivs_Pmm = {}
        self.emu_zs = []
        # self.emu = emu
        for z in list(emu.keys()): 
            self.derivs_Pmm[z] = [np.array(ll) for ll in emu[z]['derivs']]
            self.emu_zs.append(float(z))
        self.ki = emu[z]['kvec']
        del emu

        self.derivs_gm = {}
        self.derivs_gg = {}
        for zeff,Pcleft_filename in zip(zeffs,Pcleft_filenames):
            zstr = "%.3f" %(zeff)
            json_file = open(Pcleft_filename, 'r')
            emu = json.load( json_file )
            json_file.close()
    
            self.x0s = emu['x0']
            self.derivs_gm[zstr] = [np.array(ll) for ll in emu['derivs_gm']]
            self.derivs_gg[zstr] = [np.array(ll) for ll in emu['derivs_gg']]
    
            del emu

    def compute_Pmm(self,pars,z):

        cpars = pars[:len(self.x0s)]
        Pmm_grid = np.zeros((len(self.ki),len(self.emu_zs)))
        for ii,zi in enumerate(self.emu_zs):
            Pmm = taylor_approximate(cpars, self.x0s, self.derivs_Pmm[f'{zi}'], order=self.order)
            Pmm_grid[:,ii] = Pmm
        # Create the interpolator (assuming P_kz is ordered as P(k, z))
        interpolator = RectBivariateSpline(self.ki, self.emu_zs, Pmm_grid)
        Pmm_interp = interpolator(self.ki,self.zint)
        res = [Pmm_interp[:,i] for i in range(len(self.zint))]
        return np.array([self.ki]+res).T

    def compute_Pgm(self,pars,z):
        zstr = "%.3f" %(z)
        cpars = pars[:len(self.x0s)]
        b1,b2,bs = pars[len(self.x0s):]
        b3 = 0
        ptab = taylor_approximate(cpars,self.x0s,self.derivs_gm[zstr],order = self.order)
        kout,za = ptab[:,0],ptab[:,-1]
        res      = np.zeros((len(kout),3))
        res[:,0] = kout
        bias_monomials = np.array([1., 0.5*b1, 0,\
                                  0.5*b2, 0, 0,\
                                  0.5*bs, 0, 0, 0,\
                                  0.5*b3, 0.])
        res[:,1] = np.sum(ptab[:,1:-1]*bias_monomials,axis=1) 
        res[:,2] = -0.5*kout**2 * za

        return res

    def compute_Pgg(self,pars,z):
        zstr = "%.3f" %(z)
        cpars = pars[:len(self.x0s)]
        b1,b2,bs = pars[len(self.x0s):]
        b3 = 0
        ptab = taylor_approximate(cpars,self.x0s,self.derivs_gg[zstr],order = self.order)
        kout,za = ptab[:,0],ptab[:,-1]
        res      = np.zeros((len(kout),3))
        res[:,0] = kout
        bias_monomials = np.array([1, b1, b1**2,\
                                       b2, b1*b2, b2**2,\
                                       bs, b1*bs, b2*bs, bs**2,\
                                       b3, b1*b3 ])

        res[:,1] = np.sum(ptab[:,1:-1]*bias_monomials,axis=1) 
        res[:,2] = -0.5*kout**2 * za

        return res
        

        

        
        
        

        

        