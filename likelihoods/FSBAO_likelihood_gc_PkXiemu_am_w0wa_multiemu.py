import numpy as np
import time
import json
import yaml

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.signal import argrelmax
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyval

from scipy.special import spherical_jn
# from scipy.integrate import simps

from taylor_approximation import taylor_approximate
# from linear_theory import f_of_a, D_of_a
# from compute_sigma8_class import Compute_Sigma8_wCDM
# from compute_pell_tables_direct import direct_fit_theory
# from compute_xiell_tables_recsym import compute_xiell_tables, compute_pkclass
from Compute_zParams_class import Compute_zParams
from aemulus_heft.heft_emu import NNHEFTEmulator

import scipy.integrate as integrate
from numpy import sqrt, sin, cos, pi

# Class to have a full-shape likelihood for a bunch of pieces of data from both galactic caps in the same z bin
# Currently assumes all data have the same fiducial cosmology etc.
# If not I suggest chaning the theory class so that instead of being labelled by "zstr" it gets labelled by sample name.
# And each sample name indexing the fiducial cosmology numbers (chi, Hz etc) in dictionaries. For another time...

class JointLikelihood(Likelihood):
    
    zfids: list
    photo_tmp: bool
    
    basedir: str
    
    fs_sample_names: list
    bao_sample_names: list
    
    # optimize: turn this on when optimizng/running the minimizer so that the Jacobian factor isn't included
    # include_priors: this decides whether the marginalized parameter priors are included (should be yes)
    linear_param_dict_fn: str
    optimize: bool
    include_priors: bool
    
    fs_datfns: list
    bao_datfns: list

    covfn: str
    template_fn: str
    
    fs_kmins: list
    fs_mmaxs: list
    fs_qmaxs: list
    fs_hmaxs: list
    # fs_matMfns: list
    fs_matWfns: list
    w_kin_fn: str
    hexa: bool
    bao_rmaxs: list
    bao_rmins: list
    bao_ells: list
    
    # npoly: int
    cov_fac: float
    invcov_fac: float
    jeff: bool

    def initialize(self):
        """Sets up the class."""
        
        # Redshift Label for theory classes
        # self.zstr = "%.2f" %(self.zfid)

        # Load the linear parameters of the theory model theta_a such that
        # P_th = P_{th,nl} + theta_a P^a for some templates P^a we will compute
        self.linear_param_dict = yaml.load(open(self.linear_param_dict_fn), Loader=yaml.SafeLoader)
        self.linear_param_means = {key: self.linear_param_dict[key]['mean'] for key in self.linear_param_dict.keys()}
        self.linear_param_stds  = np.array([self.linear_param_dict[key]['std'] for key in self.linear_param_dict.keys()])
        self.Nlin = len(self.linear_param_dict) 
        
        # broadband parameter names:
        self.mpoly_names = [ 'M%d'%(ii) for ii in range(2)]
        self.qpoly_names = [ 'Q%d'%(ii) for ii in range(2)]
        self.qspl_names = [ 'Q%dsp'%(ii) for ii in range(2)]
        
        # Places to store theory curves
        self.pconv = {}
        self.xith = {}
        
#         self.sp_kmax = {}
        
#         for ll, fs_sample_name in enumerate(self.fs_sample_names):
#             self.sp_kmax[fs_sample_name] = self.kmax_spline[ll]
        
        # Binning matrix for correlation function, one for each BAO sample
        self.binmat = dict((name, None) for name in self.bao_sample_names)
        
        rmin, rmax, dr = 50, 160, 0.5
        rvec = np.arange(rmin, rmax, dr)
        
        self.delta = 0.06
        self.B20 = np.array([self.B20_x(self.delta*r) for r in rvec])
        self.B21 = np.array([self.B21_x(self.delta*r) for r in rvec])
        
        self.loadData()
        
        
        #
    
    def Si_x(self,x):
        result = integrate.quad(lambda t: sin(t)/t, 0, x)
        return result[0]

    def B20_x(self,x):
        prefac = 2./x**6
        terms = x**3 * self.Si_x(x) - 2*x**3*self.Si_x(2*x)
        terms += x**2 * cos(x) - x**2*cos(2*x) - x*sin(x) - 16*cos(x)
        terms += 4*cos(2*x)+x*sin(x)*cos(x)+12
        return prefac*terms

    def B21_x(self,x):
        prefac = -1./(2*x**3)
        terms = 6*x**3*self.Si_x(x) - 32*x**3*self.Si_x(2*x) + 27*x**3*self.Si_x(3*x)
        terms += 8*x**2 + 6*x**2*cos(x) -16*x**2*cos(2*x) + 9*x**2*cos(3*x)
        terms += -6*x*sin(x) + 8*x*sin(2*x) -3*x*sin(3*x)
        terms += -96*cos(x) + 64*cos(2*x) - 16*cos(3*x) + 48
        return prefac*terms
    
    def get_requirements(self):
        
        # Here we will comment out all the linear parameters
        
        req = {'taylor_pk_ell_mod': None,\
               'taylor_xi_ell_mod': None,\
               'zPars': None,\
               'ns': None,\
               'H0_emu': None,\
               # 'sigma8': None,\
               # 'omega_b': None,\
               # 'omega_cdm': None,\
               'omegam': None,\
               'w': None,\
               'wa': None,\
                'logA': None}
        
        for fs_sample_name in self.fs_sample_names:
            req_bias = { \
                   'bsig8_' + fs_sample_name: None,\
                   # 'b1_' + fs_sample_name: None,\
                   'b2sig8_' + fs_sample_name: None,\
                   'bssig8_' + fs_sample_name: None,\
                   'b3sig8_' + fs_sample_name: None,\
                   #'alpha0_' + fs_sample_name: None,\
                   #'alpha2_' + fs_sample_name: None,\
                   #'SN0_' + fs_sample_name: None,\
                   #'SN2_' + fs_sample_name: None\
                   }
            req = {**req, **req_bias}
        
        for bao_sample_name in self.bao_sample_names:
            req_bao = {\
                   'B1_' + bao_sample_name: None,\
                   'F_' +  bao_sample_name: None,\
                   # 'Sigpar_' +  bao_sample_name: None,\
                   # 'Sigperp_' +  bao_sample_name: None,\
                   # 'Sigs_' +  bao_sample_name: None,\
                   #'M0_' + bao_sample_name: None,\
                   #'M1_' + bao_sample_name: None,\
                   #'M2_' + bao_sample_name: None,\
                   #'Q0_' + bao_sample_name: None,\
                   #'Q1_' + bao_sample_name: None,\
                   #'Q2_' + bao_sample_name: None,\
                    }
            req = {**req, **req_bao}
            
        return(req)
    
    def full_predict(self, thetas=None):
        
        thy_obs = []

        if thetas is None:
            thetas = self.linear_param_means
        
        for zfid,fs_sample_name in zip(self.zfids,self.fs_sample_names):
            fs_thy  = self.fs_predict(fs_sample_name,zfid,thetas=thetas)
            fs_obs  = self.fs_observe(fs_thy, fs_sample_name,thetas=thetas)
            thy_obs = np.concatenate( (thy_obs,fs_obs) )
            
            if fs_sample_name in self.bao_sample_names:
                bao_thy = self.bao_predict(fs_sample_name,zfid,thetas=thetas)
                bao_obs = self.bao_observe(bao_thy,fs_sample_name)
                thy_obs = np.concatenate( (thy_obs, bao_obs) )
        
        # for bao_sample_name in self.bao_sample_names:
        #     bao_thy = self.bao_predict(bao_sample_name,thetas=thetas)
        #     bao_obs = self.bao_observe(bao_thy,bao_sample_name)
        #     thy_obs = np.concatenate( (thy_obs, bao_obs) )
            
        return thy_obs
    
    def logp(self,**params_values):
        """Return a log-likelihood."""
        
        # Compute the theory prediction with lin. params. at prior mean
        #t1 = time.time()
        thy_obs_0 = self.full_predict()
        self.Delta = self.dd - thy_obs_0
        #t2 = time.time()
        
        # Now compute template
        self.templates = []
        for param in self.linear_param_dict.keys():
            thetas = self.linear_param_means.copy()
            thetas[param] += 1.0
            self.templates += [ self.full_predict(thetas=thetas) - thy_obs_0 ]
        
        self.templates = np.array(self.templates)
        #t3 = time.time()
        
        # Make dot products
        self.Va = np.dot(np.dot(self.templates, self.cinv), self.Delta)
        self.Lab = np.dot(np.dot(self.templates, self.cinv), self.templates.T) + self.include_priors * np.diag(1./self.linear_param_stds**2)
        #self.Va = np.einsum('ij,jk,k', self.templates, self.cinv, self.Delta)
        #self.Lab = np.einsum('ij,jk,lk', self.templates, self.cinv, self.templates) + np.diag(1./self.linear_param_stds**2)
        self.Lab_inv = np.linalg.inv(self.Lab)
        #t4 = time.time()
        
        # Compute the modified chi2
        lnL  = -0.5 * np.dot(self.Delta,np.dot(self.cinv,self.Delta)) # this is the "bare" lnL
        lnL +=  0.5 * np.dot(self.Va, np.dot(self.Lab_inv, self.Va)) # improvement in chi2 due to changing linear params
        self.detFish = 0.5 * np.log( np.linalg.det(self.Lab) )
        if not self.optimize:
            if self.jeff:
                lnL += 0.5 * self.Nlin * np.log(2*np.pi)
            else:
                lnL += - 0.5 * np.log( np.linalg.det(self.Lab) ) + 0.5 * self.Nlin * np.log(2*np.pi) # volume factor from the determinant
        
        #t5 = time.time()
        
        #print(t2-t1, t3-t2, t4-t3, t5-t4)
        
        return lnL
        #
        
    def get_best_fit(self):
        try:
            self.p0_nl  = self.dd - self.Delta
            self.bf_thetas = np.einsum('ij,j', np.linalg.inv(self.Lab), self.Va)
            self.p0_lin = np.einsum('i,il', self.bf_thetas, self.templates)
            return self.p0_nl + self.p0_lin
        except:
            print("Make sure to first compute the posterior.")
        
    def loadData(self):
        """
        Loads the required data.
        
        Do this in two steps... first load full shape data then xirecon, concatenate after.
        
        The covariance is assumed to already be joint in the concatenated format.
        
        """
        # First load the data
        
        self.kdats = {}
        self.p0dats = {}
        self.p2dats = {}
        self.p4dats = {}
        self.fitiis = {}
        
        for ii, fs_datfn in enumerate(self.fs_datfns):
            fs_sample_name = self.fs_sample_names[ii]
            fs_dat = np.loadtxt(self.basedir+fs_datfn)
            self.kdats[fs_sample_name] = fs_dat[:,0]
            self.p0dats[fs_sample_name] = fs_dat[:,1]
            self.p2dats[fs_sample_name] = fs_dat[:,2]
            try:
                self.p4dats[fs_sample_name] = fs_dat[:,3]
                hex_dat = True
            except:
                print('No hexadecapole data found')
                hex_dat = False
            
            # Make a list of indices for the monopole and quadrupole only in Fourier space
            # This is specified to each sample in case the k's are different.
            yeses = self.kdats[fs_sample_name] > 0
            nos   = self.kdats[fs_sample_name] < 0
            if self.hexa:
                self.fitiis[fs_sample_name] = np.concatenate( (yeses, yeses, yeses ) )
            else:
                self.fitiis[fs_sample_name] = np.concatenate( (yeses,yeses, nos ) )
        
        self.rdats = {}
        self.xi0dats = {}
        self.xi2dats = {}
        
        for ii, bao_datfn in enumerate(self.bao_datfns):
            bao_sample_name = self.bao_sample_names[ii]
            bao_dat = np.loadtxt(self.basedir+bao_datfn)
            self.rdats[bao_sample_name] = bao_dat[:,0]
            self.xi0dats[bao_sample_name] = bao_dat[:,1]
            if len(self.bao_ells) ==2:
                self.xi2dats[bao_sample_name] = bao_dat[:,2]
        
        # Join the data vectors together
        self.dd = []
        
        for fs_sample_name in self.fs_sample_names:
            if self.hexa:
                self.dd = np.concatenate( (self.dd, self.p0dats[fs_sample_name], self.p2dats[fs_sample_name], self.p4dats[fs_sample_name]) )
            else:
                self.dd = np.concatenate( (self.dd, self.p0dats[fs_sample_name], self.p2dats[fs_sample_name]) )
            
            
            if fs_sample_name in self.bao_sample_names:
                self.dd = np.concatenate( (self.dd, self.xi0dats[fs_sample_name]) )
                if len(self.bao_ells) >1:
                    self.dd = np.concatenate( (self.dd, self.xi2dats[fs_sample_name]) )
        
        # We're only going to want some of the entries in computing chi^2.

        # Now load the covariance matrix.
        cov = np.loadtxt(self.basedir+self.covfn)/self.cov_fac
        
        # We're only going to want some of the entries in computing chi^2.
        
        # this is going to tell us how many indices to skip to get to the nth multipole
        startii = 0
        
        for ss, fs_sample_name in enumerate(self.fs_sample_names):
            
            kcut = (self.kdats[fs_sample_name] > self.fs_mmaxs[ss])\
                          | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:     # FS Monopole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
            
            kcut = (self.kdats[fs_sample_name] > self.fs_qmaxs[ss])\
                       | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:       # FS Quadrupole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size

            if self.hexa:
                kcut = (self.kdats[fs_sample_name] > self.fs_hmaxs[ss])\
                           | (self.kdats[fs_sample_name] < self.fs_kmins[ss])

                for i in np.nonzero(kcut)[0]:       # FS Hexadecapole.
                    ii = i + startii
                    cov[ii, :] = 0
                    cov[ :,ii] = 0
                    cov[ii,ii] = 1e25

                startii += self.kdats[fs_sample_name].size
        
        for ss, bao_sample_name in enumerate(self.bao_sample_names):
            
            rcut = (self.rdats[bao_sample_name] < self.bao_rmins[ss])\
                              | (self.rdats[bao_sample_name] > self.bao_rmaxs[ss])
            
            for i in np.nonzero(rcut)[0]:
                ii = i + startii
                cov[ii,:] = 0
                cov[:,ii] = 0
                cov[ii,ii] = 1e25
                
            startii += self.rdats[bao_sample_name].size

            if len(self.bao_ells)>1:
            
                for i in np.nonzero(rcut)[0]:
                    ii = i + startii
                    cov[ii,:] = 0
                    cov[:,ii] = 0
                    cov[ii,ii] = 1e25
                
                startii += self.rdats[bao_sample_name].size
        
        
        
        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)*self.invcov_fac
        #print(self.sample_name, np.diag(self.cinv)[:10])
        
        #load templates for rotation:
        self.rot_templates = {}
        # for ii, fs_sample_name in enumerate(self.fs_sample_names):
        tmpfile = np.loadtxt(self.basedir + self.template_fn)
        self.rot_templates['mono'] = tmpfile[:,0]
        self.rot_templates['quad'] = tmpfile[:,1]
        if self.photo_tmp:
            self.photo_template = tmpfile[:,2]
        # print(self.rot_templates['mono'],self.rot_templates['quad'])
        
        
        # Finally load the window function matrix.
        self.matWs = {}
        for ii, fs_sample_name in enumerate(self.fs_sample_names):
            self.matWs[fs_sample_name] = np.loadtxt(self.basedir+self.fs_matWfns[ii])
        
        self.w_kin = np.loadtxt(self.basedir+self.w_kin_fn)
        
    def combine_bias_terms_pkell(self,bvec, p0ktable, p2ktable, p4ktable):
        '''
        Same as function above but for the multipoles.
        
        Returns k, p0, p2, p4, assuming AP parameters from input p{ell}ktable
        '''
    
        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bvec

        bias_monomials = np.array([1, b1, b1**2,\
                                   b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3,\
                                   alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])

        p0 = np.sum(p0ktable * bias_monomials,axis=1)
        p2 = np.sum(p2ktable * bias_monomials,axis=1)
        p4 = np.sum(p4ktable * bias_monomials,axis=1)
        
        return p0, p2, p4
    
    def fs_predict(self, fs_sample_name,zfid, thetas=None):
        """Use the PT model to compute P_ell, given biases etc."""
        
        pp   = self.provider
        # zfid = self.zfid
        zstr = "%.2f" %(zfid)
        taylorPTs = pp.get_result('taylor_pk_ell_mod')
        kv, p0ktable, p2ktable, p4ktable = taylorPTs[zstr]

        #
        # sig8 = pp.get_param('sigma8')
        Om = pp.get_param('omegam')
        
        
        zPars = pp.get_result('zPars')
        sig8_z = zPars[zstr][0]
        f_z = zPars[zstr][1]
        #sig8 = pp.get_result('sigma8')
        b1   = pp.get_param('bsig8_' + fs_sample_name)/sig8_z - 1
        # b1   = pp.get_param('b1_' + fs_sample_name)
        b2   = pp.get_param('b2sig8_' + fs_sample_name)/(sig8_z**2)
        bs   = pp.get_param('bssig8_' + fs_sample_name)/(sig8_z**2)
        b3   = pp.get_param('b3sig8_' + fs_sample_name)/(sig8_z**3)
        
        # Instead of calling the linear parameters directly we will now analytically marginalize over them
        
        if thetas is None:
            alp0_tilde = self.linear_param_means['alpha0_' + fs_sample_name]
            alp2_tilde = self.linear_param_means['alpha2_' + fs_sample_name]
            sn0 = self.linear_param_means['SN0_' + fs_sample_name]
            sn2 = self.linear_param_means['SN2_' + fs_sample_name]
            if self.hexa:
                alp4_tilde = self.linear_param_means['alpha4_' + fs_sample_name]
                sn4 = self.linear_param_means['SN4_' + fs_sample_name]
            else: alp4_tilde,sn4 = 0.,0.
        else:
            alp0_tilde = thetas['alpha0_' + fs_sample_name]
            alp2_tilde = thetas['alpha2_' + fs_sample_name]
            sn0 = thetas['SN0_' + fs_sample_name]
            sn2 = thetas['SN2_' + fs_sample_name]
            
            if self.hexa:
                alp4_tilde = thetas['alpha4_' + fs_sample_name]
                sn4 = thetas['SN4_' + fs_sample_name]
            else: alp4_tilde,sn4 = 0.,0.
            
        alp0 = (1+b1)**2 * alp0_tilde
        alp2 = f_z*(1+b1)*(alp0_tilde+alp2_tilde)
        alp4 = f_z*(f_z*alp2_tilde+(1+b1)*alp4_tilde)
        alp6 = f_z**2*alp4_tilde
 
        bias = [b1, b2, bs, b3]
        cterm = [alp0,alp2,alp4,alp6]
        stoch = [sn0, sn2, sn4]
        bvec = bias + cterm + stoch
        self.bvec = bvec
        #print(self.zstr, b1, sig8)
        
        p0, p2, p4 = self.combine_bias_terms_pkell(bvec, p0ktable, p2ktable, p4ktable)
        
        #np.savetxt('pells_' + self.zstr + '_' + self.sample_name + '.txt',[kv,p0,p2,p4])
        
        # Put a point at k=0 to anchor the low-k part of the Spline.
        kv,p0 = np.append([0.,],kv),np.append([0.0,],p0)
        p2 = np.append([0.,],p2)
        p4 = np.append([0.0,],p4)
        tt    = np.array([kv,p0,p2,p4]).T
        
        if np.any(np.isnan(tt)):
            H0 = pp.get_param('H0_emu')
            print("NaN's encountered. Parameter values are: ", str(w0,wa,Om,H0))
        
        return(tt)
        #
    
    def bao_predict(self, bao_sample_name,zfid, thetas=None):
        
        pp   = self.provider
        zstr = "%.2f" %(zfid)
        # delta = 0.06
        
        B1   = pp.get_param('B1_' + bao_sample_name)
        F   = pp.get_param('F_' + bao_sample_name)
        
        # Analytically marginalize linear parmaeters so these are obtained differently
        if thetas is None:
            Mpoly = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in self.mpoly_names]
            Qpoly = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in self.qpoly_names]
            Qspl = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in self.qspl_names]
        else:
            # Mpoly = [thetas[param_name + '_' + bao_sample_name] for param_name in ['M0','M1',]]
            # Qpoly = [thetas[param_name + '_' + bao_sample_name] for param_name in ['Q0','Q1',]]
            Mpoly = [thetas[param_name + '_' + bao_sample_name] for param_name in self.mpoly_names]
            Qpoly = [thetas[param_name + '_' + bao_sample_name] for param_name in self.qpoly_names]
            Qspl = [thetas[param_name + '_' + bao_sample_name] for param_name in self.qspl_names]
        #M0, M1, M2 = [pp.get_param(param_name + '_' + bao_sample_name) for param_name in ['M0','M1','M2']]
        #Q0, Q1, Q2 = [pp.get_param(param_name + '_' + bao_sample_name) for param_name in ['Q0','Q1','Q2']]
        
        taylorPTs = pp.get_result('taylor_xi_ell_mod')
        rvec, xi0table, xi2table = taylorPTs[zstr]
        
        xi0t = xi0table[:,0] + B1*xi0table[:,1] + F*xi0table[:,2] \
             + B1**2 * xi0table[:,3] + F**2 * xi0table[:,4] + B1*F*xi0table[:,5]
        
        xi2t = xi2table[:,0] + B1*xi2table[:,1] + F*xi2table[:,2] \
             + B1**2 * xi2table[:,3] + F**2 * xi2table[:,4] + B1*F*xi2table[:,5]
        
        xi0t += Mpoly[0] + Mpoly[1]*(rvec*self.fs_kmins[0]/(2*pi))**2  #polyval(rvec*np.pi/, Mpoly)
        xi2t += Qpoly[0] + Qpoly[1]*(rvec*self.fs_kmins[0]/(2*pi))**2
        xi2t += self.delta**3*(Qspl[0]*self.B20 + Qspl[1]*self.B21)
        
        
        return np.array([rvec,xi0t,xi2t]).T
    

        
    def fs_observe(self,tt,fs_sample_name,thetas=None):
        """Apply the window function matrix to get the binned prediction."""
        
        
        # print(fs_sample_name)
        # Have to stack ell=0, 2 & 4 in bins of 0.001h/Mpc from 0-0.4h/Mpc.
        # kv  = np.linspace(0.0,0.4,400,endpoint=False) + 0.0005
        # maxk = self.sp_kmax[fs_sample_name]
        # kv  = np.linspace(0.0,maxk,int(maxk/0.001),endpoint=False) + 0.0005
        kv  = self.w_kin
        thy =                     Spline(tt[:,0],tt[:,1],ext=3)(kv)
        # thy = np.concatenate([thy,Spline(tt[:,0],tt[:,1],ext=3)(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        # thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3],ext=3)(kv)])
        
        if np.any(np.isnan(thy)) or np.max(thy) > 1e8:
            hub = self.provider.get_param('H0_emu') / 100.
            sig8 = self.provider.get_param('sig8')
            ns = self.provider.get_param('ns')
            Om = self.provider.get_param('omegam')
            print("NaN's encountered in PREDICT. Parameter values are: ns={},H0={},Om={},sig8={}".format(ns,hub,Om,sig8))
        
        # wide angle
        # expanded_model = np.matmul(self.matMs[fs_sample_name], thy )
        # Convolve with window (true) −> (conv) see eq. 2.18
        # Multiply by ad-hoc factor
        # convolved_model = np.matmul(self.matWs[fs_sample_name], expanded_model )
        convolved_model = np.matmul(self.matWs[fs_sample_name], thy )
        
        # keep only the monopole and quadrupole
        # convolved_model = convolved_model[self.fitiis[fs_sample_name]]
        
        #Marginalize over template parameters for rotation
        if thetas is None:
            s0 = self.linear_param_means['s0_' + fs_sample_name]
            s2 = self.linear_param_means['s2_' + fs_sample_name]
            if self.photo_tmp:
                ph = self.linear_param_means['ph_' + fs_sample_name]
                tmp_ph = self.photo_template
            else:
                ph = 0
                tmp_ph = np.zeros(len(convolved_model))
        else:
            s0 = thetas['s0_' + fs_sample_name]
            s2 = thetas['s2_' + fs_sample_name]
            if self.photo_tmp:
                ph = thetas['ph_' + fs_sample_name]
                tmp_ph = self.photo_template
            else:
                ph = 0
                tmp_ph = np.zeros(len(convolved_model))
            
        tmp0 = self.rot_templates['mono']
        tmp2 = self.rot_templates['quad']
        
        convolved_model += s0*tmp0 + s2*tmp2 + ph*tmp_ph
        
        # Save the model:
        self.pconv[fs_sample_name] = convolved_model
    
        return convolved_model
    
    def bao_observe(self, tt, bao_sample_name, matrix=True):
        '''
        Bin the BAO results... probabaly should eventually use a matrix.
        '''
        
        rdat = self.rdats[bao_sample_name]
        
        if matrix:
            # If no binning matrix for this sample yet, make it.
            if self.binmat[bao_sample_name] is None:  
                
                dr = rdat[1] - rdat[0]
                
                rth = tt[:,0]
                Nvec = len(rth)

                bin_mat = np.zeros( (len(rdat), Nvec) )

                for ii in range(Nvec):
                    # Define basis vector
                    xivec = np.zeros_like(rth); xivec[ii] = 1
    
                    # Define the spline:
                    thy = Spline(rth, xivec, ext='const')
    
                    # Now compute binned basis vector:
                    tmp = np.zeros_like(rdat)
    
                    for i in range(rdat.size):
                        kl = rdat[i]-dr/2
                        kr = rdat[i]+dr/2

                        ss = np.linspace(kl, kr, 100)
                        p     = thy(ss)
                        tmp[i]= np.trapz(ss**2*p,x=ss)*3/(kr**3-kl**3)
        
                    bin_mat[:,ii] = tmp
                
                self.binmat[bao_sample_name] = np.array(bin_mat)
            
            tmp0 = np.dot(self.binmat[bao_sample_name], tt[:,1])
            tmp2 = np.dot(self.binmat[bao_sample_name], tt[:,2])
        
        else:
            thy0 = Spline(tt[:,0],tt[:,1],ext='extrapolate')
            thy2 = Spline(tt[:,0],tt[:,2],ext='extrapolate')
            #thy4 = Spline(tt[:,0],tt[:,3],ext='extrapolate')
        
            dr   = rdat[1]- rdat[0]
        
            tmp0 = np.zeros_like(rdat)
            tmp2 = np.zeros_like(rdat)
        
            for i in range(rdat.size):
            
                kl = rdat[i]-dr/2
                kr = rdat[i]+dr/2

                ss = np.linspace(kl, kr, 100)
                p0     = thy0(ss)
                tmp0[i]= np.trapz(ss**2*p0,x=ss)*3/(kr**3-kl**3)
                p2     = thy2(ss)
                tmp2[i]= np.trapz(ss**2*p2,x=ss)*3/(kr**3-kl**3)
                #p4     = thy4(ss)
                #tmp4[i]= np.trapz(ss**2*p4,x=ss)*3/(kr**3-kl**3)
            
        #self.xith[bao_sample_name] = np.concatenate((tmp0,tmp2))
        if len(self.bao_ells) == 2:
            return np.concatenate((tmp0,tmp2))
        else:
            return tmp0
    

class Taylor_pk_theory_zs(Theory):
    """
    A class to return a set of derivatives for the Taylor series of Pkell.
    """
    zfids: list
    pk_filenames: list
    xi_filenames: list
    Rsmooth: list
    s8_filenames: list
    # plin_filenames: list
    basedir: str
    omega_nu: float
    bao_sample_names: list
    # derive_AP: bool
    # AP_z: float
    
    def initialize(self):
        """Sets up the class by loading the derivative matrices."""
        
        print("Loading Taylor series.")
        
        # First Load Sigma8 class:
        # self.compute_sigma8 = Compute_Sigma8_wCDM(self.basedir + self.s8_filename)
        
        # self.compute_theory = direct_fit_theory()
        self.nnemu = NNHEFTEmulator() #NN emulator to be used to get sigma8(z)
        
        # Then Load Clustering
        self.taylors_pk = {}
        self.s8_emus = {} #only need this for f(z)
        self.taylors_xi = {}
        
        for jj in range(3):
            self.taylors_pk[f'{jj}'] = {}
            self.s8_emus[f'{jj}'] = {}
            self.taylors_xi[f'{jj}'] = {}
            for ii in range(len(self.zfids)):
                zfid = self.zfids[ii]
                zstr = "%.2f"%(zfid)
                taylors_pk = {}
                taylors_xi = {}

                pk_filename = self.pk_filenames[ii][jj]
                
                # Load the power spectrum derivatives
                json_file = open(self.basedir+pk_filename, 'r')
                emu = json.load( json_file )
                json_file.close()
                
                x0s = emu['x0']
                kvec = emu['kvec']
                derivs_p0 = [np.array(ll) for ll in emu['derivs0']]
                derivs_p2 = [np.array(ll) for ll in emu['derivs2']]
                derivs_p4 = [np.array(ll) for ll in emu['derivs4']]
                
                taylors_pk['x0'] = np.array(x0s)
                taylors_pk['kvec'] = np.array(kvec)
                taylors_pk['derivs_p0'] = derivs_p0
                taylors_pk['derivs_p2'] = derivs_p2
                taylors_pk['derivs_p4'] = derivs_p4
    
                
                # Load the correlation function derivatives
                xi_filename = self.xi_filenames[ii][jj]
                
                json_file = open(self.basedir+xi_filename, 'r')
                emu = json.load( json_file )
                json_file.close()
                
                x0s = emu['x0']
                rvec = emu['rvec']
                derivs_x0 = [np.array(ll) for ll in emu['derivs0']]
                derivs_x2 = [np.array(ll) for ll in emu['derivs2']]
                
                taylors_xi['x0'] = np.array(x0s)
                taylors_xi['rvec'] = np.array(rvec)
                taylors_xi['derivs_xi0'] = derivs_x0
                taylors_xi['derivs_xi2'] = derivs_x2
    
    
                self.taylors_pk[f'{jj}'][zstr] = taylors_pk
                # self.fid_dists[zstr] =  self.compute_theory.get_fid_dists(zfid)
                self.taylors_xi[f'{jj}'][zstr] = taylors_xi
    
                s8_filename = self.s8_filenames[ii][jj]
                self.s8_emus[f'{jj}'][zstr] = Compute_zParams(self.basedir + s8_filename)
                print(pk_filename)
                print(xi_filename)
                print(s8_filename)
                del emu
    
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        zmax = max(self.zfids)
        zg  = np.linspace(0,zmax,100,endpoint=True)
        # Don't need sigma8_z, fsigma8 or radial distance
        # here, but want them up in likelihood and they
        # only depend on cosmological things (not biases).
        #
        req = {\
               'ns': None,\
               # 'omega_b': None,\
               # 'omega_cdm': None,\
               'omegam': None,\
               # 'H0': None,\
               'w': None,\
               'wa': None,\
               'logA': None,\
               'm_ncdm': None,\
              }
        
        for bao_sample_name in self.bao_sample_names:
            req_bao = {\
                   'Sigpar_' +  bao_sample_name: None,\
                   'Sigperp_' +  bao_sample_name: None,\
                   'Sigs_' +  bao_sample_name: None,\
                    }
            req = {**req, **req_bao}
        
        return(req)
    def get_can_provide(self):
        """What do we provide: a Taylor series class for pkells."""
        return ['taylor_pk_ell_mod','taylor_xi_ell_mod','zPars']
    
    def get_can_provide_params(self):
        return ['sig8','H0_emu']
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Just load up the derivatives and things.
        """
        pp = self.provider
        
        w0 = pp.get_param('w')
        wa = pp.get_param('wa')
        # ns = pp.get_param('ns')
        theta_star = 1.0411
        # ns = 0.96
        # H0 = pp.get_param('H0')
        # hub =  H0/ 100.
        logA = pp.get_param('logA')
        As = 1e-10 * np.exp(logA)
        # omega_b = pp.get_param('omega_b')
        # omega_cdm = pp.get_param('omega_cdm')
        OmM = pp.get_param('omegam')
        Mnu = pp.get_param('m_ncdm')
        cosmopars = [w0,wa, OmM, logA]

        if (wa > -0.5):
            emu_ind = '0'
        elif (wa <= -0.5) & (wa > -1.0):
            emu_ind = '1'
        elif (wa <= -1.0):
            emu_ind = '2'

        #For NN emulator:
        # params = np.array([omega_b,omega_cdm,-1.0,ns, As,H0,np.log10(Mnu),0])[self.nnemu.param_order] 
        # sig80 = self.nnemu.sigma8z_emu(params)[0]
        ptables = {}
        xitables = {}
        zPars = {}
            
        # ll=0
        # for zfid,R in zip(self.zfids,self.Rsmooth):
        for ii,zfid in enumerate(self.zfids):
            zstr = "%.2f" %(zfid)
            
            # Load pktables
            x0s = self.taylors_pk[emu_ind][zstr]['x0']
            derivs0 = self.taylors_pk[emu_ind][zstr]['derivs_p0']
            derivs2 = self.taylors_pk[emu_ind][zstr]['derivs_p2']
            derivs4 = self.taylors_pk[emu_ind][zstr]['derivs_p4']
            
            kv = self.taylors_pk[emu_ind][zstr]['kvec']
            p0ktable = taylor_approximate(cosmopars, x0s, derivs0, order=4)
            p2ktable = taylor_approximate(cosmopars, x0s, derivs2, order=4)
            p4ktable = taylor_approximate(cosmopars, x0s, derivs4, order=4)
            
            ptables[zstr] = (kv, p0ktable, p2ktable, p4ktable)
            compute_zpars = self.s8_emus[emu_ind][zstr]
            fz_emu = compute_zpars.compute_fz(cosmopars,order = 5)
            sig8_emu = compute_zpars.compute_sig8(cosmopars,order = 5)
            Dz_emu = compute_zpars.compute_Dz(cosmopars,order = 5)
            sig8z = sig8_emu*Dz_emu
            # params = np.array([omega_b,omega_cdm,-1.0,ns, As,H0,np.log10(Mnu),zfid])[self.nnemu.param_order] 
            # sig8z = self.nnemu.sigma8z_emu(params)[0]
            # compute_zpars = self.s8_emus[zstr]
            # fz_emu = compute_zpars.compute_fz(cosmopars,order = 5)
            
            zPars[zstr] = [sig8z,fz_emu]
            
            
            sigmas = (pp.get_param('Sigpar_' +  self.bao_sample_names[ii]),\
                     pp.get_param('Sigperp_' +  self.bao_sample_names[ii]),\
                     pp.get_param('Sigs_' +  self.bao_sample_names[ii]))
            
            # xipars = 

            # Load xitables
            x0s = self.taylors_xi[emu_ind][zstr]['x0']
            derivs_xi0 = self.taylors_xi[emu_ind][zstr]['derivs_xi0']
            derivs_xi2 = self.taylors_xi[emu_ind][zstr]['derivs_xi2']
            
            rv = self.taylors_xi[emu_ind][zstr]['rvec']
            xi0table = taylor_approximate(cosmopars, x0s, derivs_xi0, order=3)
            xi2table = taylor_approximate(cosmopars, x0s, derivs_xi2, order=3)
            
            xitables[zstr] = (rv, xi0table, xi2table)
        
            # ll += 1
            
        #state['sigma8'] = sig8
        H0_emu = compute_zpars.compute_H0(cosmopars,order = 5)*100 
        state['derived'] = {'sig8': sig8_emu,'H0_emu': H0_emu}
        state['zPars'] = zPars
        state['taylor_pk_ell_mod'] = ptables
        state['taylor_xi_ell_mod'] = xitables