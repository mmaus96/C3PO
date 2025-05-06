import numpy as np
import time
import json
import yaml
# import psutil

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.signal import argrelmax
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyval

from scipy.special import spherical_jn
# from scipy.integrate import simpson as simps

from likelihoods.pkCodes import pmmHEFT,pgmHEFT,pggHEFT
from likelihoods.background import classyBackground
from likelihoods.limber               import limb 
from likelihoods.pack_data_v2 import *
from aemulus_heft.heft_emu import NNHEFTEmulator

# from make_pkclass_aba import make_pkclass
from copy import deepcopy

# Class to have a shape-fit likelihood for a bunch of pieces of data from both galactic caps in the same z bin
# Currently assumes all data have the same fiducial cosmology etc.
# If not I suggest changing the theory class so that instead of being labelled by "zstr" it gets labelled by sample name.
# And each sample name indexing the fiducial cosmology numbers (chi, Hz etc) in dictionaries. For another time...

class CellLikelihood(Likelihood):
    
    zfids: list
    #sig8: float
    
    basedir: str
    
    gal_sample_names: list
    kappa_sample_names: list

    # optimize: turn this on when optimizng/running the minimizer so that the Jacobian factor isn't included
    # include_priors: this decides whether the marginalized parameter priors are included (should be yes)
    linear_param_dict_fn: str
    optimize: bool
    include_priors: bool 

    datfns: list

    covfn: str
    
    amin: list
    amax: list
    xmin: list
    xmax: list
        
    
    # Cell_lmins: list
    # Cell_lmaxs: list
    dndzfns: list
    cov_fac: float
    jeff: bool
    do_auto: bool


    def initialize(self):
        """Sets up the class."""
        # Redshift Label for theory classes
        # self.zstr = "%.2f" %(self.zfid)
        print(self.gal_sample_names,self.kappa_sample_names,self.datfns)

	# Load the linear parameters of the theory model theta_a such that
        # P_th = P_{th,nl} + theta_a P^a for some templates P^a we will compute
        self.linear_param_dict = yaml.load(open(self.linear_param_dict_fn), Loader=yaml.SafeLoader)
        self.linear_param_means = {key: self.linear_param_dict[key]['mean'] for key in self.linear_param_dict.keys()}
        self.linear_param_stds  = np.array([self.linear_param_dict[key]['std'] for key in self.linear_param_dict.keys()])
        self.Nlin = len(self.linear_param_dict)
        
        print("We are here!")
        
        
        #Set fiducial cosmology
        fid_cosmo = [0.02237,0.12,0.9649,np.log(1e10 * 2.0830e-9),67.36,0.06] # omb,omc,ns,ln(1e10 As),H0,Mnu
        fid_bias  = [0.9,0.,0.]                            # b1, b2, bs
        fid = np.array(fid_cosmo+fid_bias)
        
        #Now load dndzs:
        dndzs     = [np.loadtxt(self.basedir + self.dndzfns[i]) for i in range(len(self.gal_sample_names))]
        self.dndz = pack_dndz(dndzs)
        
        # set up the theory prediction class.
        self.cl_pred = limb(self.dndz, fid, pgmHEFT, pggHEFT, pmmHEFT, classyBackground,lmax=6000, zmin=0.001, zmax=1.8, Nz=80,zeffs=self.zfids)
         
        self.zeffs = self.cl_pred.zeff
        self.l_thy = self.cl_pred.l
        self.Nl = self.cl_pred.Nl
        self.lmax = np.max(self.l_thy)
        
        self.names = self.kappa_sample_names + self.gal_sample_names#['k', 'g1', 'g2', 'g3']

        
        # process = psutil.Process()
        # print(f'Current memory usage before loading specz Cell Data: {process.memory_info().rss*1e-9} GB')
        self.loadData()
        print('Loaded data!')
        # process = psutil.Process()
        # print(f'Current memory usage after loading specz Cell Data: {process.memory_info().rss*1e-9} GB')
        #

    def get_requirements(self):
        
        req = {'Cell_tables': None,\
               # 'ns': None,\
               # # 'w': None,\
               # 'm_ncdm': None,\
               # 'H0': None,\
               # # 'sigma8': None,\
               # # 'omegam': None,\
               # 'omega_b': None,\
               # 'omega_cdm': None,\
               # 'logA': None
              }
        
        # for gal_sample_name in self.gal_sample_names:
        #     req_bias = { \
        #            'bsig8_' + gal_sample_name: None,\
        #            # 'b1_' + fs_sample_name: None,\
        #            'b2sig8_' + gal_sample_name: None,\
        #            'bssig8_' + gal_sample_name: None,\
        #            'smag_' + gal_sample_name: None,\
        #            # 'b3sig8_' + fs_sample_name: None,\
        #           # 'alpha0_' + fs_sample_name: None,\
        #           #'alpha2_' + fs_sample_name: None,\
        #           # 'SN0_' + fs_sample_name: None,\
        #            #'SN2_' + fs_sample_name: None\
        #            }
        #     req = {**req, **req_bias}

        return(req)

    def full_predict(self, thetas=None):
        
        thy_obs = []

        if thetas is None:
            thetas = self.linear_param_means
        
        for ii,gal_sample_name in enumerate(self.gal_sample_names):
            zeff = self.zeffs[ii]
            Cell_obs  = self.Cell_predict(gal_sample_name,z_ind=ii,zeff=zeff,thetas=thetas)
            # Cell_obs  = self.Cell_observe(Cell_thy, gal_sample_name)
            # print(np.shape(Cell_obs))
            thy_obs = np.concatenate( (thy_obs,Cell_obs) )
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
        # process = psutil.Process()
        # print(f'Current memory usage after specz Cell likelihood evaluation: {process.memory_info().rss*1e-9} GB')
        
        return lnL
    
    def get_best_fit(self):
        try:
            self.p0_nl  = self.dd - self.Delta
            self.bf_thetas = np.einsum('ij,j', np.linalg.inv(self.Lab), self.Va)
            self.p0_lin = np.einsum('i,il', self.bf_thetas, self.templates)
            return self.p0_nl + self.p0_lin
        except:
            print("Make sure to first compute the posterior.")
    
    def load_cell_json(self,fn,gname,kname,gind,kind):
        with open(fn, 'r') as file:
            cell_data = json.load(file)
        ells = np.array(cell_data['ell'])
        # lcut = (ells <= self.lmax) & (ells >= 20.)
        # ells = ells[lcut]
        # Nells = len(ells)
        acuts, xcuts = get_scale_cuts(cell_data, [self.amin[gind]],[self.amax[gind]],\
                                      [[self.xmin[kind][gind]]],[[self.xmax[kind][gind]]])
        
        Ckg_dat = np.array(cell_data[f'cl_{kname}_{gname}'])[0,xcuts[0][0]]
        Cgg_dat = np.array(cell_data[f'cl_{gname}_{gname}'])[0,acuts[0]]
        Wkg = np.array(cell_data[f'w_{kname}_{gname}'])[0,xcuts[0][0],0,:self.Nl]
        Wgg = np.array(cell_data[f'w_{gname}_{gname}'])[0,acuts[0],0,:self.Nl]
        # self.Nells = Nells
        self.ldats[f'{kname}_{gname}'] = ells[xcuts[0][0]]
        self.ldats[f'{gname}_{gname}'] = ells[acuts[0]]
        del cell_data
        return Ckg_dat, Cgg_dat,Wkg,Wgg
    
    def load_cov_json(self):
        with open(self.basedir + self.covfn, 'r') as file:
            cov_data = json.load(file)
        names = self.names
        Nell = len(cov_data['ell']) #self.Nells
        
        full_cov = pack_cov(cov_data, self.kappa_sample_names, self.gal_sample_names, \
                           self.amin,self.amax, self.xmin, self.xmax)
        
        acuts, xcuts = get_scale_cuts(cov_data, self.amin,self.amax, self.xmin, self.xmax)
        # Initialize the full covariance matrix of size (3 * Nell, 3 * Nell)
#         num_gal_fields = len(names) - 1  # Exclude 'k'
#         full_cov = np.zeros((num_gal_fields * Nell, num_gal_fields * Nell))

#         # Fill in the blocks by looping over all galaxy field combinations
#         for i in range(1, len(names)):       # Start from 'g1' (ignore 'k')
#             for j in range(1, len(names)):   # Same here
#                 key = f'cov_k_{names[i]}_k_{names[j]}'  # Construct key for each cross-covariance

#                 if key in cov_data:
#                     # Calculate the index range in the full matrix for this (i, j) block
#                     row_start, row_end = (i - 1) * Nell, i * Nell
#                     col_start, col_end = (j - 1) * Nell, j * Nell

#                     # Fill the full covariance matrix with the corresponding covariance block
#                     full_cov[row_start:row_end, col_start:col_end] = cov_data[key]
#                     #symmetrize
#                     if i != j: 
#                         full_cov[col_start:col_end,row_start:row_end] = cov_data[key]
        
        
        self.mc_corr = {}
        for i,gname in enumerate(self.gal_sample_names):
            for j,kname in enumerate(self.kappa_sample_names):
                self.mc_corr[f'{kname}_{gname}'] = np.array(cov_data[f'mccorr_{kname}_{gname}'])[xcuts[j][i]]
                        
        return full_cov

        
    def loadData(self):
        """
        Loads the required data.
        
        Do this in two steps... first load full shape data then xirecon, concatenate after.
        
        The covariance is assumed to already be joint in the concatenated format.
        
        """
        
        #First load the covariance matrix.
        cov = self.load_cov_json()/self.cov_fac
        
        
        
        # Now load the data
        
        #data ells, Cells, and mode-coupling/window matrix from catalog-based Cell file
        self.ldats = {}
        self.Ckg_dats = {}
        self.Cgg_dats = {}
        self.Wkgs = {}
        self.Wggs = {}
        
        # for ii, datfn in enumerate(self.datfns):
        for ii,gname in enumerate(self.gal_sample_names):
            for jj,kname in enumerate(self.kappa_sample_names):
                datfn = self.datfns[ii][jj]
                Ckg_dat, Cgg_dat,Wkg,Wgg = self.load_cell_json(self.basedir + datfn,gname,kname,ii,jj)
                sname = f'{kname}_{gname}'
                
                # self.ldats[sname] = ells
                self.Ckg_dats[sname] = Ckg_dat*self.mc_corr[sname]
                self.Wkgs[sname] = Wkg
            self.Cgg_dats[gname] = Cgg_dat
            self.Wggs[gname] = Wgg
        
        # Join the data vectors together
        self.dd = []        
        for gname in self.gal_sample_names:
            self.dd = np.concatenate( (self.dd, self.Cgg_dats[gname]) )
            for kname in self.kappa_sample_names:
                self.dd = np.concatenate((self.dd, self.Ckg_dats[f'{kname}_{gname}']))
        
        
        
        
        # We're only going to want some of the entries in computing chi^2.
        
#         # this is going to tell us how many indices to skip to get to the nth multipole
#         startii = 0
        
#         for ss, sample_name in enumerate(self.gal_sample_names):
            
#             lcut = (self.ldats[sample_name] > self.Cell_lmaxs[ss])\
#                           | (self.ldats[sample_name] < self.Cell_lmins[ss])
            
#             for i in np.nonzero(lcut)[0]:     # FS Monopole.
#                 ii = i + startii
#                 cov[ii, :] = 0
#                 cov[ :,ii] = 0
#                 cov[ii,ii] = 1e25
            
#             startii += self.ldats[sample_name].size
            
        
        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)
        #print(self.sample_name, np.diag(self.cinv)[:10])
        
        
        
    def get_cosmo_parameters(self):
        pp  = self.provider
        omb = pp.get_param('omega_b')
        omc = pp.get_param('omega_cdm')
        ns  = pp.get_param('ns')
        logAs  = pp.get_param('logA')
        H0  = pp.get_param('H0')
        Mnu = pp.get_param('m_ncdm')
        return omb,omc,ns,logAs,H0,Mnu
    
    def Cell_predict(self, gal_sample_name,z_ind,zeff, thetas=None):
        """Use the Aemulus emulator to compute C_ell, given biases etc."""
        
        pp   = self.provider
        # zstr = "%.2f" %(zeff)
        Cell_tables = pp.get_result('Cell_tables')
        
        # Cgg and Ckg are tables of shape (nell,4)
        # where the four columns correspond to 
        # 1, alpha_auto, shot noise, alpha_cross
        Cgg = Cell_tables[gal_sample_name]['auto'] 
        Ckg = Cell_tables[gal_sample_name]['cross']
        
        
       # Instead of calling the linear parameters directly we will now analytically marginalize over them
        
        if thetas is None:
            alphax = self.linear_param_means['alphax_' + gal_sample_name]
            # sn0 = self.linear_param_means['SN0_' + gal_sample_name]
        else:
            alphax = thetas['alphax_' + gal_sample_name]
            # sn0 = thetas['SN0_' + gal_sample_name]
        
        monomials = np.array([1.,0.,0.,alphax])
        
        thy_gg = np.dot(Cgg,monomials)
        thy_kg = np.dot(Ckg,monomials)
        
        conv = np.dot(self.Wggs[gal_sample_name],thy_gg)
        for i, kappa_sample_name in enumerate(self.kappa_sample_names):
            sname = f'{kappa_sample_name}_{gal_sample_name}'
            conv = np.concatenate((conv, np.dot(self.Wkgs[sname],thy_kg)))
        
        # Mmat = self.matMs[gal_sample_name]
        
        return conv #np.matmul(Mmat,thy)



class Taylor_Cells(Theory):
    """
    A class to return a set of derivatives for the Taylor series of sigma8, Dz, fz.
    """
    zeffs: list
    basedir: str
    gal_sample_names: list
    dndzfns: list
    do_auto: bool
    
    def initialize(self):
        """Sets up the class by loading the derivative matrices."""
        
        self.nnemu = NNHEFTEmulator() #NN emulator to be used to get sigma8(z)
        
        #Set fiducial cosmology
        fid_cosmo = [0.02237,0.12,0.9649,np.log(1e10 * 2.0830e-9),67.36,0.06] # omb,omc,ns,ln(1e10 As),H0,Mnu
        fid_bias  = [0.9,0.,0.]                            # b1, b2, bs
        fid = np.array(fid_cosmo+fid_bias)
        
        #Now load dndzs:
        dndzs     = [np.loadtxt(self.basedir + self.dndzfns[i]) for i in range(len(self.gal_sample_names))]
        self.dndz = pack_dndz(dndzs)
        
        # set up the theory prediction class.
        self.cl_pred = limb(self.dndz, fid, pgmHEFT, pggHEFT, pmmHEFT, classyBackground,lmax=6000, zmin=0.001, zmax=1.8, Nz=80,zeffs=self.zeffs)
        
    
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        # zmax = max(self.zfids)
        # zg  = np.linspace(0,zmax,100,endpoint=True)
        # Don't need sigma8_z, fsigma8 or radial distance
        # here, but want them up in likelihood and they
        # only depend on cosmological things (not biases).
        #
        req = {\
               'ns': None,\
               # 'w': None,\
               'omega_b': None,\
               'omega_cdm': None,\
               'H0': None,\
               'logA': None,\
               'm_ncdm': None,\
              }
        
        for gal_sample_name in self.gal_sample_names:
            req_bias = { \
                   'bsig8_' + gal_sample_name: None,\
                   # 'b1_' + fs_sample_name: None,\
                   'b2sig8_' + gal_sample_name: None,\
                   'bssig8_' + gal_sample_name: None,\
                   'smag_' + gal_sample_name: None,\
                   }
            req = {**req, **req_bias}
        
        return(req)
    
    def get_cosmo_parameters(self):
        pp  = self.provider
        omb = pp.get_param('omega_b')
        omc = pp.get_param('omega_cdm')
        ns  = pp.get_param('ns')
        logAs  = pp.get_param('logA')
        H0  = pp.get_param('H0')
        Mnu = pp.get_param('m_ncdm')
        return omb,omc,ns,logAs,H0,Mnu
    
    def get_can_provide(self):
        """What do we provide: a dictionary with tables for Cells."""
        return ['Cell_tables']
    
    # def get_can_provide_params(self):
    #     return ['sigma8','omegam']
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Just load up the derivatives and things.
        """
        pp   = self.provider
        
        omb,omc,ns,logAs,H0,Mnu = self.get_cosmo_parameters()
        As = np.exp(logAs)*1e-10 
        
        Cell_tables = {}
        
        for ii,gal_sample_name in enumerate(self.gal_sample_names):
            zeff = self.zeffs[ii]
            zstr = "%.2f" %(zeff)
            
            params = np.array([omb,omc,-1.0,ns, As,H0,np.log10(Mnu),zeff])[self.nnemu.param_order] 
            sig8_z = self.nnemu.sigma8z_emu(params)[0]
            
            b1   = pp.get_param('bsig8_' + gal_sample_name)/sig8_z - 1
            # b1   = pp.get_param('b1_' + fs_sample_name)
            b2   = pp.get_param('b2sig8_' + gal_sample_name)/(sig8_z**2)
            bs   = pp.get_param('bssig8_' + gal_sample_name)/(sig8_z**2)
            smag = pp.get_param('smag_' + gal_sample_name)
            
            params = np.array([omb,omc,ns,logAs,H0,Mnu,b1,b2,bs])
            
            Cell_tables[gal_sample_name] = {}
            
            if self.do_auto:
                Cgg,Ckg = self.cl_pred.computeCggCkg(ii,params,smag)
                Cell_tables[gal_sample_name]['auto'] = Cgg
                Cell_tables[gal_sample_name]['cross'] = Ckg
            else:
                Ckg = self.cl_pred.computeCkg(ii,params,smag)
                Cell_tables[gal_sample_name]['auto'] = Ckg #will not be used
                Cell_tables[gal_sample_name]['cross'] = Ckg
            
        state['Cell_tables'] = Cell_tables
        # process = psutil.Process()
        # print(f'Current memory usage after specz Cell theory evaluation: {process.memory_info().rss*1e-9} GB')
