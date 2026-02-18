from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad, cumulative_trapezoid as cumtrapz, trapezoid as trapz, simpson
from scipy.interpolate import interp1d,RegularGridInterpolator, InterpolatedUnivariateSpline
from scipy.optimize import brentq

from colossus.cosmology import cosmology as colossus_cosmo

import camb
from camb import get_matter_power_interpolator, model, nonlinear

from fastpt import FASTPT

class units_and_constants:
    def __init__(self):
        self.Mpc      = 1.
        self.kpc      = self.Mpc/1000.
        self.pc       = self.kpc/1000.
        self.cm       = self.pc/3.086e18
        self.km       = 1.e5*self.cm
        self.s        = 1.
        self.yr       = 3.15576e7*self.s
        self.Gyr      = 1.e9*self.yr
        self.Msun     = 1.
        self.gram     = self.Msun/1.988e33
        self.c        = 2.9979e10*self.cm/self.s
        self.G        = 6.6742e-8*self.cm**3/self.gram/self.s**2

        self.As       = 2.19e-9 
        self.ns       = 0.97

class cosmology(units_and_constants):
    
    def __init__(self):
        units_and_constants.__init__(self)
        self.OmegaB        = 0.049
        self.OmegaM        = 0.315
        self.OmegaC        = self.OmegaM-self.OmegaB
        self.OmegaL        = 1.-self.OmegaM
        self.h             = 0.674
        self.H0            = self.h*100*self.km/self.s/self.Mpc 
        self.rhocrit0      = 3*self.H0**2/(8.0*np.pi*self.G)  # Msun / Mpc^3 

    def g(self, z):
        return self.OmegaM*(1.+z)**3+self.OmegaL

    def Hubble(self, z):
        return self.H0*np.sqrt(self.OmegaM*(1.+z)**3+self.OmegaL)

    def rhocrit(self, z):
        return 3.*self.Hubble(z)**2/(np.pi*8.0*self.G)
    
    def growth_factor(self, z):
        Omega_Lz = self.OmegaL/(self.OmegaL+self.OmegaM*(1.+z)**3)
        Omega_Mz = 1-Omega_Lz
        phiz     = Omega_Mz**(4./7.)-Omega_Lz+(1.+Omega_Mz/2.0)*(1.+Omega_Lz/70.0)
        phi0     = self.OmegaM**(4./7.)-self.OmegaL+(1.+self.OmegaM/2.0)*(1.+self.OmegaL/70.0)
        return (Omega_Mz/self.OmegaM)*(phi0/phiz)/(1.+z)

        
    def dDdz(self, z):
        def dOdz(z):
            return -self.OmegaL*3*self.OmegaM*(1+z)**2*(self.OmegaL+self.OmegaM*(1+z)**3.)**-2
        Omega_Lz = self.OmegaL*pow(self.OmegaL+self.OmegaM*pow(self.h,-2)*pow(1+z,3),-1)
        Omega_Mz = 1-Omega_Lz
        phiz     = Omega_Mz**(4./7.)-Omega_Lz+(1+Omega_Mz/2.)*(1+Omega_Lz/70.)
        phi0     = self.OmegaM**(4./7.)-self.OmegaL+(1+self.OmegaM/2.)*(1+self.OmegaL/70.)
        dphidz   = dOdz(z)*(-4./7.*Omega_Mz**(-3.0/7.0)+(Omega_Mz-Omega_Lz)/140.+1./70.-3./2.)
        return (phi0/self.OmegaM)*(-dOdz(z)/(phiz*(1+z))-Omega_Mz*(dphidz*(1+z)+phiz)/phiz**2/(1+z)**2)
    


    def sigma_R(self, R, Pk, k_grid):
        def window_func(k, R):
            x = k * R
            W = np.ones_like(x)
            W = 3 * (np.sin(x) - x*np.cos(x)) / x**3
            return W

        integrand = k_grid**2 * Pk * window_func(k_grid, R)**2
        integral = simpson(integrand, k_grid)
        
        return float(np.sqrt(integral / (2 * np.pi**2)))


    def sigma8(self, Pk, k_grid):
        return self.sigma_R(8.0/self.h, Pk, k_grid)
    
    def neff_at_R(self, R, Pk, k_grid, eps=1e-2):
        """
        Compute n_eff at a single R using symmetric log-derivative.
        eps = fractional log step size
        """

        R1 = R * np.exp(-eps)
        R2 = R * np.exp(+eps)

        sigma1 = self.sigma_R(R1, Pk, k_grid)
        sigma2 = self.sigma_R(R2, Pk, k_grid)

        ln_sigma2_1 = np.log(sigma1**2)
        ln_sigma2_2 = np.log(sigma2**2)

        dlnsigma2_dlnR = (ln_sigma2_2 - ln_sigma2_1) / (2*eps)

        neff = -dlnsigma2_dlnR - 3.0

        return neff
    
    def find_Rsigma(self, Pk, k_grid):

        def func(R):
            return self.sigma_R(R, Pk, k_grid) - 1.0

        return brentq(func, 0.01, 50.0)  


    def comoving_distance(self, z):
        """
        Comoving distance [Mpc] at redshift z, accepts scalar and vector input
        """
        integrand = lambda z: 1 / self.Hubble(z)
        if np.isscalar(z):
            integral, _ = quad(integrand, 0, z)
            return  self.c * integral
        else:
            integrals =  np.array([quad(integrand, 0, x)[0] for x in z])
            return  self.c * integrals
    
    def chi_to_z_interp(self,zmin,zmax):
        """
        Creates interpolator from z in range (zmin,zmax) to chi 

        """
        z_grid = np.linspace(zmin, zmax, 1000)  
        chi_grid = self.comoving_distance(z_grid)
        
        chi_to_z_interp = interp1d(chi_grid, z_grid, kind='cubic', bounds_error=True)
        
        return chi_to_z_interp
    
    def nz_to_nchi_interp(self, nz, z, z_mean=0, shift=0, stretch=1):
        # Apply shift/stretch to z
        z = (z - z_mean) / stretch + z_mean - shift
        nz = nz / stretch

        # Normalize n(z) so ∫ n(z) dz = 1
        norm = np.trapezoid(nz, z)
        nz = nz / norm

        # Convert to n(chi): n(chi) = n(z) * H(z) / c
        nchi = nz * self.Hubble(z) / self.c 
        chi = self.comoving_distance(z)

        # Interpolator
        nchi_interp = interp1d(chi, nchi, kind='cubic', bounds_error=False, fill_value=0)
        return nchi_interp


class power_spectrum(cosmology):

    def __init__(self, zmin=0.01, zmax=2.0, n=256, kmin=1e-4, kmax=5, Pmm_file=None,Pmm_field=None,use_camb=False,nonLinear=True,nonlin_model = 'mead2020', feedback=False):
        super().__init__()

        self.zmin = zmin
        self.zmax = zmax
        self.n = n
        self.kmin = kmin 
        self.kmax = kmax 
        self.Pmm_file = Pmm_file  
        self.Pmm_field = Pmm_field
        self.use_camb = use_camb
        self.nonLinear = nonLinear
        self.nonlin_model = nonlin_model
        self.feedback = feedback

        # k-grid 
        self.k = np.logspace(np.log10(kmin), np.log10(kmax), n) # 1/Mpc
        self.kh = self.k / self.h  # Convert to h/Mpc

        # z-grid for power spectra
        self.z_ps = np.linspace(zmin, zmax, n)

        # chi-grid [Mpc/h]
        chi_min = self.comoving_distance(zmin)
        chi_max = self.comoving_distance(zmax)
        self.chi = np.linspace(chi_min, chi_max, n)

        # chi <-> z mapping
        self.chi_to_z = self.chi_to_z_interp(zmin=zmin, zmax=zmax)
        self.z = self.chi_to_z(self.chi)

        print(
            f"kmin = {self.k[0]:.3e}, kmax = {self.k[-1]:.3e}, "
            f"chimin = {self.chi[0]:.3e}, chimax = {self.chi[-1]:.3e}, "
            f"zmin = {self.z[0]:.3f}, zmax = {self.z[-1]:.3f}"
        )

        # Initialize Colossus cosmology
        col_params = {
            'flat': True,
            'H0': self.h * 100,                 # H0 in km/s/Mpc
            'Om0': self.OmegaM,                 # Omega_matter
            'Ob0': self.OmegaB,                 # Omega_baryon
            'sigma8' : 0.84648,              
            'ns': self.ns,                      # spectral index
        }
        self.cosmo_col = colossus_cosmo.setCosmology('custom', col_params)

        # Build Pmm interpolator
        if Pmm_file is not None:
            self._logPmm_interp = self.get_logP_interp_from_file(self.Pmm_file, field=self.Pmm_field)
            print(f"Loaded Pmm from file: {Pmm_file}, field: {self.Pmm_field}")
        else:
            if use_camb:
                self._Pmm_grid = self.get_Pmm_interp_from_camb(nonLinear=self.nonLinear,feedback=self.feedback)(self.z_ps, self.k)
                print("Computed Pmm using CAMB")
                self._logPmm_interp = self.make_logP_interpolator(self.z_ps, self.k, self._Pmm_grid)
            else:
                self._Pmm_grid = self._compute_Pmm_colossus()
                print("Computed Pmm using Colossus")
                self._logPmm_interp = self.make_logP_interpolator(self.z_ps, self.k, self._Pmm_grid)

    def get_sigma8(self):
        sigma8s = np.zeros_like(self.z_ps)
        for i, z in enumerate(self.z_ps):
            Pk_z = np.exp(self._logPmm_interp((z, self.k)))
            sigma8s[i] = self.sigma8(Pk_z,self.k)
        return sigma8s

    def _compute_Pmm_colossus(self):
        P_grid = np.zeros((len(self.z_ps), len(self.k)))
        for i, z in enumerate(self.z_ps):
            P_grid[i, :]  = self.cosmo_col.matterPowerSpectrum(self.kh, z=z) / self.h**3  # Convert from (Mpc/h)^3 to Mpc^3
        return P_grid

    def get_Pmm_interp_from_camb(self,nonLinear=True,feedback=False):  
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.h*100,                          # CAMB expects km/s/Mpc
            ombh2=self.OmegaB * self.h**2,
            omch2=self.OmegaC * self.h**2,
            mnu=0.0,                                    # no massive neutrinos by default
            omk=0.0,                                    # flat universe
            tau=0.054                                   # optical depth, Planck 2018 default
        )
    
        pars.InitPower.set_params(As=self.As, ns=self.ns)
        pars.set_matter_power(redshifts=self.z_ps.tolist(), kmax=self.kmax)
        if nonLinear:
            pars.NonLinear = model.NonLinear_both
            if feedback:
                pars.NonLinearModel = nonlinear.Halofit(halofit_version='mead2020', HMCode_logT_AGN=7.8)
            else:
                pars.NonLinearModel = nonlinear.Halofit(halofit_version=self.nonlin_model)


        else:
            pars.NonLinear = model.NonLinear_none
        
        PK = get_matter_power_interpolator(
            pars,
            nonlinear=nonLinear,
            hubble_units=False,
            k_hunit=False,
            kmax=self.kmax,
            zmax=self.zmax
        )

        return PK.P  

    def make_logP_interpolator(self, z, k, P):
        interp = RegularGridInterpolator(
            (z, k),
            np.log(P),
            bounds_error=False,
            fill_value=None
        )
        return interp

    def get_logP_interp_from_file(self, filename, field):

        data = np.load(filename)

        k = data["k"]                   # (Nk,)
        z = data["z"]                   # (Nz,)
        P = data[field]                 # (Nz, Nk)

        # Check if k and z grids cover the required ranges

        if k.min() > self.kmin:
            raise ValueError(
                "k grid in file does not cover the required range: "
                "kmin = {:.3e} > {:.3e}".format(k.min(), self.kmin)
            )

        if k.max() < self.kmax:
            raise ValueError(
                "k grid in file does not cover the required range: "
                "kmax = {:.3e} < {:.3e}".format(k.max(), self.kmax)
            )

        if z.min() > self.z_ps.min():
            raise ValueError(
                "z grid in file does not cover the required range: "
                "zmin = {:.3f} > {:.3f}".format(z.min(), self.z_ps.min())
            )

        if z.max() < self.z_ps.max():
            raise ValueError(
                "z grid in file does not cover the required range: "
                "zmax = {:.3f} < {:.3f}".format(z.max(), self.z_ps.max())
            )

        return self.make_logP_interpolator(z, k, P)

    def get_logPmm_interp(self):
        return self._logPmm_interp

    def get_logPgm_interp(self, galaxy_bias):

        logP = self._logPmm_interp

        Z, K = np.meshgrid(self.z_ps, self.k, indexing="ij")
        pts = np.column_stack([Z.ravel(), K.ravel()])

        P = np.exp(logP(pts)).reshape(len(self.z_ps), len(self.k))

        P *= galaxy_bias

        return self.make_logP_interpolator(self.z_ps, self.k, P)

    def get_logPgg_interp(self, galaxy_bias):

        logP = self.get_logPmm_interp()

        Z, K = np.meshgrid(self.z_ps, self.k, indexing="ij")
        pts = np.column_stack([Z.ravel(), K.ravel()])

        P = np.exp(logP(pts)).reshape(len(self.z_ps), len(self.k)) 

        P *= galaxy_bias**2

        return self.make_logP_interpolator(self.z_ps, self.k, P)

    def get_PgI_interp(
        self,
        NLA=False,
        C1=5e-14,
        IA_pars=np.array([0.7, -1.36, -1.7, -2.5, 1.0, 0.62])
        ):

        k = self.k                      
        z_values = self.z_ps

        rho_crit = self.rhocrit0  # Msun Mpc-3
        C1 *= self.h**-2 #h−2 Msun−1 Mpc3 -> Msun-1 Mpc3
        Om0 = self.OmegaM

        logPmm = self._logPmm_interp

        # ---------- NLA MODEL ----------
        if NLA:

            Z, K = np.meshgrid(z_values, k, indexing="ij")
            pts = np.column_stack([Z.ravel(), K.ravel()])
            Pmm = np.exp(logPmm(pts)).reshape(len(z_values), len(k))

            GI = np.zeros_like(Pmm)

            for i, z in enumerate(z_values):
                A1 = (
                    -IA_pars[0] * C1 * rho_crit * Om0
                    / self.growth_factor(z)
                    * ((1 + z) / (1 + IA_pars[5]))**IA_pars[2]
                )
                GI[i] = A1 * Pmm[i]

            return RegularGridInterpolator((z_values, k), GI, bounds_error=False, fill_value=None)

        fastpt = FASTPT(k, n_pad=800)

        # Pmm at z=0
        pts0 = np.column_stack([np.zeros_like(k), k])
        
        P_mm0 = np.exp(logPmm(pts0)) 

        P_deltaE1, P_deltaE2, *_ = fastpt.IA_ta(P_mm0) 
        P_A, P_B, *_ = fastpt.IA_mix(P_mm0) 

        GI = np.zeros((len(z_values), len(k)))

        for i, z in enumerate(z_values):

            pts = np.column_stack([np.full_like(k, z), k])
            Pmm = np.exp(logPmm(pts))

            A1 = (
                -IA_pars[0] * C1 * rho_crit * Om0
                / self.growth_factor(z)
                * ((1 + z) / (1 + IA_pars[5]))**IA_pars[2]
            )

            A2 = (
                5 * IA_pars[1] * C1 * rho_crit * Om0
                / self.growth_factor(z)**2
                * ((1 + z) / (1 + IA_pars[5]))**IA_pars[3]
            )

            GI[i] = (
                A1 * Pmm
                + IA_pars[4] * A1 * (P_deltaE1 + P_deltaE2) 
                + A2 * (P_A + P_B) 
            )
        return RegularGridInterpolator((z_values, k), GI, bounds_error=False, fill_value=None)



    def lensing_efficiency(self, nchi_interp, shear=0):
        z = self.z
        chi = self.chi

        prefactor = 1.5 * self.H0**2 * self.OmegaM / self.c**2

        f = nchi_interp(chi) / chi
        fchi = nchi_interp(chi)

        int_f = cumtrapz(f, chi, initial=0)
        int_fchi = cumtrapz(fchi, chi, initial=0)

        total_f = int_f[-1]
        total_fchi = int_fchi[-1]

        integral_from_i = (total_fchi - int_fchi) - chi * (total_f - int_f)

        q = prefactor * chi * (1 + z) * integral_from_i

        return q * (1 + shear)
    
    def cls_gI(self,l_bins, zl, zs, nz_lens, nz_source, zl_mean, nz_lens_stretch = 1,  nz_lens_shift = 0, NLA=True,  C1=5e-14,IA_pars = np.array([0.7,-1.36,-1.7,-2.5,1.0,0.62])):
        chi = self.chi
        z = self.z
        PgI = self.get_PgI_interp(NLA=NLA,C1=C1,IA_pars = IA_pars)

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,z_mean=zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)
        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        cls = []
        for l in l_bins:
            kp = (l + 0.5) / chi    
            pts = np.vstack([z, kp]).T    
            PgI_chi = PgI(pts)

            integrand = nchi_lens_interp(chi) * nchi_source_interp(chi) * PgI_chi / chi**2
            cls.append(simpson(integrand, chi))

        return np.array(cls)

    def cls_mag_I(self,l_bins,zl, zs, nz_lens, nz_source, zl_mean, magnification_bias, nz_lens_stretch = 1,  nz_lens_shift = 0, NLA=True, C1=5e-14,IA_pars = np.array([0.7,-1.36,-1.7,-2.5,1.0,0.62])):
        chi = self.chi
        z = self.z
        PgI = self.get_PgI_interp(NLA=NLA,C1=C1,IA_pars = IA_pars)

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,z_mean=zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)
        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        q_lens = self.lensing_efficiency(nchi_interp = nchi_lens_interp,shear=0)

        cls = []
        for l in l_bins:
            kp = (l + 0.5) / chi 
            pts = np.vstack([z, kp]).T     
            PgI_chi = PgI(pts)

            integrand = magnification_bias * q_lens  * nchi_lens_interp(chi) * nchi_source_interp(chi) * PgI_chi / chi**2
            cls.append( simpson(integrand, chi))

        return np.array(cls)

    def cls_mag_k(self,l_bins, galaxy_bias, zl, zs, nz_lens, nz_source, zl_mean,magnification_bias, nz_lens_stretch = 1,  nz_lens_shift = 0, shear=0):
        chi = self.chi
        z = self.z
        logPgm = self.get_logPgm_interp(galaxy_bias=galaxy_bias) 

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)
        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        q_source = self.lensing_efficiency(nchi_interp = nchi_source_interp,shear=shear)
        q_lens = self.lensing_efficiency(nchi_interp = nchi_lens_interp,shear=0)

        cls = []
        for l in l_bins:
            kp = (l + 0.5) / chi 

            pts = np.column_stack([z, kp])

            Pgm_chi = np.exp(logPgm(pts)) 

            integrand = (magnification_bias * q_lens * nchi_lens_interp(chi) * q_source * Pgm_chi / chi**2)

            cls.append(simpson(integrand, chi))

        return np.array(cls)
 
    def cls_gk(self, l_bins, galaxy_bias, zl, zs, nz_lens, nz_source, zl_mean, nz_lens_stretch = 1,  nz_lens_shift = 0, shear=0):
        chi = self.chi
        z = self.z
        logPgm = self.get_logPgm_interp(galaxy_bias=galaxy_bias) 

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)
        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        q_source = self.lensing_efficiency(nchi_interp = nchi_source_interp,shear=shear)

        cls = []
        for l in l_bins:
            Pgm_chi = np.zeros_like(chi)
            kp = (l + 0.5) / chi
            pts = np.vstack([z, kp]).T      # shape (nchi, 2)

            Pgm_chi = np.exp(logPgm(pts))
            integrand = q_source * nchi_lens_interp(chi) * Pgm_chi / chi**2
            cls.append( simpson(integrand, chi))

        return np.array(cls)

        
    def cls_gg(self, l_bins, galaxy_bias, zl, nz_lens,  zl_mean, nz_lens_stretch = 1,  nz_lens_shift = 0):
        chi = self.chi
        z = self.z
        logPgg = self.get_logPgg_interp(galaxy_bias = galaxy_bias)

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)

        cls = []
        for l in l_bins:
            Pgg_chi = np.zeros_like(chi)
            kp = (l + 0.5) / chi
            pts = np.vstack([z, kp]).T      # shape (nchi, 2)
            Pgg_chi = np.exp(logPgg(pts)) 

            integrand = nchi_lens_interp(chi)**2 * Pgg_chi / chi**2
            cls.append( simpson(integrand, chi))

        return np.array(cls)
    
    def cls_kk(self, l_bins, zs, nz_source,  shear=0):
        chi = self.chi
        z = self.z
        logPmm = self.get_logPmm_interp()

        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        q_source = self.lensing_efficiency(nchi_interp = nchi_source_interp,shear=shear)

        cls = []
        for l in l_bins:
            Pmm_chi = np.zeros_like(chi)  
            kp = (l + 0.5) / chi 
            pts = np.vstack([z, kp]).T      # shape (nchi, 2)
            Pmm_chi = np.exp(logPmm(pts))

            integrand = q_source **2 * Pmm_chi / chi**2
            cls.append( simpson(integrand, chi))

        return np.array(cls)