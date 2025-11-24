from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad, cumulative_trapezoid as cumtrapz, trapezoid as trapz
from scipy.interpolate import interp1d,RegularGridInterpolator

import camb
from camb import get_matter_power_interpolator, model

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
        self.rhocrit0      = 3*self.H0**2/(8.0*np.pi*self.G) 

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
        dz = np.gradient(z)
        norm = np.sum(nz * dz)
        nz /= norm

        # Convert to n(chi): n(chi) = n(z) * H(z) / c
        nchi = nz * self.Hubble(z) / self.c 
        chi = self.comoving_distance(z)

        # Interpolator
        nchi_interp = interp1d(chi, nchi, kind='cubic', bounds_error=False, fill_value=0)
        return nchi_interp
    
class power_spectrum(cosmology):
    def __init__(self, zmin=0.15, zmax=6.0, n=256, kmin=1e-4, kmax=5):
        super().__init__()  
        self.zmin = zmin
        self.zmax = zmax
        self.n = n
        self.kmin = kmin
        self.kmax = kmax

        # k-grid in Fourier space [1/Mpc]
        self.k = np.logspace(np.log10(kmin), np.log10(kmax), n)
        self.z_ps = np.linspace(0,zmax,n)

        # chi-grid in comoving space [Mpc]
        chi_min = self.comoving_distance(zmin)
        chi_max = self.comoving_distance(zmax)
        self.chi = np.linspace(chi_min, chi_max, n)

        # mapping chi <-> z
        self.chi_to_z = self.chi_to_z_interp(zmin=zmin, zmax=zmax)
        self.z = self.chi_to_z(self.chi)

        print(
            f"kmin = {self.k[0]:.3e}, kmax = {self.k[-1]:.3e}, "
            f"chimin = {self.chi[0]:.3e}, chimax = {self.chi[-1]:.3e}, "
            f"zmin = {self.z[0]:.3f}, zmax = {self.z[-1]:.3f}"
        )
   
    def get_Pmm_interp(self):    
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.h * 100.0,                          # CAMB expects km/s/Mpc
            ombh2=self.OmegaB * self.h**2,
            omch2=self.OmegaC * self.h**2,
            mnu=0.0,                                    # no massive neutrinos by default
            omk=0.0,                                    # flat universe
            tau=0.054                                   # optical depth, Planck 2018 default
        )
    
        pars.InitPower.set_params(As=self.As, ns=self.ns)
        
        pars.set_matter_power(redshifts=self.z_ps.tolist(), kmax=self.kmax)
        pars.NonLinear = model.NonLinear_both
        
        PK = get_matter_power_interpolator(
            pars,
            nonlinear=True,
            hubble_units=False,
            k_hunit=False,
            kmax=self.kmax,
            zmax=self.zmax
        )
        
        return PK.P   
    
    def get_Pgm_interp(self, galaxy_bias):
        PK = self.get_Pmm_interp()
        Pgm = RegularGridInterpolator((self.z_ps, self.k), PK(self.z_ps, self.k)*galaxy_bias)
        return Pgm
    
    def get_Pgg_interp(self, galaxy_bias):
        PK = self.get_Pmm_interp()
        Pgm = RegularGridInterpolator((self.z_ps, self.k), PK(self.z_ps, self.k)*galaxy_bias**2)
        return Pgm
    
    def get_Pia_interp(self, NLA = False, C1=5e-14, IA_pars=np.array([0.7, -1.36, -1.7, -2.5, 1.0, 0.62])):
        """
        Build interpolator(s) for intrinsic alignment (IA) power spectra 
        using FAST-PT.

        Parameters
        ----------
        C1 : float
            Normalization constant for IA amplitude (default ~ 5e-14 h^-2 Msun^-1 Mpc^3).
        IA_pars : array_like
            Intrinsic alignment parameters:
            [a1, a2, alpha1, alpha2, b_TA, z0].

        Returns
        -------
        PGI : RegularGridInterpolator
            Interpolator for the GI cross-power spectrum, callable as PGI(z, k).

        Notes
        -----
        - Uses tidal alignment (A1) and tidal torquing (A2) models.
        - II spectrum is prepared but currently commented out.
        """

        if NLA == True:
            rho_crit = self.rhocrit0 / self.h**2 # Msun/(Mpc^3) -> h^2 Msun/(Mpc^3)
            Om0 = self.OmegaM
            k = self.k / self.h # 1/Mpc -> h/Mpc
            z_values = self.z_ps
                    # Matter power spectrum interpolator
            pars = camb.CAMBparams()
            pars.set_cosmology(
                H0=self.h * 100.0,                          # CAMB expects km/s/Mpc
                ombh2=self.OmegaB * self.h**2,
                omch2=self.OmegaC * self.h**2,
                mnu=0.0,                                    # no massive neutrinos by default
                omk=0.0,                                    # flat universe
                tau=0.054                                   # optical depth, Planck 2018 default
            )
        
            pars.InitPower.set_params(As=self.As, ns=self.ns)
            
            pars.set_matter_power(redshifts=self.z_ps.tolist(), kmax=self.kmax)
            pars.NonLinear = model.NonLinear_both
            
            PK = get_matter_power_interpolator(
                pars,
                nonlinear=True,
                hubble_units=True,
                k_hunit=True,
                kmax=self.kmax,
                zmax=self.zmax
            )
            
            Pmm = PK.P  
            GI_z = []
            for z in z_values:
                P_mm = Pmm(z, k)
                A1 = -IA_pars[0] * C1 * rho_crit * Om0 / self.growth_factor(z)  * ((1 + z) / (1 + IA_pars[5]))**IA_pars[2]
                GI = A1 * P_mm
                GI_z.append(GI)
            GI_array = np.array(GI_z)* self.h**3 # (Mpc/h)^3 -> Mpc^3
            PGI = RegularGridInterpolator((z_values, self.k), GI_array, bounds_error=False, fill_value=None)

            return PGI

        k = self.k / self.h # 1/Mpc -> h/Mpc
        z_values = self.z_ps

        rho_crit = self.rhocrit0 / self.h**2 # Msun/(Mpc^3) -> h^2 Msun/(Mpc^3)
        Om0 = self.OmegaM

        # Matter power spectrum interpolator
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.h * 100.0,                          # CAMB expects km/s/Mpc
            ombh2=self.OmegaB * self.h**2,
            omch2=self.OmegaC * self.h**2,
            mnu=0.0,                                    # no massive neutrinos by default
            omk=0.0,                                    # flat universe
            tau=0.054                                   # optical depth, Planck 2018 default
        )
    
        pars.InitPower.set_params(As=self.As, ns=self.ns)
        
        pars.set_matter_power(redshifts=self.z_ps.tolist(), kmax=self.kmax)
        pars.NonLinear = model.NonLinear_both
        
        PK = get_matter_power_interpolator(
            pars,
            nonlinear=True,
            hubble_units=True,
            k_hunit=True,
            kmax=self.kmax,
            zmax=self.zmax
        )
        
        Pmm = PK.P   
        # Initialize FAST-PT
        P_window = np.array([0.2, 0.2])
        C_window = 0.65
        n_pad = 800
        fastpt = FASTPT(k, to_do=['IA'], n_pad=n_pad)

        # Precompute redshift-independent FAST-PT spectra at z=0
        P_mm0 = Pmm(0.0, k) 
        P_deltaE1, P_deltaE2, P_0E0E, P_0B0B = fastpt.IA_ta(P_mm0, P_window=P_window, C_window=C_window)
        P_A, P_Btype2, P_DEE, P_DBB = fastpt.IA_mix(P_mm0, P_window=P_window, C_window=C_window)

        # Arrays for GI spectra
        GI_z = []

        for z in z_values:
            P_mm = Pmm(z, k)

            # IA amplitudes
            A1 = -IA_pars[0] * C1 * rho_crit * Om0 / self.growth_factor(z)  * ((1 + z) / (1 + IA_pars[5]))**IA_pars[2]
            A2 = 5 * IA_pars[1] * C1 * rho_crit * Om0 / self.growth_factor(z)**2  * ((1 + z) / (1 + IA_pars[5]))**IA_pars[3]

            # GI power spectrum
            GI = A1 * P_mm + IA_pars[4] * A2 * (P_deltaE1 + P_deltaE2) + A2 * (P_A + P_Btype2)
            GI_z.append(GI)

        GI_array = np.array(GI_z)* self.h**3 # (Mpc/h)^3 -> Mpc^3
        # Interpolator for GI spectrum
        PGI = RegularGridInterpolator((z_values, self.k), GI_array, bounds_error=False, fill_value=None)

        return PGI

    def lensing_efficiency(self, nchi_interp, shear=0):
        """
        Compute the lensing efficiency kernel q(chi) for weak lensing.

        Parameters
        ----------
        nchi_interp : function
            Interpolator for source distribution in comoving distance space n(chi).
            Should be normalized so that ∫ n(chi) dchi = 1.
        shear : float, optional
            Multiplicative shear calibration bias (default 0 → no correction).

        Returns
        -------
        q : ndarray
            Lensing efficiency q(chi) evaluated on self.chi grid.
        """
        z = self.z
        chi = self.chi

        # Prefactor: (3/2) * (H0^2 * Ωm / c^2)
        prefactor = 1.5 * (self.H0**2) * self.OmegaM / self.c**2

        # n(chi)/chi and n(chi)
        f = nchi_interp(chi) / chi
        fchi = nchi_interp(chi)

        # Cumulative integrals from 0 → chi
        int_f = cumtrapz(f, chi, initial=0)
        int_fchi = cumtrapz(fchi, chi, initial=0)

        # Totals (integral up to chi_max)
        total_f = int_f[-1]
        total_fchi = int_fchi[-1]

        # Efficient way to compute ∫_{chi_i}^{chi_max} n(chi')(chi'-chi_i)/chi' dchi'
        integral_from_i = (total_fchi - int_fchi) - chi * (total_f - int_f)

        # Lensing efficiency kernel
        q = prefactor * chi * (1 + z) * integral_from_i

        # Apply shear calibration (q -> (1+m) q)
        return q * (1 + shear)
    
    def ia_power(self,l_bins, zl, zs, nz_lens, nz_source, zl_mean, galaxy_bias, nz_lens_stretch = 1,  nz_lens_shift = 0,  C1=5e-14,IA_pars = np.array([0.7,-1.36,-1.7,-2.5,1.0,0.62])):
        print(f"Working on IA power")
        chi = self.chi
        z = self.z
        Pia = self.get_Pia_interp(NLA=True,C1=C1,IA_pars = IA_pars)

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)
        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)


        cls = []
        for l in l_bins:
            kp = (l + 0.5) / chi            
            pts = np.vstack([z, kp]).T    
            Pia_chi = Pia(pts) * galaxy_bias

            integrand = nchi_lens_interp(chi) * nchi_source_interp(chi) * Pia_chi / chi**2
            cls.append(np.trapezoid(integrand, chi))

        return np.array(cls)

    def ia_mag_power(self,l_bins, zl, zs, nz_lens, nz_source, zl_mean, magnification_bias, nz_lens_stretch = 1,  nz_lens_shift = 0,  C1=5e-14,IA_pars = np.array([0.7,-1.36,-1.7,-2.5,1.0,0.62]),shear=0):
        print(f"Working on IAxmagnification power")
        chi = self.chi
        z = self.z
        Pia = self.get_Pia_interp(NLA=True,C1=C1,IA_pars = IA_pars)

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)
        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        q_lens = self.lensing_efficiency(nchi_interp = nchi_lens_interp,shear=0)

        cls = []
        for l in l_bins:
            kp = (l + 0.5) / chi            
            pts = np.vstack([z, kp]).T     
            Pia_chi = Pia(pts)

            integrand = magnification_bias * q_lens * nchi_source_interp(chi) * Pia_chi / chi**2
            cls.append( np.trapezoid(integrand, chi))

        return np.array(cls)

    def mag_power(self,l_bins, zl, zs, nz_lens, nz_source, zl_mean,magnification_bias, nz_lens_stretch = 1,  nz_lens_shift = 0, shear=0):
        print(f"Working on magnificatino power")
        chi = self.chi
        z = self.z
        Pmm = self.get_Pmm_interp()

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)
        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        q_source = self.lensing_efficiency(nchi_interp = nchi_source_interp,shear=shear)
        q_lens = self.lensing_efficiency(nchi_interp = nchi_lens_interp,shear=0)


        cls = []
        for l in l_bins:
            Pmm_chi = np.zeros_like(chi)  
            for i, chip in enumerate(chi):
                kp = (l + 0.5) / chip            
                Pmm_chi[i] = Pmm(z[i],kp)
            integrand = magnification_bias * q_lens * q_source * Pmm_chi / chi**2
            cls.append( np.trapezoid(integrand, chi))

        return np.array(cls)
 
    def lensing_power(self, l_bins, galaxy_bias, zl, zs, nz_lens, nz_source, zl_mean, nz_lens_stretch = 1,  nz_lens_shift = 0, shear=0):
        print(f"Working on lensing power")
        chi = self.chi
        z = self.z
        Pgm = self.get_Pgm_interp(galaxy_bias = galaxy_bias)

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)
        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        q_lens = self.lensing_efficiency(nchi_interp = nchi_source_interp,shear=shear)

        cls = []
        for l in l_bins:
            Pgm_chi = np.zeros_like(chi)
            kp = (l + 0.5) / chi            
            pts = np.vstack([z, kp]).T      # shape (nchi, 2)
            Pgm_chi = Pgm(pts)

            integrand = q_lens * nchi_lens_interp(chi) * Pgm_chi / chi**2
            cls.append( np.trapezoid(integrand, chi))

        return np.array(cls)
        
    def galaxy_auto_power(self, l_bins, galaxy_bias, zl, nz_lens,  zl_mean, nz_lens_stretch = 1,  nz_lens_shift = 0):
        print(f"Working on galaxy-galaxy power")
        chi = self.chi
        z = self.z
        Pgg = self.get_Pgg_interp(galaxy_bias = galaxy_bias)

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)

        cls = []
        for l in l_bins:
            Pgg_chi = np.zeros_like(chi)
            kp = (l + 0.5) / chi            
            pts = np.vstack([z, kp]).T      # shape (nchi, 2)
            Pgg_chi = Pgg(pts)

            integrand = nchi_lens_interp(chi)**2 * Pgg_chi / chi**2
            cls.append( np.trapezoid(integrand, chi))

        return np.array(cls)
    
    def lensing_auto_power(self, l_bins, zs, nz_source,  shear=0):
        print(f"Working on lensing convergence auto power")
        chi = self.chi
        z = self.z
        Pmm = self.get_Pmm_interp()

        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        q_lens = self.lensing_efficiency(nchi_interp = nchi_source_interp,shear=shear)

        cls = []
        for l in l_bins:
            Pmm_chi = np.zeros_like(chi)  
            for i, chip in enumerate(chi):
                kp = (l + 0.5) / chip            
                Pmm_chi[i] = Pmm(z[i],kp)

            integrand = q_lens **2 * Pmm_chi / chi**2
            cls.append( np.trapezoid(integrand, chi))

        return np.array(cls)