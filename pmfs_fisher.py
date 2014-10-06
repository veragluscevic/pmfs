#This module contains functions necessary for Fisher analysis for PMFs.

import numpy as np
import cosmo_functions as cf
import pmfs_transfer as pt
from constants import *

#set up Pdelta:
val_Pdelta = cf.read_Pdelta_fn()



def P21_N(z, dA=3.1695653382881036e+28, H_z=1.1905643664441961e-16, sin_thetak_n=0.5, Tsys=1000, tobs=10*86400., Ae=3500**2, Lmax=1200000, Lmin=1000, Omega_survey=0.3, nk=0.75, lambda_z=443.1):
    """ This takes k [1/cm comoving], thetak [rad], and instrumental parameters:
          observation time tobs[sec], angular size of the survey [sr], system temperature Tsys[K],
          effective area of an antenna Ae[cm^2], minimum and maximum baselines, both in [cm],
          sin of angle between k and line of sight, and the number density of baselines per k mode nk[unitless].
          It returns noise power in [(erg/sr)^2 cm^3]."""
    
    res = dA**2 * c * (1+z)**2 * lambda_z**2 * Tsys**2 * Omega_survey / ( Ae * nu21 * tobs *nk * H_z )
    return res

#calculate survey-volume element at z as Vpath_factor*dz = dVpatch:
def Vpatch_factor(z, dA=3.1695653382881036e+28, H_z=1.1905643664441961e-16, Omega_survey=0.3):
    """This is the volume element in the Fisher integral, *for a uniform magnetic field* B=B0*(1+z)^2. Vpath_factor*dz = dVpatch.
    It takes z, dA[cm comoving], H_z[1/sec], and Omega_survey[sr], and returns result in CGS units, [1/(cm*sr)]."""
    res = c / H_z * dA**2 * Omega_survey
    return res


def one_over_u_nk(k, Ntot=523776, lambda_z=443.1, dA=10271.861709829398, sin_thetak_n=0.5, Lmax=1200000, Lmin=1000):
    """This gives the UV coverage density in baselines per k mode, which in UV space goes as ~1/u; the result is unitless.
    It takes k[1/cm comoving], total number of antenna pairs (baselines) Ntot, 21cm wavelength at z,
    ang. diam. distance in [comoving cm], sin of vector k wrt line of sight, maximum and minimum baselines [both in cm]. """

    res = 2. * Ntot * lambda_z / ( k * dA * sin_thetak_n * ( Lmax - Lmin ) )
    return res



def Fisher_integrand(z, k_in_Mpc, thetak=0.3, phik=0.2, thetan=0.1, phin=0., Ts=11., xalpha=34.247221, xc=0.004176,
                     Tsys=1000, tobs=10*86400., Ae=(3500)**2, Lmax=100000, Lmin=100, N_ant=1024, x1s=1., Omega_survey=0.3,
                     dA=3.16e+28, H_z=1.19e-16, lambda_z=443.1, nk=0.75, Tg=57.23508, xBcoeff=3.65092e18):
    """This takes k[1/Mpc comoving]. Result is returned in CGS units, [???]."""

    k = k_in_Mpc/Mpc_in_cm

    thetak_n = np.arccos(np.sin(thetak)*np.cos(phik)*np.sin(thetan)*np.cos(phin) +
                          np.sin(thetak)*np.sin(phik)*np.sin(thetan)*np.sin(phin) +
                          np.cos(thetak)*np.cos(thetan))

    sin_thetak_n = np.sin(thetak_n)
    if sin_thetak_n == 0.:
        return 0.
    print thetak_n
    Vpatch = Vpatch_factor( z, dA=dA, H_z=H_z, Omega_survey=Omega_survey )
    
    Pnoise = P21_N( dA=dA, H_z=H_z, sin_thetak_n=sin_thetak_n, z=z, Tsys=Tsys, tobs=tobs, Ae=Ae, 
                   Lmax=Lmax, Lmin=Lmin, Omega_survey=Omega_survey, lambda_z=lambda_z, nk=nk )
    if np.isnan( Pnoise ):
        raise ValueError( 'Pnoise is nan.' )

    Pdelta = val_Pdelta( z, k_in_Mpc ) * Mpc_in_cm**3 
    
    G = pt.calc_G(thetak=thetak, phik=phik, thetan=thetan, phin=phin, 
            Ts=Ts, Tg=Tg, z=z, verbose=False, xalpha=xalpha, xc=xc, xB=0., x1s=x1s)
  
    dGdB = pt.calc_dGdB(thetak=thetak, phik=phik, thetan=thetan, phin=phin, 
            Ts=Ts, Tg=Tg, z=z, verbose=False, xalpha=xalpha, xc=xc, xBcoeff=xBcoeff, x1s=x1s)
    
    Psignal = Pdelta*G**2
    
    Numerator = (2.*G*dGdB*Pdelta)**2
    Denominator = 2.*(Psignal + Pnoise)**2

    print '@(z, k[1/Mpc])=(%i, %i):  Psignal=%e, Pnoise=%e\n' % (z, k_in_Mpc, Psignal, Pnoise)
    res = k**2*np.sin(thetak)* Vpatch * Numerator/Denominator/(2.*np.pi)**3   #*(1+z)**4 #is this NOT there, given the definition of dGdB??
    print Numerator
    print Denominator
    print G,dGdB


    return res
    
