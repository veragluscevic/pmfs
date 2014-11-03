# This module contains 21-cm-cosmology-related functions, handeled in CGS units.

import numpy as np
from scipy.interpolate import UnivariateSpline as interpolate

import cosmo_functions as cf
reload(cf)
from constants import *
from globals import *

#set up kappas:
KAPPA = np.loadtxt(INPUTS_PATH + 'Kcoeffexpanded.txt')
TKAPPA = KAPPA[:,0]
KAPPAP = KAPPA[:,1]
KAPPAM = KAPPA[:,2]

val_kappam = interpolate(TKAPPA, KAPPAM, s=0)
val_kappap = interpolate(TKAPPA, KAPPAP, s=0)

def calc_simple_Jlya( z ):
    """ This returns Lyman-alpha flux as a function of redshift, Jlya(z) [# of photons/cm^2/sec/Hz/sr], as calculated using Chen and Miralda-Escude 2004 Eq. (16). """
    res = 1.65*1e-13 * ( 1 + z )**3 * ( Obaryonh2 / 0.02 )
    return res

def Tk_simple( z ):
    """This is the eyeballed kinetic temperature of the gas from the 21cmfast paper, for one of "extreme heating" models.
    It takes only z and returns Tk [K]."""

    k_l = -0.1872
    n_l = 4.6499 
    k_r = 0.0257
    n_r = 0.4994
    
    #zs = np.atleast_1d(z)
    #res = np.zeros(len(zs))
    if z  >= 19.5:
        n = n_r
        k = k_r
    else:
        n = n_l
        k = k_l
    res = 10**(k*z+n)
    #res[np.where(zs >= 19.5)] = 10**(k_r*zs[np.where(zs >= 19.5)] + n_r)
    #res[np.where(zs < 19.5)] = 10**(k_l*zs[np.where(zs >= 19.5)] + n_l)

    return res



def ones_x1s(z):
    """Neutral function as a function of redshift."""
    return z/z

def val_Tceff(Tk, Ts):
    """This returns Tc^{eff} in [K], from Eq. (42) in Hirata 2005, as a function of Tk[K] and Ts[K]."""

    res = 1. / ( 1./Tk + 0.405535/Tk*(1./Ts - 1./Tk) )
    return res


def Ts_Hirata05( z, Tk=10., Jlya=1.7e-9, x1s=1. ):
    """This takes z, Tk[K], and Jlya[# of photons/cm^2/sec/Hz/sr],
     and returns spin temperature in [K], according to Eq (39) of Hirata 2005."""
    niterations = 50
    
    Tg = Tcmb * ( 1 + z )
    Ts = Tg*1.

    xc = val_xc_tilde(z, Tk=Tk, Tg=Tg)
     
    for j in np.arange(niterations):
        
        Salpha = cf.val_Salpha(Ts, Tk, z, x1s, 0)
        xalpha = val_xalpha_tilde(Salpha=Salpha, Jlya=Jlya, Tg=Tg)
        Tceff = val_Tceff(Tk, Ts)

        Ts = ( 1. + xalpha + xc ) / ( 1./Tg + xalpha/Tceff + xc/Tk )

    return Ts


def val_xc_tilde(z, Tk=10., Tg=57.23):
    """For calculation of Ts, quantity from Hirata paper.
    This takes z, Tk[K], T_gamma at a given z also in [K], and returns xc [unitless]."""

    kappap = val_kappap(Tk)
    kappam = val_kappam(Tk)
    res = 0.5*(kappap + kappam)*cf.nH(z,0)*Tstar/(A*Tg)
    return res

def val_xc(z, Tk=10., Tg=57.23):
    """Quantity from our paper. 
    This takes z, Tk[K], T_gamma at a given z also in [K], and returns xc [unitless]."""

    kappap = val_kappap(Tk)
    kappam = val_kappam(Tk)
    res =  2.*( val_kappap(Tk) + val_kappam(Tk) ) * cf.nH(z,0) * Tstar / ( A * Tg )  
    return res

def val_xalpha_tilde(Salpha=0.6, Jlya=1.7e-9, Tg=57.23):
    """For calculation of Ts, quantity from Hirata paper.
    This takes Salpha[unitless], Jlya[# of photons/cm^2/sec/Hz/sr], and T_gamma[K] at z, and returns xalpha[unitless]"""

    res =  8./9.*np.pi * lambdaLya**2. * gamma * Tstar * Salpha * Jlya / ( A * Tg )
    return res


def val_xalpha(Salpha=0.6, Jlya=1.7e-9, Tg=57.23):
    """Quantity from our paper. This takes Salpha[unitless], Jlya[# of photons/cm^2/sec/Hz/sr], and T_gamma[K] at z, and returns xalpha[unitless]"""

    res =  3.607*np.pi * lambdaLya**2. * gamma * Tstar * Salpha * Jlya / ( A * Tg )
    return res

