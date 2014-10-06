# This module contains 21-cm-cosmology-related functions, handeled in CGS units.

import numpy as np
from constants import *

def calc_simple_Jlya(z):
    """ This returns Lyman-alpha flux as a function of redshift, Jlya(z) [# of photons/cm^2/sec/Hz/sr], as calculated using Chen and Miralda-Escude 2004 Eq. (16). """
    res = 1.65*1e-13 * ( 1 + z )**3 * ( Obaryonh2 / 0.02 )
    return res

def Tk_simple(zs):
    """This is the eyeballed kinetic temperature of the gas from the 21cmfast paper, for one of "extreme heating" models."""
    
    k_l = -0.1872
    n_l = 4.6499
    k_r = 0.0257
    n_r = 0.4994
    
    zs = np.atleast_1d(zs)
    res = np.zeros(len(zs))
    for i,z in enumerate(zs):
        if z >= 19.5:
            k = k_r
            n = n_r
        else:
            k = k_l
            n = n_l
        res[i] = 10**(k*z + n)

    return res


def val_x1s(z):
    """Neutral function as a function of redshift."""
    return 1.

def val_Tceff(Tk, Ts):
    """This returns Tc^{eff} in K, from Eq. (42) in Hirata 2005, as a function of Tk[K] and Ts[K]."""

    res = 1. / ( 1./Tk + 0.405535/Tk*(1./Ts - 1./Tk) )
    return res

