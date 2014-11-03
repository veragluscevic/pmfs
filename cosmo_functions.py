#This module contains the most general cosmological functions, handeled in CGS units.

import numpy as np
from scipy.interpolate import LinearNDInterpolator as interpnd
import cosmolopy.distance as cd
from constants import *
from globals import *

def cosmo_E(z):
    """This takes redshift and returns the usual sqrt-part of the integrand used for calculating comoving distance, or Hubble rate.
    Result is unitless. """
    res =  ( Omatter*(1. + z)**3 + Olambda + 1./0.6711**2 * 2.47e-5*( 1. + Neff*( 7./8. )*( 4./11. )**( 4./3. ) )*(1. + z)**4 )**0.5
    return res

def H(z):
    """ This function calculates the Hubble rate as a function of redshift z. Result is in [1/sec]."""
    res = H0 * cosmo_E(z)
    return res


def nH(z, delta):
    """This function takes redshift z and overdensity delta [unitless] to calculate
    local hydrogen number denstity [1/cm^3], given He fraction, Omega_baryon*h^2,
    and critical density, all given in constants.py. """
    
    res = Obaryonh2 * ( 1. - yHe ) * ( 1.+ z )**3 / mH * rho_critical * ( 1. + delta )
    return res   



def val_dA( z ):
    """This returns angular-diameter distance in [cm comoving]. It only takes z.""" 
    cosmo = {'omega_M_0' : Omatter, 'omega_lambda_0' : Olambda, 'h' : h0}
    cosmo = cd.set_omega_k_0( cosmo )

    res =  cd.angular_diameter_distance( z, **cosmo ) * Mpc_in_cm * ( 1 + z )
    return res

def gpdepth(z,x1s,delta):
    """ Given neutral fraction x1s, redshift, and overdensity delta [unitless],
    this returns Gunn-Peterson optical depth [unitless], according to Hirata 2005 Eq. (35).
    Note that it uses physical constants from constants.py."""
    
    res = 3. * nH( z, delta ) * x1s * lambdaLya**3 * gamma / ( 2. * H(z) )
    return res

def val_Salpha(Ts, Tk, z, x1s, delta):
    """ Given neutral fraction x1s, redshift, overdensity [unitless], kinetic temperature Tk [K],
    and spin temperature Ts [K], this returns tilde S_alpha [unitless] function from Hirata 2005 Eq. (40)."""
    
    gpd = gpdepth(z,x1s,delta)
    zeta = (1e-7*gpd)**(1./3.)/Tk**(2./3.)
    res = ((1.0 - 0.0631789/Tk + 0.115995/(Tk**2) 
            - 0.401403/(Tk*Ts) + 0.336463/(Ts*Tk**2))
            /(1.0 + 2.98394*zeta + 1.53583*zeta**2 + 3.85289*zeta**3))
    return res



def read_Pdelta(root=MAIN_PATH+'matter_power/',**kwargs):
    """This is an interpolation in k and z, for array P_delta(k,z) output by CAMB.
    Input parameters for this function are the root where the power spectra arrays are stored, plus kwargs.
    This returns result in [cm^3 comoving]. It takes z, and k[1/Mpc comoving].
    Note1: Array of redshifts corresponding to the grid P must be stored in the location root/zs.txt.
    Note2: This also relies on particular file naming (regulated by CAMB), which can be adjusted in the function."""

    
    zs = np.loadtxt('%s/zs.txt' % root)
    all_Pdeltas = []
    for i,z in enumerate(zs):
        ks,Ps = np.loadtxt('%s/_matterpower_%i.dat' % (root,i),unpack=True)
        ks *= h0 
        Ps *= (Mpc_in_cm/h0)**3
        all_Pdeltas.append(Ps)
    all_Pdeltas = np.array(all_Pdeltas).T
    Zs,Ks = np.meshgrid(zs,ks)
    points = np.array([Zs.ravel(),Ks.ravel()]).T
    return interpnd(points,all_Pdeltas.ravel(),**kwargs)


val_Pdelta = read_Pdelta()
