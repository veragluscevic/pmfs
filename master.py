import matplotlib as plt
import pylab as pl
import numpy as np
from scipy.interpolate import UnivariateSpline as interpolate
import cosmolopy.distance as cd

import reion_functions as rf
import cosmo_functions as cf
import geometric_functions as gf
import pmfs_transfer as pt
import pmfs_fisher as pf
import camb_setup as cs

from constants import *
from globals import *

#set up z-range of interest:
zmin = 10
zmax = 36

#set up kappas:
KAPPA = np.loadtxt(MAIN_PATH + 'inputs/Kcoeffexpanded.txt')
TKAPPA = KAPPA[:,0]
KAPPAP = KAPPA[:,1]
KAPPAM = KAPPA[:,2]

val_kappam = interpolate(TKAPPA, KAPPAM, s=0)
val_kappap = interpolate(TKAPPA, KAPPAP, s=0)

#choose Jlya function:
val_Jlya = rf.calc_simple_Jlya

#set up Tk:
zs = np.arange(zmin, zmax, 1)
Tk_array = rf.Tk_simple(zs)
val_Tk = interpolate(zs, Tk_array, s=0)

#set up Ts:
#this uses Eq (39) of Hirata 2005.

Ts_array = np.zeros(len(zs))
niterations = 50
for i,z in enumerate(zs):
    
    Tg = Tcmb * ( 1 + z )
    Ts = Tg*1.

    Tk = val_Tk(z)
    kappap = val_kappap(Tk)
    kappam = val_kappam(Tk)
    #xc = (1.5*kappap + 2.*kappam)*cf.nH(z,0)*Tstar/(A*Tg)
    xc = 0.5*(kappap + kappam)*cf.nH(z,0)*Tstar/(A*Tg)
    
    x1s = rf.val_x1s(z)
     
    for j in np.arange(niterations):
        
        Salpha = cf.val_Salpha(Ts, Tk, z, x1s, 0)
        xalpha = 8./9.*np.pi*lambdaLya**2.*gamma*Tstar*Salpha*val_Jlya(z)/(A*Tg)
        Tceff = rf.val_Tceff(Tk, Ts)

        Ts = ( 1. + xalpha + xc ) / ( 1./Tg + xalpha/Tceff + xc/Tk )
        
    Ts_array[i] = Ts

val_Ts = interpolate(zs, Ts_array, s=0)


#set up dAs:
#Note: val_dA returns angular-diameter distance in [cm comoving].
cosmo = {'omega_M_0' : Omatter, 'omega_lambda_0' : Olambda, 'h' : h0}
cosmo = cd.set_omega_k_0( cosmo )
grid_dA = np.zeros( len( zs ) )
for i,z in enumerate( zs ):
    grid_dA[i] = cd.angular_diameter_distance( z, **cosmo ) * ( 1 + z ) * Mpc_in_cm
val_dA = interpolate( zs, grid_dA, s=0 )     

#set up power spectra for density:
#if the CAMB outputs are not ready, you should do this from command line: cs.make_camb_ini(zs), and then run CAMB for parameters.ini; it might take a while.
val_Pdelta = cf.read_Pdelta_fn()

#set up the choice of uv coverage:
val_nk = pf.one_over_u_nk
#use: (k, Ntot=Ntot, lambda_z=lambda_z, dA=dA, sin_thetak_n=sin_thetak_n, Lmax=Lmax, Lmin=Lmin)




#plot stuff:
#pl.figure()
#pl.semilogy(zs,Ts_array)
#pl.ylabel('Ts')
#pl.xlabel('z')
#pl.figure()
#pl.semilogy(zs,Tk_array)
#pl.ylabel('Tk')
#pl.xlabel('z')

#this is how to calculate x's et al:
    #Tg = Tcmb * ( 1 + z )
    #xalpha = 3.607*np.pi * lambdaLya**2. * gamma * Tstar * Salpha * Jlya / ( A * Tg )
    #xc = 2.*( kappap + kappam ) * cf.nH(z,delta) * Tstar / ( A * Tg )
    #xB = ge * muB * Tstar * B / ( 2.*hbar * A * Tg )
