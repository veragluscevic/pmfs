#!/usr/bin/env python
import matplotlib as plt
import pylab as pl
import numpy as np
from scipy.interpolate import UnivariateSpline as interpolate
import os,os.path,shutil
import cosmolopy.distance as cd
 
import reion_functions as rf
reload(rf)
import cosmo_functions as cf
reload(cf)
import geometric_functions as gf
reload(gf)
import pmfs_transfer as pt
reload(pt)
import pmfs_fisher as pf
reload(pf)
import camb_setup as cs
reload(cs)

from constants import *
from globals import *

#mode = 'zeta'
mode = 'B0'

showres=True
force_grid = False
make_plots = False

file_label = 'may7'
grid_path = RESULTS_PATH + file_label + '_' + mode + '/'

#noise-calculation parameters:
tobs = 365.*86400 #observation time in sec.
DeltaL = 200000. #length of a square-shaped antenna in cm.
Ae = None #1000**2 #3500**2 #effective area of a single dish in cm.
Lmax = None #1200000 #maximum baseline in cm.
Lmin = None #1000 #minimum baseline in cm. 
N_ant = None #1024 #number of antennas.
kminmin = 0.001 #minimum k to be analyzed in [1/Mpc comoving]
kmaxmax = 100 #maximum k to be analyzed in [1/Mpc comoving]
Omega_patch = 2e-4 #0.1 #area coverage of the survey in sr.
Omega_survey = 1.



#set up z-range of interest:
z_lims = (10,35)# (15,29.9) 
Nzs = int(z_lims[1]-z_lims[0])
Nks = 100



#choose Jlya function:
#val_Jlya = rf.calc_simple_Jlya
val_Jlya = rf.Jlya_21cmfast_interp

#choose sky temperature function:
val_Tsys = pf.Tsys_Mao2008 #pf.Tsys_zero #pf.Tsys_simple

val_Tg = rf.Tg_21cmfast_interp

#set up kinetic-temperature function:
val_Tk = rf.Tk_21cmfast_interp
"""
#this is old piece of code, for analytic JLya which does not work yet:
zs = np.arange(z_lims[0], z_lims[1], 1)
Tk_array = np.zeros(len(zs))
Jlya_array = val_Jlya(zs)
for i,z in enumerate(zs):
    Tk_array[i] = rf.Tk_1000K(z) #rf.Tk_simple(z)
Tk_simple_interp = interpolate(zs, Tk_array, s=0)
val_Tk = Tk_simple_interp
"""
#set up spin-temperature function:
#val_Ts = rf.Ts_Hirata05
val_Ts = rf.Ts_21cmfast_interp

#set up the choice of uv coverage:
val_nk = pf.FFTT_nk #pf.one_over_u2_nk

#set evolution of ionized fraction:
val_x1s = rf.ones_x1s



#set up power spectra for density:
#if the CAMB outputs are not ready, you should do this from command line:
# cs.make_camb_ini(zs), and then run CAMB for parameters.ini; it might take a while.
#also, make sure that the fineness of z, and k grid for CAMB corresponds to what is set up
#at the beginning of this master file.


##write Fisher grid that can then be read from files, and integrated:
##note: this example corresponds to SKAII:

if force_grid or (not(os.path.exists(grid_path))):
    if(os.path.exists(grid_path)):
        shutil.rmtree(grid_path)
    os.makedirs(grid_path)
    make_plots = True
    if mode == 'B0':
        pf.write_Fisher_grid(grid_path, val_nk=val_nk, val_Ts=val_Ts, val_Tk=val_Tk, val_x1s=val_x1s,
                            z_lims=z_lims, Nzs=Nzs, Nks=Nks, Nthetak=21, Nphik=22, phin=0., thetan=np.pi/2.,
                            val_Tsys=val_Tsys, tobs=tobs, Ae=Ae, Lmax=Lmax, Lmin=Lmin, N_ant=N_ant, Omega_patch=Omega_patch,
                            kminmin=kminmin, kmaxmax=kmaxmax,DeltaL=DeltaL)
    if mode == 'zeta':
        pf.write_Fisher_grid_zeta(grid_path, val_nk=val_nk, val_Ts=val_Ts, val_Tk=val_Tk, val_x1s=val_x1s,
                            z_lims=z_lims, Nzs=Nzs, Nks=Nks, Nthetak=21, Nphik=22, phin=0., thetan=np.pi/2.,
                            val_Tsys=val_Tsys, tobs=tobs, Ae=Ae, Lmax=Lmax, Lmin=Lmin, N_ant=N_ant, Omega_patch=Omega_patch,
                            kminmin=kminmin, kmaxmax=kmaxmax)

##now read the fisher grid, and integrate it in all dimensions:
fisher_grid = np.load(grid_path + 'fisher_grid.npy')
zs = np.load(grid_path + 'z_grid.npy')
phiks = np.load(grid_path + 'phik_grid.npy')
thetaks = np.load(grid_path + 'thetak_grid.npy')
ks = np.load(grid_path + 'k_grid.npy')

if showres:
    result = pf.trapznd(fisher_grid,zs,phiks,thetaks,ks)
    alpha_survey = (Omega_survey)**0.5 
    result_all_survey = result / Omega_patch * np.pi * (alpha_survey + np.cos(alpha_survey)*np.sin(alpha_survey))

    ##print the final result, in Gauss:
    #print 1./result**0.5
    print 1./result_all_survey**0.5
    

if make_plots:
    zs = np.arange(z_lims[0], z_lims[1], 1)
    zsa = np.linspace(0.1,29)
    Bval = 1e-21
    
    Tg_array = Tcmb*(1+zs)
    Salpha_array = np.zeros(len(zs))
    xcs_array = np.zeros(len(zs))
    xas_array = np.zeros(len(zs))
    xBs_array = np.zeros(len(zs))
    xBcoeff=3.65092e18
    Bval=1e-21
    
    Jlya_array = np.zeros(len(zs))
    x1s_array = val_x1s(zs)
    Tk_array = np.zeros(len(zs))
    Ts_array = np.zeros(len(zs))
    Jlya_array = val_Jlya(zs)
    x1s_array = val_x1s(zs)
    for i,z in enumerate(zs):
        Tk_array[i] = val_Tk(z)
        Ts_array[i] = val_Ts(z) #val_Ts( z, Tk=Tk_array[i], Jlya=Jlya_array[i], x1s=x1s_array[i] )
        Jlya_array[i] = val_Jlya(z)#rf.calc_simple_Jlya(z)
        Salpha_array[i] = cf.val_Salpha(Ts_array[i], Tk_array[i], z, x1s_array[i], 0)
        
        xas_array[i] = rf.val_xalpha(Salpha=Salpha_array[i], Jlya=Jlya_array[i], Tg=Tg_array[i])
        xcs_array[i] = rf.val_xc(z, Tk=Tk_array[i], Tg=Tg_array[i])
        xBs_array[i] = ge * muB * Tstar / ( 2.*hbar * A * Tg_array[i] )*Bval*(1 + z)**2

    pl.figure()
    pl.semilogy(zs,Tk_array, '--', color='r', label='$T_K$')
    pl.semilogy(zs,Ts_array, color='k', label='$T_S$')
    pl.semilogy(zs,Tg_array,':',color='g', label='$T_{\gamma}$')
    pl.legend(loc='lower right')
    pl.title('Temperature [K]')
    pl.xlabel('z')
    pl.savefig(RESULTS_PATH + 'temperatures_' + file_label + '.pdf')
    #print Tk_array,Ts_array

    pl.figure()
    pl.semilogy(zs,xcs_array, '--', color='r', label='$x_c$')
    pl.semilogy(zs,xas_array, color='b', label=r'$x_{\alpha}$')
    pl.semilogy(zs,xBs_array, '.-', color='k', label='$x_B(B_0=%.0eG)$' % Bval)
    pl.legend()
    pl.title('coupling strengths')
    pl.xlabel('z')
    pl.savefig(RESULTS_PATH + 'xs_' + file_label + '.pdf')


    pl.figure()
    #pl.semilogy(zs, Jlya_array,lw=3,color='k')
    pl.plot(zs, Jlya_array,lw=3,color='k')
    pl.xlabel('z')
    pl.ylabel(r'$J_{Ly\alpha} [cm^{-2}sec^{-1}Hz^{-1}sr^{-1}]$')
    pl.title('Evolution of Lyman-$\\alpha$ flux')
    pl.savefig(RESULTS_PATH + 'Lya_' + file_label + '.pdf')

    
    pl.figure()
    pl.plot(zsa, val_Tsys(zsa),lw=3,color='k')
    pl.xlabel('z')
    pl.ylabel(r'$T_{sys}$ [K]')
    pl.title('Sky temperature at redshifted 21cm')
    pl.savefig(RESULTS_PATH + 'Tsys_' + file_label + '.pdf')
    """
    pl.figure()
    pl.plot(zsa, cf.val_dA(zsa)/Mpc_in_cm)
    pl.xlabel('z')
    pl.ylabel('[Mpc comoving]')
    pl.title('angular diameter distance')
    pl.savefig(RESULTS_PATH + 'dA_'  + file_label + '.pdf')
    """
    pl.figure()
    sample_zs = np.array([15, 20, 25, 30])
    sample_ks = np.logspace(np.log10(0.001), np.log10(99), 100)
    Pdelta = np.zeros((len(sample_zs),len(sample_ks)))
    for i,z in enumerate(sample_zs):    
        for j,k in enumerate(sample_ks):
            Pdelta[i,j] = cf.val_Pdelta(z,k)
    for i,z in enumerate(sample_zs):
        pl.loglog(sample_ks,Pdelta[i],label='z=%i' % z)
    pl.xlabel('k [1/Mpc]')
    pl.ylabel(r'$P_{\delta\delta}[cm^3]$')
    pl.legend()
    pl.savefig(RESULTS_PATH + 'Pdelta_' + file_label + '.pdf')


