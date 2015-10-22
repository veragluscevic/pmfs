#!/usr/bin/env python

import matplotlib
if __name__=='__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
import glob
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)
mpl.rcParams['xtick.major.size']=12
mpl.rcParams['ytick.major.size']=12
mpl.rcParams['xtick.minor.size']=8
mpl.rcParams['ytick.minor.size']=8
mpl.rcParams['xtick.labelsize']=22
mpl.rcParams['ytick.labelsize']=22

import numpy as np

from constants import *
from globals import *
import cosmo_functions as cf
reload(cf)
import pmfs_transfer as pt
reload(pt)
import reion_functions as rf
reload(rf)
import fisher as f
reload(f)
from scipy.optimize import fsolve, root
import healpy as hp
from numba import jit

from scipy.interpolate import UnivariateSpline as interpolate
from scipy.ndimage import gaussian_filter1d

@jit
def vis_xT(zmin=15,zmax=35, nzs=100, B0=1e-18):

    zs = np.linspace(zmin,zmax,nzs)
    Ts = []; Tg = []; Tk = []; Jlya = []
    xc = []; xB = []; xalpha = []
    for i,z in enumerate(zs):
        B = B0/(1+z)**2
        Ts.append(rf.Ts_21cmfast_interp( z ))
        Tg.append(f.val_Tg( z ))
        Tk.append(f.val_Tk( z ))
        Jlya.append(rf.Jlya_21cmfast_interp( z ))
        Salpha = cf.val_Salpha(Ts[i], Tk[i], z, 1., 0) 
        xalpha.append(rf.val_xalpha( Salpha=Salpha, Jlya=Jlya[i], Tg=Tg[i] ))
        xc.append(rf.val_xc(z, Tk=Tk[i], Tg=Tg[i]))
        xBcoeff = ge * muB * Tstar / ( 2.*hbar * A * Tg[i] )
        xB.append(B*xBcoeff)

    #print xB, xalpha, xc
    plt.figure()
    plt.plot(zs,xc,lw=4,color='g',label='$x_c$')
    #plt.plot(zs,xalpha,lw=4,color='b',label=r'$x_{\alpha}$')
    plt.plot(zs,xB,lw=4,color='k',label='$x_B$')
    plt.legend(fontsize=22)

    plt.figure()
    plt.plot(zs,Ts,'--',lw=4,color='r',label='$T_S$ [K]')
    plt.plot(zs,Tg,':',lw=4,color='b',label='$T_{CMB}$ [K]')
    plt.plot(zs,Tk,'-.',lw=4,color='g',label='$T_K$ [K]')
    plt.legend(fontsize=22)

    plt.figure()
    plt.plot(zs,Jlya,lw=4,color='blue')

def grid_DeltaL(mode='B0',t_yr=1., 
                Jmode='default',Omega=1.,
                fontsize=22,
                ymax=None,ymin=None,
                xlabel='\Delta L [km]',
                root=RESULTS_PATH,
                color='Maroon',
                save=True,
                smooth=True,
                s=3,
                cieling_zs=[25,26,27,28,29,30]):

    """Master plotter"""
    
    DeltaLs = np.load(root + 'DeltaLs_{}_{}_tyr_{:.2f}_Omega_{:.2f}.npy'.format(mode,Jmode,t_yr,Omega))


    sigmas = []
    ds = []
    for j,d in enumerate(DeltaLs):
        infile = root + '{}_{}_tyr_{:.2f}_DeltaL_{:.2f}_Omega_{:.2f}.txt'.format(mode,
                                                                    Jmode,
                                                                    t_yr,
                                                                    d,Omega)

        if os.path.exists(infile):
            data = np.loadtxt(infile, skiprows=1, usecols=(0,))
            sigmas.append(data)
            ds.append(d)
        else:
            print('Did not find: {}'.format(infile))


    ds = np.array(ds)
    sigmas = np.array(sigmas)
    #print(ds,sigmas)
    
    plt.figure()
    ax = plt.gca()
    fig = plt.gcf()

    if mode=='B0':
        ylabel = ax.set_ylabel('B [Gauss]',fontsize=fontsize)
    if mode=='zeta':
        ylabel = ax.set_ylabel(r'$\xi$',fontsize=fontsize)

    if mode=='SI':
        ylabel = ax.set_ylabel(r'(SI amplitude)$^{1/2}$ [Gauss]',fontsize=fontsize)

    if smooth:
        x = gaussian_filter1d(ds, s)
        y = gaussian_filter1d(sigmas, s)
    else:
        x = ds
        y = sigmas
    plt.semilogy(x, y, lw=3, color=color)
    plt.ylim(ymin=ymin,ymax=ymax)
    xlabel = ax.set_xlabel(r'$\Delta$ L [km]',fontsize=fontsize)

    zs, Bsat = saturationB()
    for z in cieling_zs:
        ind = np.argmin(np.abs(zs-z))
        B = Bsat[ind]
        plt.semilogy(x,np.ones(len(x))*B,lw=2,label=z)

    plt.legend()

    if save:
        plt.savefig(root + '{}_vs_deltas.pdf'.format(mode), 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')



##########################################
##########################################
##########################################
def saturationB(Bguess0=1e-19,
                nzs=100,
                zmin=15, zmax=35,
                thetak=0.1,phik=0.83,
                val_nk=f.FFTT_nk,
                val_Ts=rf.Ts_21cmfast_interp, 
                val_Tk=rf.Tk_21cmfast_interp, 
                val_Jlya=rf.Jlya_21cmfast_interp, 
                val_Tg=rf.Tg_21cmfast_interp,
                phin=0., thetan=np.pi/2.,
                make_plot=False,
                **rootkwargs):
    """This computes the mag. field strength at saturation, at a given z.
    """
    zs = np.linspace(zmin, zmax,nzs)
    Bevol = np.zeros(nzs)
    Bsat = np.zeros(nzs)
    Gref = np.zeros(nzs)
    Gref_Binf = np.zeros(nzs)
    for i,z in enumerate(zs):
        
        H_z = cf.H( z )
        Ts = val_Ts( z )
        Tg = val_Tg( z )
        Tk = val_Tk( z )
        Jlya = val_Jlya( z )        
        Salpha = cf.val_Salpha(Ts, Tk, z, 1., 0) 
        xalpha = rf.val_xalpha( Salpha=Salpha, Jlya=Jlya, Tg=Tg )
        xc = rf.val_xc(z, Tk=Tk, Tg=Tg)
        xBcoeff = ge * muB * Tstar / ( 2.*hbar * A * Tg ) 
        x1s = 1.

        xB=xBcoeff*Bguess0
        Gref[i] = pt.calc_G(thetak=thetak, phik=phik, 
                  thetan=thetan, phin=phin,
                  Ts=Ts, Tg=Tg, z=z, 
                  verbose=False, 
                  xalpha=xalpha, xc=xc, xB=xB, x1s=1.)

        Gref_Binf[i] = pt.calc_G_Binfinity(thetak=thetak, phik=phik, 
                                      thetan=thetan, phin=phin,
                                      Ts=Ts, Tg=Tg, z=z, 
                                      verbose=False, 
                                      xalpha=xalpha, xc=xc, 
                                      x1s=1.)

        res = root(evaluate_GminusGinf, Bguess0, args=(Gref_Binf[i],xBcoeff,thetak,phik,thetan,phin,Ts,Tg,z,xalpha,xc,x1s),method='lm',options=dict(ftol=1e-13, eps=1e-6))
       
        Bsat[i] = res.x[0]/(1+z)**2
        
    if make_plot:
        plt.figure()
        plt.semilogy(zs,Bsat,lw=4,color='blue')
        plt.xlabel('z',fontsize=22)
        plt.ylabel('saturation B [Gauss]',fontsize=22)
    
    return zs,Bsat

def evaluate_GminusGinf(B,
                        G_Binfinity=1e-22,
                        xBcoeff=1,
                        thetak=0.1, phik=0.83, 
                        thetan=0., phin=np.pi/2.,
                        Ts=3., Tg=3., z=20, 
                        xalpha=1, xc=1, x1s=1.):
    """This is used by saturationB.
    """

    xB = B * xBcoeff
    
    G = pt.calc_G(thetak=thetak, phik=phik, 
                  thetan=thetan, phin=phin,
                  Ts=Ts, Tg=Tg, z=z, 
                  verbose=False, 
                  xalpha=xalpha, xc=xc, xB=xB, x1s=1.)
    return G


    
