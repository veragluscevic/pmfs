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
def vis_xT(zmin=15,zmax=35, nzs=100,
           fontsize=24,root=RESULTS_PATH):

    B0=1e-16
    #powB = np.log10(B0)
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

    plt.figure()
    ax = plt.gca()
    xlabel = ax.set_xlabel('z',fontsize=fontsize)
    ylabel = ax.set_ylabel('')
    plt.semilogy(zs,xc,lw=4,color='g',label='$x_c$')
    plt.semilogy(zs,xalpha,lw=4,color='b',label=r'$x_{\alpha}$')
    plt.semilogy(zs,xB,lw=4,color='k',label=r'$x_B$ ($10^{-16}$ G)')
    plt.legend(fontsize=fontsize,frameon=False,loc='upper right')
    plt.xlim(xmin=zmin,xmax=zmax)
    plt.savefig(RESULTS_PATH+'xs.pdf', 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')

    plt.figure()
    ax = plt.gca()
    xlabel = ax.set_xlabel('z',fontsize=fontsize)
    ylabel = ax.set_ylabel('T [K]',fontsize=fontsize)
    plt.plot(zs,Ts,lw=4,color='k',label='$T_S$')
    plt.plot(zs,Tg,'--',lw=4,color='b',label='$T_{CMB}$')
    plt.plot(zs,Tk,'-',lw=4,color='g',label='$T_K$')
    plt.legend(fontsize=fontsize,frameon=False,loc='upper left')
    plt.xlim(xmin=zmin,xmax=zmax)
    plt.savefig(RESULTS_PATH+'Ts.pdf', 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')

    plt.figure()
    ax = plt.gca()
    xlabel = ax.set_xlabel('z',fontsize=fontsize)
    ylabel = ax.set_ylabel(r'$J_{Ly\alpha}$ [$cm^{-2} sec^{-1} Hz^{-1}sr^{-1}$]',fontsize=fontsize)
    plt.semilogy(zs,Jlya,lw=4,color='DarkBlue')
    plt.xlim(xmin=zmin,xmax=zmax)
    plt.savefig(RESULTS_PATH+'Jlya.pdf', 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')

def grid_DeltaL(mode='B0',t_yr=1., 
                Jmode='default',Omega=1.,
                fontsize=24,
                ymax=None,ymin=None,
                xlabel='\Delta L [km]',
                root=RESULTS_PATH,
                color='Maroon',
                save=True,
                smooth=True,
                s=3,binned=True,nbins=20,
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

    if binned:
        npts = len(x) / nbins
        x, y = bin_data(ds, sigmas, npts)
        
    plt.semilogy(x, y, lw=3, color=color)
    plt.ylim(ymin=ymin,ymax=ymax)
    xlabel = ax.set_xlabel(r'$\Delta$ L [km]',fontsize=fontsize)

    zs, Bsat = saturationB_simple()
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


    
def GvsB(zs=[20,22,25,28,31],
         nBs=100, Bmin=1e-22,Bmax=1e-18,
        thetak=0.1,phik=0.83,
        val_nk=f.FFTT_nk,
        val_Ts=rf.Ts_21cmfast_interp, 
        val_Tk=rf.Tk_21cmfast_interp, 
        val_Jlya=rf.Jlya_21cmfast_interp, 
        val_Tg=rf.Tg_21cmfast_interp,
        phin=0., thetan=np.pi/2.,
        fontsize=24,
        make_plot=False,
        root=RESULTS_PATH,
        ymax=1e-5,
        ymin=1e-14):
    """This plots deltaG(B), at a given z.
    """
    Bref = 0.
    plt.figure()
    ax = plt.gca()
    colors = ['Violet','DarkBlue','Maroon','r','Orange','gray']
    zs, Bsat = saturationB_simple(zs=zs)
    Bs = np.logspace(np.log10(Bmin),np.log10(Bmax),nBs)
    deltaG = np.zeros(len(Bs))

    for j,z in enumerate(zs):
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
    
        
        deltaG = np.zeros(len(Bs))
        Gref = pt.calc_G(thetak=thetak, phik=phik, 
                        thetan=thetan, phin=phin,
                        Ts=Ts, Tg=Tg, z=z, 
                        verbose=False, 
                        xalpha=xalpha, xc=xc, xB=xBcoeff*Bref, x1s=1.)
        for i,B in enumerate(Bs):
            xB = xBcoeff*B
            G = pt.calc_G(thetak=thetak, phik=phik, 
                            thetan=thetan, phin=phin,
                            Ts=Ts, Tg=Tg, z=z, 
                            verbose=False, 
                            xalpha=xalpha, xc=xc, xB=xB, x1s=1.)
            deltaG[i] = np.abs((G - Gref) / Gref)


            
        ax.loglog(Bs,deltaG,lw=4,color=colors[j])
        ax.loglog(np.array([Bsat[j],Bsat[j]]),np.array([ymin,ymax]),'--',lw=2,color=colors[j])
    
    xlabel = ax.set_xlabel('B [Gauss]',fontsize=fontsize)
    ylabel = ax.set_ylabel(r'$\Delta G / G$',fontsize=fontsize)
    plt.savefig(root+'G_vs_B.pdf', 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')

        


def saturationB_simple(zs=None,nzs=100,
                zmin=15, zmax=35,
                val_nk=f.FFTT_nk,
                val_Ts=rf.Ts_21cmfast_interp, 
                val_Tk=rf.Tk_21cmfast_interp, 
                val_Jlya=rf.Jlya_21cmfast_interp, 
                val_Tg=rf.Tg_21cmfast_interp,
                make_plot=False,
                fontsize=24):
    """This computes the mag. field strength at saturation, at a given z, in a simplified way, as Bsaturation = (xc + xalpha +1) / xBcoeff.
    """
    if zs is None:
        zs = np.linspace(zmin, zmax,nzs)
    Bsat = np.zeros(len(zs))
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
        Bsat[i] = (1+xalpha+xc)/xBcoeff/(1+z)**2

       
        
    if make_plot:
        plt.figure()
        #fig = plt.gcf()
        ax = plt.gca()
        
        ylabel = ax.set_ylabel('Saturation [Gauss comov.]',fontsize=fontsize)
        xlabel = ax.set_xlabel('z',fontsize=fontsize)
       
        plt.semilogy(zs,Bsat,lw=4,color='b')
        plt.grid(b=True,which='major')

        
        plt.savefig(RESULTS_PATH+'Bsaturation.pdf',
                    bbox_extra_artists=[xlabel, ylabel], 
                    bbox_inches='tight')
        
    
    return zs,Bsat


def bin_data(x, y, npts):
    """
    A modification of Ruth Angus' function for binning your data.
    Binning is sinning, of course, but if you want to get things
    set up quickly this can be very helpful!
    It takes your data: x, y
    npts (int) is the number of points per bin.
    """
    mod, nbins = len(x) % npts, len(x) / npts
    if mod != 0:
        x, y = x[:-mod], y[:-mod]
    xb, yb = [np.zeros(nbins) for i in range(2)]
    for i in range(npts):
        xb += x[::npts]
        yb += y[::npts]
        
        x, y = x[1:], y[1:]
    return xb/npts, yb/npts
