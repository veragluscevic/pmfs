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

val_Jlya = rf.Jlya_21cmfast_interp
val_Tg = rf.Tg_21cmfast_interp
val_Tk = rf.Tk_21cmfast_interp
val_Ts = rf.Ts_21cmfast_interp
             

from scipy.interpolate import UnivariateSpline as interpolate
from scipy.ndimage import gaussian_filter1d

@jit
def plot_dA(zmin=15,zmax=35,nz=100):
    das = np.zeros(nz)
    zs = np.linspace(zmin,zmax,nz)
    for i,z in enumerate(zs):
        das[i] = cf.val_dA( z )/Mpc_in_cm # Mpc comoving


    #kmin = 2.*np.pi/(dA*sint/Mpc_in_cm) # 1/Mpc comoving
    #kmax = kmin * DeltaL_cm / lambda_z # 1/Mpc comoving
    plt.plot(zs,das,lw=3)
    
        


def sigma_z(zmin=15,zmax=28,
            mode='B0',
            t_yr=1.,
            baselines=[1.,2.,4.,10.],
            kminmin=0.01,kmaxmax=1.,
            neval=300,neval_PBi=100,
            Omega_survey=1.,
            thetan=np.pi/2.,phin=0.,
            fontsize=24, smooth=True, binned=True, 
            nbins=20,s=4,ymax=5e-18, xmax=None, label=''):

    """Integrand of SNR for SI, and integrand of sigma for B0, as a function of z"""

    
    colors = ['DarkCyan','DarkBlue','blue','cyan']

    plt.figure()
    ax = plt.gca()

    zsat, Bsat = saturationB_simple(nzs=neval, zmin=zmin, zmax=zmax)
    ceiling = np.ones(len(zsat)) * ymax
    plt.semilogy(zsat, Bsat,'--', lw=2, color='gray')
    plt.fill_between(zsat, Bsat, ceiling, alpha=0.14, color='gray')
    
    for i,DeltaL_km in enumerate(baselines):
        if mode=='SI':
            zs, sigma = f.calc_SNR(zmin=zmin,zmax=zmax,
                     t_yr=t_yr,
                      DeltaL_km=DeltaL_km,
                      kminmin=kminmin,kmaxmax=kmaxmax,
                      neval=neval,neval_PBi=neval_PBi,
                      Omega_survey=Omega_survey,
                      thetan=thetan,phin=phin,
                      plotter_calling=True)
        if mode=='B0':
            zs, sigma = f.rand_k_integrator(neval=1000, nzs=100, DeltaL_km=DeltaL_km,
                                          t_yr=t_yr,
                                          kminmin=kminmin,kmaxmax=kmaxmax,
                                          zmax=zmax,zmin=zmin,
                                          Omega_survey=Omega_survey,
                                          thetan=thetan,phin=phin)
     
        if smooth:
            x = gaussian_filter1d(zs, s)
            y = gaussian_filter1d(sigma, s)
        else:
            x = zs
            y = sigma

        if binned:
            npts = len(x) / nbins
            x, y = bin_data(zs, sigma, npts)
        plt.semilogy(x, y,lw=4,color=colors[i],label='{:.0f} km'.format(DeltaL_km))

        
    xlabel = ax.set_xlabel('z',fontsize=fontsize)
    ylabel = ax.set_ylabel(r'$1\sigma$ [Gauss]', fontsize=fontsize)
    plt.grid(b=True,which='both')
    plt.legend(loc='upper left',fontsize=fontsize)

    if xmax is None:
        xmax = zmax
    ax.set_ylim(ymax=ymax)
    ax.set_xlim(xmax=xmax,xmin=zmin)
    fname = RESULTS_PATH + 'sigma{}_vs_z{}.pdf'.format(mode,label)
    plt.savefig(fname, 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')

    return zs, sigma
    

@jit
def visualize_hp(thetak=np.pi/2.,phik=np.pi/2.,
                 nside=64, npix=None,
                fileroot=RESULTS_PATH, z=30,
                fontsize=24):
    """This produces healpy visualization of the quadrupole pattern,
    in the frame of the atom.
    """
    Bs = np.array([0.,1e-18,1e-17, 1e-16])
   
    Ts = val_Ts( z )
    Tg = val_Tg( z )
    Tk = val_Tk( z )
    Jlya = val_Jlya( z )        
    Salpha = cf.val_Salpha(Ts, Tk, z, 1., 0) 
    xalpha = rf.val_xalpha( Salpha=Salpha, Jlya=Jlya, Tg=Tg )
    xc = rf.val_xc(z, Tk=Tk, Tg=Tg)
    xBcoeff = ge * muB * Tstar / ( 2.*hbar * A * Tg ) 
    
    if npix is None:
        npix = hp.nside2npix(nside)
    for B in Bs:
        filename = fileroot + 'hp_B_{:.0}G.pdf'.format(B)
        mapa = np.zeros(npix)
        for ipix in np.arange(npix):
            thetan, phin = hp.pix2ang(nside, ipix)
            mapa[ipix] = pt.pattern_Tb(thetak=thetak, phik=phik,
                                       thetan=thetan, phin=phin, 
                                        xalpha=xalpha, xc=xc, xB=xBcoeff*B)
        if B > 0.:
            Bexponent = np.log10(B / (1+z)**2)
            title = r'$10^{{{:.0f}}}$ Gauss'.format(Bexponent)
        else:
            title = 'no magnetic field'
        hp.mollview(mapa, title='', cbar=False)
        plt.title(title, fontsize=fontsize)
        plt.savefig(filename)
    

@jit
def arb_xT(zmin=15,zmax=33, nzs=100,
           fontsize=24,root=RESULTS_PATH,
           filename='global_evolution_zetaIon31.50_Nsteps40_zprimestepfactor1.020_zetaX1.0e+56_alphaX1.2_TvirminX1.0e+04_Pop3_300_200Mpc__midFSTAR',
           filenames_uncertainty=['global_evolution_zetaIon31.50_Nsteps40_zprimestepfactor1.020_zetaX1.0e+56_alphaX1.2_TvirminX1.0e+04_Pop3_300_200Mpc__loFSTAR','global_evolution_zetaIon31.50_Nsteps40_zprimestepfactor1.020_zetaX1.0e+56_alphaX1.2_TvirminX1.0e+04_Pop3_300_200Mpc__hiFSTAR'],
           label='',B0=1e-22,
           ymax_T=100,ymin_x=1e-6):
    """Takes filenames_uncertainty as a list of 2 filenames, no root,
    e.g. ['global_evolution_zetaIon31.50_Nsteps40_zprimestepfactor1.020_zetaX1.0e+56_alphaX1.2_TvirminX1.0e+04_Pop3_300_200Mpc__loFSTAR','global_evolution_zetaIon31.50_Nsteps40_zprimestepfactor1.020_zetaX1.0e+56_alphaX1.2_TvirminX1.0e+04_Pop3_300_200Mpc__hiFSTAR']

    """

    file_21cmfast = np.loadtxt(INPUTS_PATH+filename)
    Tks_21cmfast = file_21cmfast[:,2][::-1]
    Tgs_21cmfast = file_21cmfast[:,5][::-1]
    Tss_21cmfast = file_21cmfast[:,4][::-1]
    Jlyas_21cmfast = file_21cmfast[:,6][::-1]
    zs_21cmfast = file_21cmfast[:,0][::-1]
    xH_21cmfast = file_21cmfast[:,1][::-1]

    # plot J_Lya 
    plt.figure()
    ax = plt.gca()
    xlabel = ax.set_xlabel('z',fontsize=fontsize)
    ylabel = ax.set_ylabel(r'$J_{Ly\alpha}$ [$cm^{-2} sec^{-1} Hz^{-1}sr^{-1}$]',fontsize=fontsize)
    ax.semilogy(zs_21cmfast,Jlyas_21cmfast,lw=4,color='Black')
    ax.set_xlim(xmin=zmin,xmax=zmax)
    
    # if uncertainty files given, plot band around Jlya
    if filenames_uncertainty is not None:
        file_21cmfast_lo = np.loadtxt(INPUTS_PATH+filenames_uncertainty[0])
        Jlyas_21cmfast_lo = file_21cmfast_lo[:,6][::-1]
        zs_21cmfast_lo = file_21cmfast_lo[:,0][::-1]
        
        file_21cmfast_hi = np.loadtxt(INPUTS_PATH+filenames_uncertainty[1])
        Jlyas_21cmfast_hi = file_21cmfast_hi[:,6][::-1]
        zs_21cmfast_hi = file_21cmfast_hi[:,0][::-1]

        ax.fill_between(zs_21cmfast_lo, Jlyas_21cmfast_lo, Jlyas_21cmfast_hi, 
                        facecolor='gray', interpolate=True, alpha=0.4, lw=0)
    plt.savefig(RESULTS_PATH+'Jlya{}.pdf'.format(label), 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')


    # compute x's and write to file, then plot
    xc = []; xB = []; xalpha = []
    fout = open(RESULTS_PATH + 'xs_table.txt', 'w')
    fout.write('z  x_alpha x_c xBcoeff\n')
    for i,z in enumerate(zs_21cmfast):
        B = B0*(1+z)**2

        Salpha = cf.val_Salpha(Tss_21cmfast[i], Tks_21cmfast[i], z, xH_21cmfast[i], 0) 
        xalpha.append(rf.val_xalpha( Salpha=Salpha, Jlya=Jlyas_21cmfast[i], Tg=Tgs_21cmfast[i] ))
        xc.append(rf.val_xc(z, Tk=Tks_21cmfast[i], Tg=Tgs_21cmfast[i]))
        xBcoeff = ge * muB * Tstar / ( 2.*hbar * A * Tgs_21cmfast[i] )
        xB.append(B*xBcoeff)
        fout.write('{}  {}  {} {}\n'.format(z, xalpha[i], xc[i], xBcoeff ))
    fout.close()
        
    plt.figure()
    ax = plt.gca()
    xlabel = ax.set_xlabel('z',fontsize=fontsize)
    ylabel = ax.set_ylabel('')
    plt.semilogy(zs_21cmfast,xc,lw=4,color='g',label='$x_c$')
    plt.semilogy(zs_21cmfast,xalpha,lw=4,color='b',label=r'$x_{\alpha}$')
    plt.semilogy(zs_21cmfast,xB,lw=4,color='k',label=r'$x_B$ ($10^{{{:.0f}}}$ G)'.format(np.log10(B0)))
    plt.legend(fontsize=fontsize,frameon=False,loc='lower right')
    plt.xlim(xmin=zmin,xmax=zmax)
    plt.ylim(ymin=ymin_x)
    plt.savefig(RESULTS_PATH+'xs{}.pdf'.format(label), 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')


    
    # plot ionization history
    plt.figure()
    ax = plt.gca()
    xlabel = ax.set_xlabel('z',fontsize=fontsize)
    ylabel = ax.set_ylabel('$x_H$',fontsize=fontsize)
    plt.plot(zs_21cmfast,xH_21cmfast,lw=4,color='red')
    plt.savefig(RESULTS_PATH+'xion{}.pdf'.format(label), 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')

    # plot T's
    plt.figure()
    ax = plt.gca()
    xlabel = ax.set_xlabel('z',fontsize=fontsize)
    ylabel = ax.set_ylabel('T [K]',fontsize=fontsize)
    plt.plot(zs_21cmfast,Tss_21cmfast,lw=4,color='k',label='$T_s$')
    plt.plot(zs_21cmfast,Tgs_21cmfast,'--',lw=4,color='b',label=r'$T_{\gamma}$')
    plt.plot(zs_21cmfast,Tks_21cmfast,'-.',lw=4,color='g',label='$T_k$')
    plt.legend(fontsize=fontsize,frameon=False,loc='upper left')
    plt.xlim(xmin=zmin,xmax=zmax)
    plt.ylim(ymax=ymax_T)
    plt.savefig(RESULTS_PATH+'Ts{}.pdf'.format(label), 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')






#@jit
def vis_xT(zmin=15,zmax=35, nzs=100,
           fontsize=24,root=RESULTS_PATH):

 
    B0=1e-16
    #powB = np.log10(B0)
    zs = np.linspace(zmin,zmax,nzs)
    Ts = []; Tg = []; Tk = []; Jlya = []; Jlya_noheat = []; Jlya_hiheat = []
    xc = []; xB = []; xalpha = []
    for i,z in enumerate(zs):
        B = B0/(1+z)**2
        Ts.append(rf.Ts_21cmfast_interp( z ))
        Tg.append(f.val_Tg( z ))
        Tk.append(f.val_Tk( z ))
        Jlya.append(rf.Jlya_21cmfast_interp( z ))
        #Jlya_noheat.append(rf.Jlya_21cmfast_noheat_interp( z ))
        #Jlya_hiheat.append(rf.Jlya_21cmfast_hiheat_interp( z ))
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
    plt.semilogy(zs,Jlya,lw=4,color='Gray')
    #plt.semilogy(zs,Jlya_noheat,lw=4,color='Gray')
    plt.xlim(xmin=zmin,xmax=zmax)
    plt.savefig(RESULTS_PATH+'Jlya.pdf', 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')


def grid_DeltaL(modes=['B0','SI'],t_yr=1., 
                folder='midFSTAR',
                folders_uncertainty=None,                
                Omega=1.,
                fontsize=24,
                xlabel='\Delta L [km]',
                root=RESULTS_PATH,
                colors=['Maroon','gray'],
                save=True,
                smooth=True,
                s=3,binned=False,nbins=20,
                plot_grid=True,
                debug=False,
                ymax=1e-20,
                ymin=None,
                ylabel=None):

    """Master plotter
    modes= ['B0', 'SI'] or ['xi'] 
    folders_uncertainty: this is a list of 2 filenames, for xi only, eg folders_uncertainty=['loFSTAR', 'hiFSTAR']
    """

    latexmode = {'SI':'stochastic (SI)', 'B0':'uniform', 'xi': r'$\xi$'}

    # set up figure
    plt.figure()
    ax = plt.gca()
    fig = plt.gcf()
    xlabel = ax.set_xlabel(r'$\Delta$ L [km]',fontsize=fontsize)
    if (ylabel is None) and (len(modes) > 1):
        ylabel = ax.set_ylabel('[Gauss]',fontsize=fontsize)
    else:
        if (ylabel is None):
            ylabel = ax.set_ylabel(r'$\xi$',fontsize=fontsize)
        else:
            ylabel = ax.set_ylabel(ylabel, fontsize=fontsize)
  
    if plot_grid:
        plt.grid(b=True,which='both')


    # mode by mode: read, then plot data
    for i,mode in enumerate(modes):
        Dfile = root + folder + '/' + 'DeltaLs_{}_tyr_{:.2f}_Omega_{:.2f}.npy'.format(mode,t_yr,Omega)
        DeltaLs = np.load(Dfile)
        sigmas = []
        ds = []
        for j,d in enumerate(DeltaLs):
            infile = root + folder + '/' + '{}_tyr_{:.2f}_DeltaL_{:.2f}_Omega_{:.2f}.txt'.format(mode,
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
        if mode=='SI':
            sigmas = sigmas / np.pi

        if smooth:
            x = gaussian_filter1d(ds, s)
            y = gaussian_filter1d(sigmas, s)
        else:
            x = ds
            y = sigmas

        if binned:
            npts = len(x) / nbins
            x, y = bin_data(ds, sigmas, npts)

        plt.semilogy(x, y, lw=4, color=colors[i],label=latexmode[mode])

    # add uncertainty band
    if folders_uncertainty is not None:
        x = {}
        y = {}
        for fu in folders_uncertainty:
            Dfile = root + fu + '/' + 'DeltaLs_{}_tyr_{:.2f}_Omega_{:.2f}.npy'.format(mode,t_yr,Omega)
            DeltaLs = np.load(Dfile)
            sigmas = []
            ds = []
            for j,d in enumerate(DeltaLs):
                infile = root + fu + '/' + '{}_tyr_{:.2f}_DeltaL_{:.2f}_Omega_{:.2f}.txt'.format(mode,
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
            if 'SI' in modes:
                sigmas = sigmas / np.pi

            

            if smooth:
                x[fu] = gaussian_filter1d(ds, s)
                y[fu] = gaussian_filter1d(sigmas, s)
            else:
                x[fu] = ds
                y[fu] = sigmas

            if binned:
                npts = len(x[fu]) / nbins
                x[fu], y[fu] = bin_data(ds, sigmas, npts)

        ax.fill_between(x[folders_uncertainty[0]], y[folders_uncertainty[0]], y[folders_uncertainty[1]], alpha=0.3, 
                    facecolor='red', interpolate=True, lw=0)
        

    # more global plot set up and save
    if len(modes) > 1:
        plt.legend(fontsize=fontsize)

    ax.set_ylim(ymin=ymin,ymax=ymax)

    if save:
        fname = root + 'B_vs_deltas.pdf'
        if (modes[0]=='xi') and (len(modes)==1):
            fname = root + 'xi_vs_deltas.pdf'

        plt.savefig(fname, 
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
        
        ylabel = ax.set_ylabel('Saturation [Gauss]',fontsize=fontsize)
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


