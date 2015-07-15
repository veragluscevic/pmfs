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
mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=18
mpl.rcParams['ytick.labelsize']=18

import numpy as np

import cosmo_functions as cf
reload(cf)
import reion_functions as rf
reload(rf)
import pmfs_transfer as pt
reload(pt)
import pmfs_fisher as pf

from constants import *
from globals import *
from geometric_functions import *

from scipy.interpolate import UnivariateSpline as interpolate
from scipy.ndimage import gaussian_filter1d


def grid_DeltaL_Omega(tobs=1, zmin=10, zmax=35, mode='B0', Jmode='default',test_plots_om=False,test_plots_del=False):
    deltaLs = np.load(RESULTS_PATH + '/DeltaLs_{}_{}_tobs_{:.1f}.npy'.format(mode,
                                                                         Jmode,
                                                                         tobs))
    omegas = np.load(RESULTS_PATH + '/Omegasurveys_{}_{}_tobs_{:.1f}.npy'.format(mode,
                                                                         Jmode,
                                                                         tobs))

    grid = np.zeros((len(omegas), len(deltaLs)))
    for j,delta in enumerate(deltaLs):
        for i,om in enumerate(omegas):
            name = '{}_{}_tobs_{:.2f}_DeltaL_{:.2f}_Omega_{:.2f}'.format(mode,
                                                                        Jmode,
                                                                        tobs,
                                                                        delta,om)
            filename = '{}/{}/{}.txt'.format(RESULTS_PATH,name,name)
            data = np.loadtxt(filename, skiprows=1, usecols=(0,))
            #print om,delta,data
            grid[i,j] = data

    print np.shape(omegas)
    
    if test_plots_om:
        plt.figure()
        for j,delta in enumerate(deltaLs):
            plt.semilogy(omegas,grid[:,j])
            plt.xlabel('Omega')
            plt.ylabel('B[G]')
            plt.savefig('B_vs_deltas.pdf')

    if test_plots_del:
        plt.figure()
        for i,om in enumerate(omegas):
            plt.semilogy(deltaLs,grid[i,:])
            plt.xlabel('DeltaL')
            plt.ylabel('B[G]')
            plt.savefig('B_vs_omegas.pdf')
            
    return deltaLs, omegas, grid

def plot_grid(imfile='B0_imshow.png', Bsat=1e-21,
              **kwargs):
    d,o,g = grid_DeltaL_Omega(**kwargs)
    g = 1. - (Bsat - g)/Bsat 
    extent = [d.min(), d.max(), o.min(), o.max()]
    #plt.imshow(np.log10(g), origin='lower', extent=extent)
    plt.imshow(g, origin='lower', extent=extent, cmap='hot')
    plt.xlabel('deltaL')
    plt.ylabel('omega')
    plt.yscale('log')
    plt.colorbar()
    plt.savefig(imfile)
    
if __name__=='__main__':
    plot_grid()

def grid_DeltaL(tobs=1., 
                zmin=15, zmax=35, 
                mode='B0', 
                Jmode='default',Omega=1.,
                Nomegas=100,omegamax=1.,omegamin=np.pi/180.,
                fontsize=20,
                plot2d=False,
                ymax=None,
                xlabel='\Delta L [km]',
                sigma=None,
                check=False):
    deltaLs = np.load(RESULTS_PATH + '/DeltaLs_{}_{}_tobs_{:.1f}.npy'.format(mode,
                                                                         Jmode,
                                                                         tobs))


    grid_delta = np.zeros(len(deltaLs))
    grid_2d = np.zeros((Nomegas,len(deltaLs)))
    omegas = np.linspace(omegamin,omegamax,Nomegas)
    alphas = omegas**0.5
    omegas1 = np.ones(Nomegas) * Omega
    alphas1 = omegas1**0.5
    omega_factors = (alphas + np.cos(alphas)*np.sin(alphas))**0.5/(alphas1 + np.cos(alphas1)*np.sin(alphas1))**0.5
    ds = np.zeros(len(deltaLs))
    for j,delta in enumerate(deltaLs):
        name = '{}_{}_tobs_{:.2f}_DeltaL_{:.2f}_Omega_{:.2f}'.format(mode,
                                                                    Jmode,
                                                                    tobs,
                                                                    delta,Omega)
        
        filename = '{}/{}/{}.txt'.format(RESULTS_PATH,name,name)
        if os.path.exists(filename):
            data = np.loadtxt(filename, skiprows=1, usecols=(0,))
            grid_delta[j] = data
            ds[j] = delta

    if plot2d:
        for j,delta in enumerate(deltaLs):
            grid_2d[:,j] = grid_delta[j] * omega_factors
            

    
    plt.figure()
    ax = plt.gca()
    fig = plt.gcf()

    if mode=='B0':
        if sigma is None:
            sigma = 1
        x = gaussian_filter1d(ds, sigma)
        y = gaussian_filter1d(grid_delta, sigma)
        plt.semilogy(x,y, color='BlueViolet',lw=3)
        if check:
            plt.semilogy(ds,grid_delta,'.', color='BlueViolet') # mediumslateblue
        ylabel = ax.set_ylabel('B [Gauss]',fontsize=fontsize)
    if mode=='zeta':
        if sigma is None:
            sigma = 3
        x = gaussian_filter1d(ds, sigma)
        y = gaussian_filter1d(grid_delta, sigma)
        plt.semilogy(x,y, color='Maroon',lw=3)
        if check:
            plt.plot(ds,grid_delta, '.',color='Maroon')
        ylabel = ax.set_ylabel(r'\xi',fontsize=fontsize)
        plt.ylim(ymax=ymax)
    xlabel = ax.set_xlabel(xlabel,fontsize=fontsize)
    
    plt.savefig('{}_vs_deltas.pdf'.format(mode), 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')

    if plot2d:
        plt.figure()
        plt.imshow(np.log10(grid_2d))
        plt.colorbar()

