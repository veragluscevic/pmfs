#!/usr/bin/env python

import matplotlib
if __name__=='__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def plot_fishergrid(imfile='fisher_grid.png',slices=[0,0,0,0],plotlog=False,
                    fig_path='/home/verag/Dropbox/MagFields21cm/prelim_res/',savefig=False,
                    grid_path = RESULTS_PATH + 'B0_default_tobs_1.00_DeltaL_4.00_Omega_1.13/'):
    grid = np.load(grid_path + 'fisher_grid.npy')
    zs = np.load(grid_path + 'z_grid.npy')
    phiks = np.load(grid_path + 'phik_grid.npy')
    thetaks = np.load(grid_path + 'thetak_grid.npy')
    ks = np.load(grid_path + 'k_grid.npy')

    if plotlog:
        gridp = np.log10(np.abs(grid))
    else:
        gridp = grid
    #print np.shape(fisher_grid)
    plt.figure()
    extent = [ks.min(), ks.max(), thetaks.min(), thetaks.max()]
    plt.imshow(gridp[slices[0],slices[1],:,:], origin='lower', aspect='auto', extent=extent)
    plt.ylabel('theta')
    plt.xlabel('k')
    plt.xscale('log')
    plt.colorbar()
    plt.title('z={:.2f}, phi={:.2f}'.format(zs[slices[0]],phiks[slices[1]]))
    figname = fig_path + 'grid_k_theta.png'
    if savefig:
        plt.savefig(figname)
     
    plt.figure()
    extent = [ks.min(), ks.max(), phiks.min(), phiks.max()]
    plt.imshow(gridp[slices[0],:,slices[2],:], origin='lower', aspect='auto', extent=extent)
    plt.ylabel('phi')
    plt.xlabel('k')
    plt.xscale('log')
    plt.colorbar()
    plt.title('z={:.2f}, theta={:.2f}'.format(zs[slices[0]],thetaks[slices[2]]))
    figname = fig_path + 'grid_k_phi.png'
    if savefig:
        plt.savefig(figname)
    
    plt.figure()
    extent = [thetaks.min(), thetaks.max(), phiks.min(), phiks.max()]
    plt.imshow(gridp[slices[0],:,:,slices[3]], origin='lower', aspect='auto', extent=extent)
    plt.ylabel('phi')
    plt.xlabel('theta')
    plt.colorbar()
    plt.title('z={:.2f}, k={:.4f}'.format(zs[slices[0]],ks[slices[3]]))
    figname = fig_path + 'grid_theta_phi.png'
    if savefig:
        plt.savefig(figname)
    
    plt.figure()
    extent = [ks.min(), ks.max(), zs.min(), zs.max()]
    plt.imshow(gridp[:,slices[1],slices[2],:], origin='lower', aspect='auto', extent=extent)
    plt.ylabel('z')
    plt.xlabel('k')
    plt.xscale('log')
    #plt.yscale('log')
    plt.colorbar()
    plt.title('phi={:.2f}, theta={:.2f}'.format(phiks[slices[1]],thetaks[slices[2]]))
    figname = fig_path + 'grid_k_z.png'
    if savefig:
        plt.savefig(figname)
    
    plt.figure()
    extent = [thetaks.min(), thetaks.max(), zs.min(), zs.max()]
    plt.imshow(gridp[:,slices[1],:,slices[3]], origin='lower', aspect='auto', extent=extent)
    plt.ylabel('z')
    plt.xlabel('theta')
    #plt.yscale('log')
    plt.colorbar()
    plt.title('phi={:.2f}, k={:.4f}'.format(phiks[slices[1]],ks[slices[3]]))
    figname = fig_path + 'grid_theta_z.png'
    if savefig:
        plt.savefig(figname)
    
    plt.figure()
    extent = [phiks.min(), phiks.max(), zs.min(), zs.max()]
    plt.imshow(gridp[:,:,slices[2],slices[3]], origin='lower', aspect='auto', extent=extent)
    plt.ylabel('z')
    plt.xlabel('phi')
    #plt.yscale('log')
    plt.colorbar()
    plt.title('z={:.2f}, phi={:.4f}'.format(zs[slices[2]],phiks[slices[3]]))
    figname = fig_path + 'grid_phi_z.png'
    if savefig:
        plt.savefig(figname)
