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


def grid_DeltaL_Omega(tobs=1, zmin=10, zmax=35, mode='B0', Jmode='default'):
    deltaLs = np.load(RESULTS_PATH + '/DeltaLs_{}_{}_tobs_{:.1f}.npy'.format(mode,
                                                                         Jmode,
                                                                         tobs))
    omegas = np.load(RESULTS_PATH + '/Omegasurveys_{}_{}_tobs_{:.1f}.npy'.format(mode,
                                                                         Jmode,
                                                                         tobs))

    grid = np.zeros((len(omegas), len(deltaLs)))
    for i,om in enumerate(omegas):
        for j,delta in enumerate(deltaLs):
            name = '{}_{}_tobs_{:.2f}_DeltaL_{:.2f}_Omega_{:.2f}'.format(mode,
                                                                        Jmode,
                                                                        tobs,
                                                                        delta,om)
            filename = '{}/{}/{}.txt'.format(RESULTS_PATH,name,name)
            data = np.loadtxt(filename, skiprows=1, usecols=(0,))
            print om,delta,data
            grid[i,j] = data#[0]
    return deltaLs, omegas, grid

def plot_grid(imfile='B0_imshow.png',**kwargs):
    d,o,g = grid_DeltaL_Omega(**kwargs)
    extent = [d.min(), d.max(), o.min(), o.max()]
    plt.imshow(np.log10(g), origin='upper', extent=extent)
    plt.xlabel('deltaL')
    plt.ylabel('omega')
    plt.colorbar()
    plt.savefig(imfile)
    
if __name__=='__main__':
    plot_grid()

