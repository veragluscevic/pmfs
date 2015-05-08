
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
            name = RESULTS_PATH + '/{}_{}_tobs_{:.2f}_DeltaL_{:.2f}_Omega_{:.2f}'.format(mode,
                                                                                             Jmode,
                                                                                             tobs,
                                                                                        delta,om)
            filename = '{}/{}.txt'.format(name,name)
            data = np.loadtxt(filename, skiprows=1, usecols=(0,))
            grid[i,j] = data[0]
    return deltaLs, omegas, grid

if __name__=='__main__':
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    deltas, omegas, grid = grid_DeltaL_Omega()

    plt.contour(deltas, omegas, grid)
    plt.savefig('B0_grid.png')

