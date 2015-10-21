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

from scipy.interpolate import UnivariateSpline as interpolate
from scipy.ndimage import gaussian_filter1d


def grid_DeltaL(mode='B0',t_yr=1., 
                Jmode='default',Omega=1.,
                fontsize=22,
                ymax=None,ymin=None,
                xlabel='\Delta L [km]',
                root=RESULTS_PATH,
                color='Maroon',
                save=True,
                smooth=True):

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
        s = 2
        ylabel = ax.set_ylabel('B [Gauss]',fontsize=fontsize)
    if mode=='zeta':
        s = 3
        ylabel = ax.set_ylabel(r'$\xi$',fontsize=fontsize)

    if mode=='SI':
        s = 1
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

    if save:
        plt.savefig(root + '{}_vs_deltas.pdf'.format(mode), 
                bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')
