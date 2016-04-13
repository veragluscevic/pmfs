"""
This module plot the shear transverse power spectrum calculated by pt.py

---Xiao Fang, 2015.01
"""

import pylab as pl
import numpy as np
import matplotlib

def main():
    pl.ion()
    fig=pl.figure()
    x1=np.loadtxt('power_spectrum_1.dat')
    x2=np.loadtxt('power_spectrum_2.dat')
    x3=np.loadtxt('power_spectrum_3.dat')
    line1,=pl.loglog(x1[:,0],x1[:,1],label='z=10')
    line2,=pl.loglog(x2[:,0],x2[:,1],label='z=20')
    line3,=pl.loglog(x3[:,0],x3[:,1],label='z=30')
    pl.show()
    pl.xlabel('$l$')
    pl.ylabel('$P_t(l)$')
    pl.legend([line1,line2,line3],['z=10','z=20','z=30'])
    fig.savefig('pt_1.eps')


    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import rc

    fig=plt.figure()

    rc('font',**{'family':'serif','serif':['Times','Palatino']})
    rc('text', usetex=True)

    ax=plt.subplot(111)

    ax.tick_params(axis='both', which='major', length=12, width=2)
    ax.tick_params(axis='both', which='minor', length=8, width=1)
    ax.tick_params(axis='both', labelsize=22)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\ell$',fontsize=22)
    ax.set_ylabel(r'$\ell(\ell+1)P_t(\ell)/2\pi$',fontsize=22)
    ax.set_xlim(2,3E3)
    ax.set_ylim(1E-9,3E-7)

    ax.plot(x1[:,0],x1[:,2],'--',label='z=10',linewidth=3)
    ax.plot(x2[:,0],x2[:,2],'-',label='z=20',linewidth=3)
    ax.plot(x3[:,0],x3[:,2],'-.',label='z=30',linewidth=3)

    ax.grid(which='both')
    plt.legend(fontsize=22,loc=3)
    plt.gcf().subplots_adjust(left=0.15,bottom=0.15)
    plt.show()

    fig.savefig('pt_2.eps')
    fig.savefig('pt_2.pdf')

main()
