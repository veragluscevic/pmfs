"""
This module calculates the effective spectral index n_eff at given k for some input power spectrum.
---Xiao Fang, 2016.04
"""


import numpy as np

def n_eff(k,P):
    ln_p=np.log(P)
    ln_k=np.log(k)
    
    return k[:-1], np.diff(ln_p)/np.diff(ln_k)


if __name__ == '__main__':
    d=np.loadtxt('matterpower.txt') 
    k=d[:,0]; P=d[:,1]
    k_fin, neff = n_eff(k,P)
    print(k)

    k_sensitive = 1.
    print(np.interp(k_sensitive, k_fin, neff))