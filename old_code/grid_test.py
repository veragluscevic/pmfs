#this script is just to verify how trapznd works.

import numpy as np

def integrate_test(xmin=2.,xmax=5.,ymin=10.,ymax=100.,zmin=-4.,zmax=5.,wmin=1.,wmax=2.,Nxs=15, Nys=100,Nzs=20,Nws=21):

    xs = np.logspace(np.log10(xmin), np.log10(xmax), Nxs)
    ys = np.logspace(np.log10(ymin), np.log10(ymax), Nys)
    zs = np.linspace(zmin, zmax, Nzs)
    ws = np.linspace(wmin, wmax, Nws)
  
    fisher_grid = np.zeros((Nxs,Nys,Nzs,Nws))
    for i1,x in enumerate(xs):
        #print x
        
        for i2,y in enumerate(ys):
            for i3,z in enumerate(zs):
                for i4,w in enumerate(ws):
                    res = np.sin(x)*y*z**2*np.cos(w)
                    fisher_grid[i1,i2,i3,i4] = res
                                            
    #np.save(root + "fisher_grid_test.npy", fisher_grid)
    #np.save(root + "x_grid.npy", xs)
    #np.save(root + "y_grid.npy", ys)
    #np.save(root + "z_grid.npy", zs)
    #np.save(root + "w_grid.npy", ws)
    
    return trapznd(fisher_grid,xs,ys,zs,ws)
    
def trapznd(arr,*axes):
    """This is a simple trapeziodal integration function.
    It takes an array of integrand values, and arrays for each axis of the grid:
    trapznd(fisher_grid,zs,phiks,thetaks,ks),
    and it returns a single real number."""
    
    n = len(arr.shape)
    if len(axes) != n:
        raise ValueError('must provide same number of axes as number of dimensions!')
    val = np.trapz(arr,axes[0],axis=0)
    for i in np.arange(1,n):
        val = np.trapz(val,axes[i],axis=0)
    return val

print integrate_test()
