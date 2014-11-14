#This module contains functions necessary for Fisher analysis for PMFs.

import numpy as np
import cosmo_functions as cf
reload(cf)
import pmfs_transfer as pt
reload(pt)
import reion_functions as rf
reload(rf)
from constants import *
from globals import *

def P21_N(z, dA=3.1695653382881036e+28, H_z=1.1905643664441961e-16, Tsys=1000, tobs=365*86400., Ae=3500**2, Lmax=1200000, Lmin=1000, Omega_survey=0.3, nk=0.37, lambda_z=443.1):
    """ This takes k [1/cm comoving], thetak [rad], and instrumental parameters:
          observation time tobs[sec], angular size of the survey [sr], system temperature Tsys[K],
          effective area of an antenna Ae[cm^2], minimum and maximum baselines, both in [cm],
          sin of angle between k and line of sight, and the number density of baselines per k mode nk[unitless].
          It returns noise power in [K^2 (comoving cm)^3 comoving]. """
    
    res = dA**2 * c * (1+z)**2 * lambda_z**2 * Tsys**2 * Omega_survey / ( Ae * nu21 * tobs *nk * H_z ) 
    return res

def Tsys_simple(z):
    """This takes redshift and returns the simple sky temperature in [K], normalized such that
    Tsys=30K @ 400MHz."""
    nu_z = nu21 / ( 1 + z )
    norm = 30. * (4e8)**2.6
    res = norm / nu_z**2.6
    return res

def Tsys_Mao2008(z):
    """This takes redshift and returns system temperature in [K], which is dominated by sky temperature
    from synchrotron radiation. Formula taken from Mao 2008, paragraph after eq 33."""
    lambda_z = lambda21 * ( 1. + z )
    res = 60. * (lambda_z/100.)**2.55
    return res

def Tsys_zero(z):
    """This is for the Tsys=0 limit."""
    return (z-z)

#calculate survey-volume element at z as Vpath_factor*dz = dVpatch:
def Vpatch_factor(z, dA=3.1695653382881036e+28, H_z=1.1905643664441961e-16, Omega_survey=0.3):
    """This is the volume element in the Fisher integral, *for a uniform magnetic field* B=B0*(1+z)^2. Vpath_factor*dz = dVpatch.
    It takes z, dA[cm comoving], H_z[1/sec], and Omega_survey[sr], and returns result in [(comoving Mpc)^3]."""
    res = c / H_z * dA**2 * Omega_survey / Mpc_in_cm**3 #/ (1+z)**2 #should there be *(1+z) ?
    return res


def one_over_u_nk(k, Ntot=523776, lambda_z=443.1, dA=3.1695653382881032e+28, sin_thetak_n=0.7, Lmax=1200000, Lmin=1000):
    """This gives the UV coverage density in baselines per k mode, which in UV space goes as ~1/u; the result is unitless.
    It takes k[1/cm comoving], total number of antenna pairs (baselines) Ntot, 21cm wavelength at z,
    ang. diam. distance in [comoving cm], sin of vector k wrt line of sight, maximum and minimum baselines [both in cm]. """

    res = 2. * Ntot * lambda_z / ( k * dA * sin_thetak_n * ( Lmax - Lmin ) )
    
    return res



def Fisher_integrand(z, k_in_Mpc, thetak=np.pi/2., phik=0., thetan=np.pi/2., phin=np.pi/4., Ts=11., xalpha=34.247221, xc=0.004176,
                     Tsys=1000, tobs=10*86400., Ae=(3500)**2, Lmax=100000, Lmin=100, N_ant=1024, x1s=1., Omega_survey=0.3,
                     dA=3.16e+28, H_z=1.19e-16, lambda_z=443.1, nk=0.3, Tg=57.23508, xBcoeff=3.65092e18, verbose=False):
    """This takes k[1/Mpc comoving]. Result is returned in CGS units, [???]."""

    k = k_in_Mpc/Mpc_in_cm

    #thetak_n = np.arccos(np.sin(thetak)*np.cos(phik)*np.sin(thetan)*np.cos(phin) +
    #                      np.sin(thetak)*np.sin(phik)*np.sin(thetan)*np.sin(phin) +
    #                      np.cos(thetak)*np.cos(thetan))

    #sin_thetak_n = np.sin(thetak_n)
    #if sin_thetak_n == 0.:
    #    return 0.
   
    Vpatch = Vpatch_factor( z, dA=dA, H_z=H_z, Omega_survey=Omega_survey )
    
    Pnoise = P21_N( dA=dA, H_z=H_z, z=z, Tsys=Tsys, tobs=tobs, Ae=Ae, 
                   Lmax=Lmax, Lmin=Lmin, Omega_survey=Omega_survey, lambda_z=lambda_z, nk=nk )
    if np.isnan( Pnoise ):
        raise ValueError( 'Pnoise is nan.' )

    ###only for testing!!!!!
    #Pnoise = 0.

    
    Pdelta = cf.val_Pdelta( z, k_in_Mpc ) 
    
    G = pt.calc_G(thetak=thetak, phik=phik, thetan=thetan, phin=phin, 
            Ts=Ts, Tg=Tg, z=z, verbose=False, xalpha=xalpha, xc=xc, xB=0., x1s=x1s)
  
    dGdB = pt.calc_dGdB(thetak=thetak, phik=phik, thetan=thetan, phin=phin,
                        Ts=Ts, Tg=Tg, z=z, verbose=False, xalpha=xalpha, xc=xc, xBcoeff=xBcoeff, x1s=x1s)
    
    Psignal = Pdelta*G**2
    
    Numerator = (2.*G*dGdB*Pdelta)**2
    #for testing purposes: Numerator = (2.*G*G*Pdelta)**2
    Denominator = 2.*(Psignal + Pnoise)**2

    res = k_in_Mpc**2*np.sin(thetak)* Vpatch * Numerator/Denominator/(2.*np.pi)**3 
    if verbose:
        print '@(z, k[1/Mpc])=(%i, %i):  Psignal=%e, Pnoise=%e\n' % (z, k_in_Mpc, Psignal, Pnoise)
        print Numerator
        print Denominator
        print G,dGdB
        
    return res
    




def write_Fisher_grid(root, val_nk=one_over_u_nk, val_Ts=rf.Ts_Hirata05, val_Tk=rf.Tk_simple, val_Jlya=rf.calc_simple_Jlya, val_x1s=rf.ones_x1s,
                      z_lims=(15,30), Nzs=20, Nks=100, Nthetak=21, Nphik=22, phin=0., thetan=np.pi/2.,
                      val_Tsys=Tsys_simple, tobs=365.*86400, Ae=3500**2, Lmax=1200000, Lmin=1000, N_ant=1024, Omega_survey=0.3,
                      kminmin=0.001, kmaxmax=100):
    """ This writes a grid of Fisher integrands, for a homogeneous B field.
    The grid has the following dimensions: z, k, thetak, and phik, where the last two are the position angles of
    the density wave-vector k, in the coordinate system with LOS n along the z-axis. nz, nk, nthetak, and nphik define the number of points
    in these four directions, respectively. Note: This takes phin, and thetan which are the position angles of LOS in the coordinate frame where
    B is along the z axis. It also takes Tsys[K], observation time in [sec], effective area of a single dish [cm^2], maximum and minimum baseline [both in cm],
    total number of antennas, the angular size of the survey in [sr], minimum and maximum k of density perturbations that we want to consider [both in 1/Mpc comoving].
    It also takes the function names for uv-coverage, for Jlya, for Ts, and for Tk."""
    
    zs = np.logspace(np.log10(z_lims[0]), np.log10(z_lims[1]), Nzs)
    ks = np.logspace(np.log10(kminmin), np.log10(kmaxmax), Nks)
    thetaks = np.linspace(0., np.pi, Nthetak)
    phiks = np.linspace(0., 2*np.pi, Nphik)


    #total number of baselines:
    Ntot = N_ant*(N_ant + 1.)/2.
    
    fisher_grid = np.zeros((Nzs,Nphik,Nthetak,Nks))
    for i1,z in enumerate(zs):
        print 'z=%.2f' % z
        dA = cf.val_dA( z )
        H_z = cf.H( z )
        x1s = val_x1s( z )
        lambda_z = lambda21 * (1. + z)
        Tsys = val_Tsys( z )
        Tg = Tcmb * (1. + z)
        Tk = val_Tk( z )
        Jlya = val_Jlya( z )        
        Ts = val_Ts( z, Tk=Tk, Jlya=Jlya, x1s=x1s )        
        Salpha = cf.val_Salpha(Ts, Tk, z, x1s, 0) #is this true that delta = 0 here???
        
        xalpha = rf.val_xalpha( Salpha=Salpha, Jlya=Jlya, Tg=Tg )
        xc = rf.val_xc(z, Tk=Tk, Tg=Tg)
        xBcoeff = ge * muB * Tstar / ( 2.*hbar * A * Tg ) 
                
        for i2,phik in enumerate(phiks):
            for i3,thetak in enumerate(thetaks):
                for i4,k in enumerate(ks):
                    
                    #compute the polar position angle of k in the LOS-coordinate frame, from thetak, phik, thetan, and phin:
                    thetak_n = np.arccos(np.sin(thetak)*np.cos(phik)*np.sin(thetan)*np.cos(phin) +
                          np.sin(thetak)*np.sin(phik)*np.sin(thetan)*np.sin(phin) +
                          np.cos(thetak)*np.cos(thetan))                  
                    sint = np.sin(thetak_n)
                    #print sint

                    k_in_cm = k / Mpc_in_cm
                    nk = val_nk(k_in_cm, Ntot=Ntot, lambda_z=lambda_z, dA=dA, sin_thetak_n=sint, Lmax=Lmax, Lmin=Lmin)
                    

                    #if the direction we are integrating over is along the LOS, kmin and kmax are the limits imposed by the configuration of the survey:
                    if sint==0:
                        kmin = 0.
                        kmax=0.

                    #otherwise, limits are:
                    else:
                        rho_min = Lmin/lambda_z
                        kmin = rho_min*2.*np.pi/(dA*sint/Mpc_in_cm)
                        rho_max = Lmax/lambda_z
                        kmax = rho_max*2.*np.pi/(dA*sint/Mpc_in_cm)
                        #print kmin,kmax

                    #note: these kmin and kmax are in 1/Mpc comoving

                    #if the considered k is in the range of observable k's, compute integrand, otherwise set it to zero:
                    if k<=kmax and k>=kmin and sint>0.:                        
                        res = Fisher_integrand(z, k, thetak=thetak, phik=phik, thetan=thetan, phin=phin,
                                               Ts=Ts, xalpha=xalpha, xc=xc, Tg=Tg, xBcoeff=xBcoeff,
                                               Tsys=Tsys, tobs=tobs, Ae=Ae, Lmax=Lmax, Lmin=Lmin, N_ant=N_ant, x1s=x1s, Omega_survey=Omega_survey,
                                               dA=dA, H_z=H_z, lambda_z=lambda_z, nk=nk)
                        #print 'z=%f, k=%f in (%f, %f), res=%e' % (z,k,kmin,kmax,res)

                        #if res>0.:
                        #    print 'z=%f, k=%f, res=%e\n' % (z,k,res)
                        #print kmin, kmax
                        #if res==0:
                        #    print res
                    else:
                        #if k>kmax or k<kmin:
                        #    print 'z=%f, k=%f is outside the range (%f, %f)' % (z,k,kmin,kmax)
                        res = 0.
                        
                    if np.isnan(res):
                        raise ValueError('res is nan at: z=%f, k=%f, thetak_n=%f, phik=%f, thetak=%f, kmin=%f, kmax=%f' 
                                         % (z,k,thetak_n,phik,thetak,kmin,kmax))
                    fisher_grid[i1,i2,i3,i4] = res
                                            
    np.save(root + "fisher_grid.npy", fisher_grid)
    np.save(root + "z_grid.npy", zs)
    np.save(root + "k_grid.npy", ks)
    np.save(root + "thetak_grid.npy", thetaks)
    np.save(root + "phik_grid.npy", phiks)
    
    


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





def Fisher_integrand_zeta(z, k_in_Mpc, thetak=np.pi/2., phik=0., thetan=np.pi/2., phin=np.pi/4., Ts=11., xalpha=34.247221, xc=0.004176,
                     Tsys=1000, tobs=10*86400., Ae=(3500)**2, Lmax=100000, Lmin=100, N_ant=1024, x1s=1., Omega_survey=0.3,
                     dA=3.16e+28, H_z=1.19e-16, lambda_z=443.1, nk=0.3, Tg=57.23508, verbose=False):
    """This calculates the integrand for estimating sigma of the zeta parameter.
        It takes k[1/Mpc comoving]. Result is returned in CGS units, [???].
    """

    k = k_in_Mpc/Mpc_in_cm
   
    Vpatch = Vpatch_factor( z, dA=dA, H_z=H_z, Omega_survey=Omega_survey )
    
    Pnoise = P21_N( dA=dA, H_z=H_z, z=z, Tsys=Tsys, tobs=tobs, Ae=Ae, 
                   Lmax=Lmax, Lmin=Lmin, Omega_survey=Omega_survey, lambda_z=lambda_z, nk=nk )
    if np.isnan( Pnoise ):
        raise ValueError( 'Pnoise is nan.' )
    
    Pdelta = cf.val_Pdelta( z, k_in_Mpc ) 
    
    G_Bzero = pt.calc_G_Bzero(thetak=thetak, phik=phik, thetan=thetan, phin=phin, 
            Ts=Ts, Tg=Tg, z=z, verbose=False, xalpha=xalpha, xc=xc, x1s=x1s)
    G_Binfinity = pt.calc_G_Binfinity(thetak=thetak, phik=phik, thetan=thetan, phin=phin, 
            Ts=Ts, Tg=Tg, z=z, verbose=False, xalpha=xalpha, xc=xc, x1s=x1s)
 
    Psignal_Bzero = Pdelta*G_Bzero**2
    Psignal_Binfinity = Pdelta*G_Binfinity**2
    
    Numerator = (Psignal_Binfinity - Psignal_Bzero)**2
    Denominator = 2.*(Psignal_Bzero + Pnoise)**2

    res = k_in_Mpc**2*np.sin(thetak)* Vpatch * Numerator/Denominator/(2.*np.pi)**3 
    if verbose:
        print '@(z, k[1/Mpc])=(%i, %i):  Psignal=%e, Pnoise=%e\n' % (z, k_in_Mpc, Psignal_Bzero, Pnoise)
        print Numerator
        print Denominator
        print G_Bzero,G_Binfinity
        
    return res
    



def write_Fisher_grid_zeta(root, val_nk=one_over_u_nk, val_Ts=rf.Ts_Hirata05, val_Tk=rf.Tk_simple,
                           val_Jlya=rf.calc_simple_Jlya, val_x1s=rf.ones_x1s,
                           z_lims=(15,30), Nzs=20, Nks=100, Nthetak=21, Nphik=22, phin=0., thetan=np.pi/2.,
                           val_Tsys=Tsys_simple, tobs=365.*86400, Ae=3500**2, Lmax=1200000, Lmin=1000, N_ant=1024, Omega_survey=0.3,
                           kminmin=0.001, kmaxmax=100):
    """ This writes a grid of Fisher integrands for zeta, for a homogeneous B field.
    The grid has the following dimensions: z, k, thetak, and phik, where the last two are the position angles of
    the density wave-vector k, in the coordinate system with LOS n along the z-axis. nz, nk, nthetak, and nphik define the number of points
    in these four directions, respectively. Note: This takes phin, and thetan which are the position angles of LOS in the coordinate frame where
    B is along the z axis. It also takes Tsys[K], observation time in [sec], effective area of a single dish [cm^2], maximum and minimum baseline [both in cm],
    total number of antennas, the angular size of the survey in [sr], minimum and maximum k of density perturbations that we want to consider [both in 1/Mpc comoving].
    It also takes the function names for uv-coverage, for Jlya, for Ts, and for Tk."""
    
    zs = np.logspace(np.log10(z_lims[0]), np.log10(z_lims[1]), Nzs)
    ks = np.logspace(np.log10(kminmin), np.log10(kmaxmax), Nks)
    thetaks = np.linspace(0., np.pi, Nthetak)
    phiks = np.linspace(0., 2*np.pi, Nphik)


    #total number of baselines:
    Ntot = N_ant*(N_ant + 1.)/2.
    
    fisher_grid = np.zeros((Nzs,Nphik,Nthetak,Nks))
    for i1,z in enumerate(zs):
        print z
        dA = cf.val_dA( z )
        H_z = cf.H( z )
        x1s = val_x1s( z )
        lambda_z = lambda21 * (1. + z)
        Tsys = val_Tsys( z )
        Tg = Tcmb * (1. + z)
        Tk = val_Tk( z )
        Jlya = val_Jlya( z )        
        Ts = val_Ts( z, Tk=Tk, Jlya=Jlya, x1s=x1s )        
        Salpha = cf.val_Salpha(Ts, Tk, z, x1s, 0) #is this true that delta = 0 here???
        
        xalpha = rf.val_xalpha( Salpha=Salpha, Jlya=Jlya, Tg=Tg )
        xc = rf.val_xc(z, Tk=Tk, Tg=Tg)

        
        for i2,phik in enumerate(phiks):
            for i3,thetak in enumerate(thetaks):
                for i4,k in enumerate(ks):
                    
                    #compute the polar position angle of k in the LOS-coordinate frame, from thetak, phik, thetan, and phin:
                    thetak_n = np.arccos(np.sin(thetak)*np.cos(phik)*np.sin(thetan)*np.cos(phin) +
                          np.sin(thetak)*np.sin(phik)*np.sin(thetan)*np.sin(phin) +
                          np.cos(thetak)*np.cos(thetan))                  
                    sint = np.sin(thetak_n)
                    #print sint

                    k_in_cm = k / Mpc_in_cm
                    nk = val_nk(k_in_cm, Ntot=Ntot, lambda_z=lambda_z, dA=dA, sin_thetak_n=sint, Lmax=Lmax, Lmin=Lmin)
                    

                    #if the direction we are integrating over is along the LOS, kmin and kmax are the limits imposed by the configuration of the survey:
                    if sint==0:
                        kmin = 0.
                        kmax=0.

                    #otherwise, limits are:
                    else:
                        rho_min = Lmin/lambda_z
                        kmin = rho_min*2.*np.pi/(dA*sint/Mpc_in_cm)
                        rho_max = Lmax/lambda_z
                        kmax = rho_max*2.*np.pi/(dA*sint/Mpc_in_cm)
                        #print kmin,kmax

                    #note: these kmin and kmax are in 1/Mpc comoving

                    #if the considered k is in the range of observable k's, compute integrand, otherwise set it to zero:
                    if k<=kmax and k>=kmin and sint>0.:                        
                        res = Fisher_integrand_zeta(z, k, thetak=thetak, phik=phik, thetan=thetan, phin=phin,
                                               Ts=Ts, xalpha=xalpha, xc=xc, Tg=Tg,
                                               Tsys=Tsys, tobs=tobs, Ae=Ae, Lmax=Lmax, Lmin=Lmin, N_ant=N_ant, x1s=x1s, Omega_survey=Omega_survey,
                                               dA=dA, H_z=H_z, lambda_z=lambda_z, nk=nk)

                    else:
                        res = 0.
                        
                    if np.isnan(res):
                        raise ValueError('res is nan at: z=%f, k=%f, thetak_n=%f, phik=%f, thetak=%f, kmin=%f, kmax=%f' 
                                         % (z,k,thetak_n,phik,thetak,kmin,kmax))
                    fisher_grid[i1,i2,i3,i4] = res
                                            
    np.save(root + "fisher_grid.npy", fisher_grid)
    np.save(root + "z_grid.npy", zs)
    np.save(root + "k_grid.npy", ks)
    np.save(root + "thetak_grid.npy", thetaks)
    np.save(root + "phik_grid.npy", phiks)
    
    
