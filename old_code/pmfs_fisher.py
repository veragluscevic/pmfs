#This module contains functions necessary for Fisher analysis for PMFs.
import pylab as pl
import numpy as np
import cosmo_functions as cf
reload(cf)
import pmfs_transfer as pt
reload(pt)
import reion_functions as rf
reload(rf)
from constants import *
from globals import *

import fisher as fish
reload(fish)

def P21_N(z, dA=3.1695653382881036e+28, H_z=1.1905643664441961e-16, Tsys=1000, t1=365*86400., Ae=3500**2, Lmax=1200000, Lmin=1000, nk=0.37, lambda_z=443.1):
    """ This takes k [1/cm comoving], thetak [rad], and instrumental parameters:
          observation time t1[sec], system temperature Tsys[K],
          effective area of an antenna Ae[cm^2], minimum and maximum baselines, both in [cm],
          sin of angle between k and line of sight, and the number density of baselines per k mode nk[unitless].
          
          It returns noise power in [K^2 (comoving cm)^3]. """

    res = dA**2 * c * (1+z)**2 * lambda_z**4 * Tsys**2 / ( Ae**2 * nu21 * t1 *nk * H_z ) 
    #res = dA**2 * c * (1+z)**2 * lambda_z**2 * Tsys**2 * Omega_survey / ( Ae * nu21 * tobs *nk * H_z ) 
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
def Vpatch_factor(z, dA=3.1695653382881036e+28, H_z=1.1905643664441961e-16, Omega_patch=0.3):
    """This is the volume element in the Fisher integral, *for a uniform magnetic field* B=B0*(1+z)^2. 
    Vpath_factor*dz = dVpatch.
    It takes z, dA[cm comoving], H_z[1/sec], and Omega_patch[sr], and returns result in [(comoving Mpc)^3]."""
    res = c / H_z * dA**2 * Omega_patch / Mpc_in_cm**3 #/ (1+z)**2 #should there be *(1+z) ?
    return res


def one_over_u_nk(k, Ntot=523776, lambda_z=443.1, dA=3.1695653382881032e+28, sin_thetak_n=0.7, Lmax=1200000, Lmin=1000,
                  DeltaL=None,Omega_beam=1):
    """This gives the UV coverage density in baselines per k mode, which in UV space goes as ~1/u; the result is unitless.
    It takes k[1/cm comoving], total number of antenna pairs (baselines) Ntot, 21cm wavelength at z,
    ang. diam. distance in [comoving cm], sin of vector k wrt line of sight, maximum and minimum baselines [both in cm]. """

    res = 2. * Ntot * lambda_z / ( k * dA * sin_thetak_n * ( Lmax - Lmin ) ) / Omega_beam
    
    return res

def one_over_u2_nk(k, Ntot=523776, lambda_z=443.1, dA=3.1695653382881032e+28, sin_thetak_n=0.7, Lmax=1200000, Lmin=1000,
                   DeltaL=None,Omega_beam=1):
    """This gives the UV coverage density in baselines per k mode, which in UV space goes as ~1/u^2; the result is unitless.
    It takes k[1/cm comoving], total number of antenna pairs (baselines) Ntot, 21cm wavelength at z,
    ang. diam. distance in [comoving cm], sin of vector k wrt line of sight, maximum and minimum baselines [both in cm]. """

    res = 4. * np.pi * Ntot / ( k * dA * sin_thetak_n )**2 / np.log( Lmax / Lmin ) / Omega_beam
    
    return res


def FFTT_nk(k, Ntot=None, lambda_z=443.1, dA=3.1695653382881032e+28,
            sin_thetak_n=0.7, Lmax=None, Lmin=None,
            DeltaL=100000,Omega_beam=1.):
    """
            This gives the UV coverage density in baselines per k mode, for uniform tiling by dipoles of a square
            on the ground with surface area (\DeltaL)^2; the result is unitless, and averaged over the phik_n angle, going between 0 and 2\pi.
            It takes k[1/cm comoving], 21cm wavelength at z, assumes effective area that is = lambda^2 [cm^2], 
            ang. diam. distance in [comoving cm], sin of vector k wrt line of sight, and its azimuthal angle wrt n, phik_n.
    """

    var1 = DeltaL / lambda_z
    var2 = k* dA * sin_thetak_n
    res = var1**2 - 4.*var1*var2/np.pi + var2**2/np.pi
    #res = ( DeltaL / lambda_z - k* dA * sin_thetak_n * np.cos(phik_n) / (2.*np.pi)**2 ) * ( DeltaL / lambda_z - k*dA * sin_thetak_n * np.sin(phik_n) / (2.*np.pi)**2 )
    
    return res

def test_integrand(z, Nks=100, Nthetak=100, Nphik=100,
                   kminmin=0.0001,kmaxmax=100,
                   thetak0=np.pi/4., phik0=0.2, k0=0.1,
                   thetan=np.pi/2., phin=0., 
                   val_nk=FFTT_nk, 
                   val_Ts=rf.Ts_21cmfast_interp, val_Tk=rf.Tk_21cmfast_interp, 
                   val_x1s=rf.ones_x1s,
                   val_Tg=rf.Tg_21cmfast_interp,val_Tsys=Tsys_Mao2008,
                   val_Jlya=rf.Jlya_21cmfast_interp,
                   tobs=365.*86400, 
                   Omega_patch= 2e-4,
                   verbose=False, DeltaL=200000.,
                   runk=False,runtheta=False,runphi=False):
    lambda_z = 21. * (1. + z)
    N_ant = (DeltaL/lambda_z)**2
    Omega_beam = 1.
    Lmin = lambda_z
    Lmax = DeltaL 
    Ae = lambda_z**2
    Ntot = N_ant*(N_ant + 1.)/2.
    dA = cf.val_dA( z )
    H_z = cf.H( z )
    x1s = val_x1s( z )
    Tsys = val_Tsys( z )
    Tg = val_Tg( z ) 
    Tk = val_Tk( z )
    Jlya = val_Jlya( z )        
    Ts = val_Ts( z )
    Salpha = cf.val_Salpha(Ts, Tk, z, x1s, 0)        
    xalpha = rf.val_xalpha( Salpha=Salpha, Jlya=Jlya, Tg=Tg )
    xc = rf.val_xc(z, Tk=Tk, Tg=Tg)
    xBcoeff = ge * muB * Tstar / ( 2.*hbar * A * Tg ) 

    ks = np.logspace(np.log10(kminmin), np.log10(kmaxmax), Nks)
    thetaks = np.linspace(0., np.pi, Nthetak)
    phiks = np.linspace(0., 2*np.pi-0.001, Nphik)
    #print thetaks, phiks

    #FOR K's:
    if runk:
        thetak=thetak0
        phik=phik0
        thetak_n = np.arccos(np.sin(thetak)*np.cos(phik)*np.sin(thetan)*np.cos(phin) +
                             np.sin(thetak)*np.sin(phik)*np.sin(thetan)*np.sin(phin) +
                             np.cos(thetak)*np.cos(thetan))                  
        sint = np.sin(thetak_n)

        integrands = np.zeros(len(ks))
        for i,k in enumerate(ks):
            #print k
            k_in_cm = k / Mpc_in_cm
            nk = val_nk(k_in_cm, Ntot=Ntot, lambda_z=lambda_z, dA=dA, sin_thetak_n=sint,
                        Lmax=Lmax, Lmin=Lmin, DeltaL=DeltaL, Omega_beam=Omega_beam)
            integrands[i] = Fisher_integrand(z, k, 
                                             thetak=thetak0, phik=phik0, 
                                             thetan=thetan, phin=phin, 
                                             Ts=Ts, x1s=x1s,
                                             xalpha=xalpha, xc=xc,
                                             Tsys=Tsys, 
                                             t1=tobs, Ae=Ae, Lmax=Lmax, Lmin=Lmin, 
                                             N_ant=N_ant, Omega_patch=Omega_patch,
                                             DeltaL=DeltaL,
                                             dA=dA, H_z=H_z, lambda_z=lambda_z, nk=nk, Tg=Tg, xBcoeff=xBcoeff, 
                                             verbose=verbose);
        pl.figure()
        pl.plot(ks,integrands)
        pl.xlabel('k')
        return integrands

    #FOR thetak's: 
    if runtheta:
        k=k0
        phik=phik0
        integrands = np.zeros(len(thetaks))
        dGdBs = np.zeros(len(thetaks))
        for i,thetak in enumerate(thetaks):
            #print thetak,phik,k
            thetak_n = np.arccos(np.sin(thetak)*np.cos(phik)*np.sin(thetan)*np.cos(phin) +
                             np.sin(thetak)*np.sin(phik)*np.sin(thetan)*np.sin(phin) +
                             np.cos(thetak)*np.cos(thetan))                  
            sint = np.sin(thetak_n)

            k_in_cm = k / Mpc_in_cm
            nk = val_nk(k_in_cm, Ntot=Ntot, lambda_z=lambda_z, dA=dA, sin_thetak_n=sint,
                        Lmax=Lmax, Lmin=Lmin, DeltaL=DeltaL, Omega_beam=Omega_beam)
            
            integrands[i] = Fisher_integrand(z, k0, 
                                             thetak=thetak, phik=phik0, thetan=thetan, phin=phin, 
                                             Ts=Ts, x1s=x1s,
                                             xalpha=xalpha, xc=xc,
                                             Tsys=Tsys, 
                                             t1=tobs, Ae=Ae, Lmax=Lmax, Lmin=Lmin, 
                                             N_ant=N_ant, Omega_patch=Omega_patch,
                                             DeltaL=DeltaL,
                                             dA=dA, H_z=H_z, lambda_z=lambda_z, nk=nk, Tg=Tg, xBcoeff=xBcoeff, 
                                             verbose=verbose);
            dGdBs[i] = pt.calc_dGdB(thetak=thetak, phik=0.2, 
                                    thetan=np.pi/2., phin=0.,
                                    Ts=Ts, Tg=Tg, z=z, verbose=False, 
                                    xalpha=xalpha, xc=xc, xBcoeff=xBcoeff, x1s=x1s)
        pl.figure()
        pl.plot(thetaks,integrands)
        #pl.figure()
        #pl.plot(thetaks,dGdBs)
        pl.xlabel('thetak')
        return integrands #dGdBs

    #FOR phik's: 
    if runphi:
        k=k0
        thetak=thetak0
        integrands = np.zeros(len(phiks))
        for i,phik in enumerate(phiks):
            #print phik
            thetak_n = np.arccos(np.sin(thetak)*np.cos(phik)*np.sin(thetan)*np.cos(phin) +
                             np.sin(thetak)*np.sin(phik)*np.sin(thetan)*np.sin(phin) +
                             np.cos(thetak)*np.cos(thetan))                  
            sint = np.sin(thetak_n)
            

            k_in_cm = k / Mpc_in_cm
            nk = val_nk(k_in_cm, Ntot=Ntot, lambda_z=lambda_z, dA=dA, sin_thetak_n=sint,
                        Lmax=Lmax, Lmin=Lmin, DeltaL=DeltaL, Omega_beam=Omega_beam)
            integrands[i] = Fisher_integrand(z, k0, 
                                             thetak=thetak0, phik=phik, thetan=thetan, phin=phin, 
                                             Ts=Ts, x1s=x1s,
                                             xalpha=xalpha, xc=xc,
                                             Tsys=Tsys, 
                                             t1=tobs, Ae=Ae, Lmax=Lmax, Lmin=Lmin, 
                                             N_ant=N_ant, Omega_patch=Omega_patch,
                                             DeltaL=DeltaL,
                                             dA=dA, H_z=H_z, lambda_z=lambda_z, nk=nk, Tg=Tg, xBcoeff=xBcoeff, 
                                             verbose=verbose);
        pl.figure()
        pl.plot(phiks,integrands)
        pl.xlabel('phik')
        return integrands



##############
def test2(z, Nks=10, Nthetaks=100, Nphiks=100,
                   kminmin=0.0001,kmaxmax=100,
                   thetak0=np.pi/4., phik0=0.2, k0=0.1,
                   thetan=np.pi/2., phin=0., 
                   val_nk=FFTT_nk, 
                   val_Ts=rf.Ts_21cmfast_interp, val_Tk=rf.Tk_21cmfast_interp, 
                   val_x1s=rf.ones_x1s,
                   val_Tg=rf.Tg_21cmfast_interp,val_Tsys=Tsys_Mao2008,
                   val_Jlya=rf.Jlya_21cmfast_interp,
                   tobs=365.*86400, 
                   Omega_patch= 2e-4,
                   verbose=False, DeltaL=200000.,
                   runk=False,runtheta=False,runphi=False):
    lambda_z = lambda21 * (1. + z)
    N_ant = (DeltaL/lambda_z)**2
    Omega_beam = 1.
    Lmin = lambda_z
    Lmax = DeltaL 
    Ae = lambda_z**2
    Ntot = N_ant*(N_ant + 1.)/2.
    dA = cf.val_dA( z )
    H_z = cf.H( z )
    x1s = val_x1s( z )
    Tsys = val_Tsys( z )
    Tg = val_Tg( z ) 
    Tk = val_Tk( z )
    Jlya = val_Jlya( z )        
    Ts = val_Ts( z )
    Salpha = cf.val_Salpha(Ts, Tk, z, x1s, 0)        
    xalpha = rf.val_xalpha( Salpha=Salpha, Jlya=Jlya, Tg=Tg )
    xc = rf.val_xc(z, Tk=Tk, Tg=Tg)
    xBcoeff = ge * muB * Tstar / ( 2.*hbar * A * Tg ) 

    #ks = np.logspace(np.log10(kminmin), np.log10(kmaxmax), Nks)
    ks = np.linspace(kminmin,kmaxmax, Nks)
    thetaks = np.linspace(0., np.pi, Nthetaks+1)
    phiks = np.linspace(0., 2*np.pi, Nphiks+1)

    grid = np.zeros((Nks,Nthetaks,Nphiks))
    gridnew = np.zeros((Nks,Nthetaks,Nphiks))
    for j,thetak in enumerate(thetaks[:-1]):
        print thetak
        for l,phik in enumerate(phiks[:-1]):
            thetak_n = np.arccos(np.sin(thetak)*np.cos(phik)*np.sin(thetan)*np.cos(phin) +
                                 np.sin(thetak)*np.sin(phik)*np.sin(thetan)*np.sin(phin) +
                                 np.cos(thetak)*np.cos(thetan))                  
            sint = np.sin(thetak_n)
            for i,k in enumerate(ks):
                k_in_cm = k / Mpc_in_cm
                nk = val_nk(k_in_cm, Ntot=Ntot, lambda_z=lambda_z, dA=dA, sin_thetak_n=sint,
                            Lmax=Lmax, Lmin=Lmin, DeltaL=DeltaL, Omega_beam=Omega_beam)
                grid[i,j,l] = Fisher_integrand(z, k, 
                                         thetak=thetak, phik=phik, 
                                         thetan=thetan, phin=phin, 
                                         Ts=Ts, x1s=x1s,
                                         xalpha=xalpha, xc=xc,
                                         Tsys=Tsys, 
                                         t1=tobs, Ae=Ae, Lmax=Lmax, Lmin=Lmin, 
                                         N_ant=N_ant, Omega_patch=Omega_patch,
                                         DeltaL=DeltaL,
                                         dA=dA, H_z=H_z, lambda_z=lambda_z, 
                                         nk=nk, Tg=Tg, xBcoeff=xBcoeff, 
                                         verbose=verbose)
                x = np.array([z,k,thetak,phik])
                gridnew[i,j,l] = fish.integrand(x, 
                                                Omega_patch=Omega_patch,
                                                DeltaL_km=DeltaL/1.e5,
                                                t_yr=tobs/365./24./3600.)
    return grid,gridnew


##################
def Fisher_integrand(z, k_in_Mpc, thetak=np.pi/2., phik=0., 
                     thetan=np.pi/2., phin=np.pi/4., 
                     Ts=11., xalpha=34.247221, xc=0.004176,
                     Tsys=1000, t1=10*86400., Ae=(3500)**2, 
                     Lmax=100000, Lmin=100, N_ant=1024, x1s=1.,Omega_patch=0.1,
                     dA=3.16e+28, H_z=1.19e-16, lambda_z=443.1, 
                     nk=0.3, Tg=57.23508, xBcoeff=3.65092e18, 
                     verbose=False,DeltaL=None):
    """This takes k[1/Mpc comoving]. Result is returned in CGS units, [???]."""

    #print thetak,phik,thetan,phin,Ts,xalpha,xc,Tsys,t1,Ae,Lmax,Lmin,N_ant,x1s,Omega_patch,dA,H_z,lambda_z,nk,Tg,XBcoeff,DeltaL

    k = k_in_Mpc/Mpc_in_cm
   
    Vpatch = Vpatch_factor( z, dA=dA, H_z=H_z, Omega_patch=Omega_patch )
    
    Pnoise = P21_N( dA=dA, H_z=H_z, z=z, Tsys=Tsys, t1=t1, Ae=Ae, 
                    Lmax=Lmax, Lmin=Lmin, lambda_z=lambda_z, nk=nk )

    #print dA,H_z,z,Ae,Lmax,Lmin,lambda_z,nk
    #print Tsys,t1
    #print Vpatch,Pnoise,nk
    if np.isnan( Pnoise ):
        raise ValueError( 'Pnoise is nan.' )
    #print Tsys

    
    Pdelta = cf.val_Pdelta( z, k_in_Mpc ) 
    
    G = pt.calc_G(thetak=thetak, phik=phik, thetan=thetan, phin=phin, 
            Ts=Ts, Tg=Tg, z=z, verbose=False, xalpha=xalpha, xc=xc, xB=0., x1s=x1s)
  
    dGdB = pt.calc_dGdB(thetak=thetak, phik=phik, thetan=thetan, phin=phin,
                        Ts=Ts, Tg=Tg, z=z, verbose=False, xalpha=xalpha, xc=xc, xBcoeff=xBcoeff, x1s=x1s)
    
    Psignal = Pdelta*G**2
    
    Numerator = (2.*G*dGdB*Pdelta)**2
    #for testing purposes: Numerator = (2.*G*G*Pdelta)**2
    Denominator = 2.*(Psignal + Pnoise)**2
    #print 'signal=%e' % Pdelta
    res = k_in_Mpc**2*np.sin(thetak)* Vpatch * Numerator/Denominator/(2.*np.pi)**3 
    if verbose:
        #print '@(z, k[1/Mpc])=(%i, %i):  Psignal=%e, Pnoise=%e' % (z, k_in_Mpc, Psignal, Pnoise)
        #print Numerator
        #print k_in_Mpc**2*np.sin(thetak)* Vpatch/Denominator
        print dGdB

    #print res
    return res
 
    




def write_Fisher_grid(root, val_nk=one_over_u_nk,
                      verbose_steps=False,verbose=False,
                      val_Ts=rf.Ts_21cmfast_interp, 
                      val_Tk=rf.Tk_21cmfast_interp, 
                      val_Jlya=rf.Jlya_21cmfast_interp, 
                      val_x1s=rf.ones_x1s,
                      z_lims=(15,30), Nzs=20, Nks=100, 
                      Nthetak=21, Nphik=22, phin=0., 
                      thetan=np.pi/2.,
                      val_Tsys=Tsys_simple, tobs=365.*86400, 
                      Ae=None, N_ant=None, 
                      Lmax=None, Lmin=None, Omega_patch=0.1,
                      kminmin=0.001, kmaxmax=100, 
                      val_Tg=rf.Tg_21cmfast_interp, DeltaL=100000):
    """ This writes a grid of Fisher integrands, for a homogeneous B field.
    The grid has the following dimensions: z, k, thetak, and phik, where the last two are the position angles of
    the density wave-vector k, in the coordinate system with LOS n along the z-axis. nz, nk, nthetak, and nphik define the number of points
    in these four directions, respectively. Note: This takes phin, and thetan which are the position angles of LOS in the coordinate frame where
    B is along the z axis. It also takes Tsys[K], observation time in [sec], effective area of a single dish [cm^2], maximum and minimum baseline [both in cm],
    total number of antennas, the angular size of the survey in [sr], minimum and maximum k of density perturbations that we want to consider [both in 1/Mpc comoving].

    Note that the total number of antennas and Ae will be set to correspond to FFTT-like tiling with dipoles, which means N_ant is a function
    of z, and Ae = \lambda^2.
    It also takes the function names for uv-coverage, for Jlya, for Ts, and for Tk."""
    
    #zs = np.logspace(np.log10(z_lims[0]), np.log10(z_lims[1]), Nzs)
    #ks = np.logspace(np.log10(kminmin), np.log10(kmaxmax), Nks)
    
    zs = np.linspace(z_lims[0], z_lims[1], Nzs)
    ks = np.linspace(kminmin,kmaxmax, Nks)
    thetaks = np.linspace(0., np.pi, Nthetak)
    phiks = np.linspace(0., 2*np.pi, Nphik)


    #total number of baselines, set as an effective number, as a function of lambda, if not specified:
    lambda_zs = lambda21 * (1. + zs)
    if (N_ant is None) or (Ae is None) or (Lmin is None) or (Lmax is None):
        N_ants = (DeltaL/lambda_zs)**2
        Omega_beams = np.ones(len(zs))
        Lmins = DeltaL / N_ants**0.5
        Lmaxs = DeltaL * np.ones(len(zs))
        Aes = lambda_zs**2
    else:
        N_ants = N_ant*np.ones(len(zs))
        Omega_beams = lambda_zs**2 / Ae
        Lmins = Lmin*np.ones(len(zs))
        Lmaxs = Lmax*np.ones(len(zs))
        Aes = Ae*np.ones(len(zs))
    Ntots = N_ants*(N_ants + 1.)/2.
    
    #pl.figure()
    #pl.plot(zs,N_ants)
    #pl.xlabel('z')
    #pl.plot(zs,lambda_zs)
            
    
    fisher_grid = np.zeros((Nzs,Nphik,Nthetak,Nks))
    for i1,z in enumerate(zs):
        #if verbose_steps:
        print 'z=%.2f' % z
        N_ant = N_ants[i1]
        Ntot = Ntots[i1]
        lambda_z = lambda_zs[i1]
        Omega_beam = Omega_beams[i1]
        Lmin = Lmins[i1]
        Lmax = Lmaxs[i1]
        Ae = Aes[i1]
        t1 = tobs
        if Omega_patch>Omega_beam:
            t1 /= (Omega_patch / Omega_beam) 
        
        dA = cf.val_dA( z )
        H_z = cf.H( z )
        x1s = val_x1s( z )
        
        Tsys = val_Tsys( z )
        Tg = val_Tg( z ) #Tcmb * (1. + z)
        Tk = val_Tk( z )
        Jlya = val_Jlya( z )        
        Ts = val_Ts( z ) #old setup: val_Ts( z, Tk=Tk, Jlya=Jlya, x1s=x1s )
            
        Salpha = cf.val_Salpha(Ts, Tk, z, x1s, 0) #is this true that delta = 0 here???
        
        xalpha = rf.val_xalpha( Salpha=Salpha, Jlya=Jlya, Tg=Tg )
        xc = rf.val_xc(z, Tk=Tk, Tg=Tg)
        xBcoeff = ge * muB * Tstar / ( 2.*hbar * A * Tg ) 
                
        for i2,phik in enumerate(phiks):
            if verbose:
                print 'phik={:.2f}pi:'.format(phik/np.pi)
            for i3,thetak in enumerate(thetaks):
                #compute the polar position angle of k in the LOS-coordinate frame, from thetak, phik, thetan, and phin:
                thetak_n = np.arccos(np.sin(thetak)*np.cos(phik)*np.sin(thetan)*np.cos(phin) +
                      np.sin(thetak)*np.sin(phik)*np.sin(thetan)*np.sin(phin) +
                      np.cos(thetak)*np.cos(thetan))                  
                sint = np.sin(thetak_n)
                #phik_n = phik - phin #this is NOT correct!!

                #WRONG: if the direction we are integrating over is along the LOS, kmin and kmax are the limits imposed by the configuration of the survey:
                #if k is along LOS, both kmin and kmax are infinity, or as large as possible
                if np.isclose(sint, 0.):
                    kmin = kmaxmax #0.
                    kmax = kmaxmax #0.

                #otherwise, limits are:
                else:
                    rho_min = Lmin/lambda_z
                    kmin = rho_min*2.*np.pi/(dA*sint/Mpc_in_cm)
                    rho_max = Lmax/lambda_z
                    kmax = rho_max*2.*np.pi/(dA*sint/Mpc_in_cm)
                if verbose:
                    print 'thetak={:.2f}pi:  kmin={:.4f}, kmax={:.3f}'.format(thetak/np.pi,kmin,kmax)
                    #if np.isclose(phik,0.):
                    #    if thetak < np.pi/2.:
                    #        print 'thetak + theta_n = {:.3f} pi'.format((thetak+thetak_n)/np.pi)
                    #    else:
                    #        print 'thetak + theta_n = {:.3f} pi'.format((np.pi-thetak+thetak_n)/np.pi)
                #note: these kmin and kmax are in 1/Mpc comoving
                    
                for i4,k in enumerate(ks):
                    k_in_cm = k / Mpc_in_cm
                    nk = val_nk(k_in_cm, Ntot=Ntot, lambda_z=lambda_z, dA=dA, sin_thetak_n=sint,
                                Lmax=Lmax, Lmin=Lmin, DeltaL=DeltaL, Omega_beam=Omega_beam)
                    #if the considered k is in the range of observable k's, compute integrand, otherwise set it to zero:
                    if k<=kmax and k>=kmin and sint>0.:     
                        
                        res = Fisher_integrand(z, k, thetak=thetak, phik=phik, thetan=thetan, phin=phin,
                                               Ts=Ts, xalpha=xalpha, xc=xc, Tg=Tg, xBcoeff=xBcoeff,
                                               Tsys=Tsys, t1=t1, Ae=Ae, Lmax=Lmax, Lmin=Lmin, N_ant=N_ant, x1s=x1s, Omega_patch=Omega_patch,
                                               dA=dA, H_z=H_z, lambda_z=lambda_z, nk=nk)
                        if res < 0.:
                            print '--negative integrand value: {}, z={:.0f}, k={:.4f}, theta={:.2}, phi={:.2}\n'.format(res,z,k,thetak,phik)

                       
                    else:
                       res = 0.
                       #print res 
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
                     Tsys=1000, t1=10*86400., Ae=(3500)**2, Lmax=100000, Lmin=100, N_ant=1024, x1s=1., Omega_patch=0.1,
                     dA=3.16e+28, H_z=1.19e-16, lambda_z=443.1, nk=0.3, Tg=57.23508, verbose=False):
    """This calculates the integrand for estimating sigma of the zeta parameter.
        It takes k[1/Mpc comoving]. Result is returned in CGS units, [???].
    """

    k = k_in_Mpc/Mpc_in_cm
   
    Vpatch = Vpatch_factor( z, dA=dA, H_z=H_z, Omega_patch=Omega_patch )
    
    Pnoise = P21_N( dA=dA, H_z=H_z, z=z, Tsys=Tsys, t1=t1, Ae=Ae, 
                   Lmax=Lmax, Lmin=Lmin, lambda_z=lambda_z, nk=nk )
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
    



def write_Fisher_grid_zeta(root, val_nk=one_over_u_nk, val_Ts=rf.Ts_21cmfast_interp, val_Tk=rf.Tk_21cmfast_interp,
                           val_Jlya=rf.Jlya_21cmfast_interp, val_x1s=rf.ones_x1s,
                           z_lims=(15,30), Nzs=20, Nks=100, Nthetak=21, Nphik=22, phin=0., thetan=np.pi/2.,
                           val_Tsys=Tsys_simple, tobs=365.*86400, Ae=None, Lmax=None, Lmin=None, N_ant=None, Omega_patch=0.1,
                           kminmin=0.001, kmaxmax=100, val_Tg=rf.Tg_21cmfast_interp, DeltaL=100000):
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


    #total number of baselines, set as an effective number, as a function of lambda, if not specified:
    lambda_zs = lambda21 * (1. + zs)
    if (N_ant is None) or (Ae is None) or (Lmin is None) or (Lmax is None):
        N_ants = (DeltaL/lambda_zs)**2
        Omega_beams = np.ones(len(zs))
        Lmins = DeltaL / N_ants**0.5
        Lmaxs = DeltaL * np.ones(len(zs))
        Aes = lambda_zs**2
    else:
        N_ants = N_ant*np.ones(len(zs))
        Omega_beams = lambda_zs**2 / Ae
        Lmins = Lmin*np.ones(len(zs))
        Lmaxs = Lmax*np.ones(len(zs))
        Aes = Ae*np.ones(len(zs))
    Ntots = N_ants*(N_ants + 1.)/2.

    
    fisher_grid = np.zeros((Nzs,Nphik,Nthetak,Nks))
    for i1,z in enumerate(zs):
        print z
        N_ant = N_ants[i1]
        Ntot = Ntots[i1] #total number of baselines/antenna pairs
        lambda_z = lambda_zs[i1]
        Omega_beam = Omega_beams[i1]
        Lmin = Lmins[i1]
        Lmax = Lmaxs[i1]
        Ae = Aes[i1]
        t1 = tobs
        if Omega_patch>Omega_beam:
            t1 /= (Omega_patch / Omega_beam) 
        
        dA = cf.val_dA( z )
        H_z = cf.H( z )
        x1s = val_x1s( z )
        
        Tsys = val_Tsys( z )
        Tg = val_Tg( z ) #Tcmb * (1. + z)
        Tk = val_Tk( z )
        Jlya = val_Jlya( z )        
        Ts = val_Ts( z )
               
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

                    k_in_cm = k / Mpc_in_cm
                    nk = val_nk(k_in_cm, Ntot=Ntot, lambda_z=lambda_z, dA=dA, sin_thetak_n=sint, Lmax=Lmax, Lmin=Lmin, DeltaL=DeltaL, Omega_beam=Omega_beam)
                    

                    #WRONG: if the direction we are integrating over is along the LOS, kmin and kmax are the limits imposed by the configuration of the survey:
                    #if k is along LOS, both kmin and kmax are infinity, or as large as possible
                    if sint==0:
                        kmin = kmaxmax #0.
                        kmax = kmaxmax #0.

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
                                               Tsys=Tsys, t1=t1, Ae=Ae, Lmax=Lmax, Lmin=Lmin, N_ant=N_ant, x1s=x1s,  Omega_patch=Omega_patch,
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
    
    
