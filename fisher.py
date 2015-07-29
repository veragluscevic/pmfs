from __future__ import print_function

#import vegas
import math

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


val_Jlya = rf.Jlya_21cmfast_interp

val_Tg = rf.Tg_21cmfast_interp

val_Tk = rf.Tk_21cmfast_interp


def P21_N(z, dA=3.1695653382881036e+28, 
          H_z=1.1905643664441961e-16, 
          Tsys=1000, t1=365*86400., 
          Ae=3500**2, Lmax=1200000, 
          Lmin=1000, nk=0.37, 
          lambda_z=443.1):
    """ This takes k [1/cm comoving], thetak [rad], 
    and instrumental parameters:
    observation time t1[sec], 
    system temperature Tsys[K],
    effective area of an antenna Ae[cm^2], 
    minimum and maximum baselines, both in [cm],
    sin of angle between k and line of sight, 
    and the number density of baselines per k mode nk[unitless].
          
    It returns noise power in [K^2 (comoving cm)^3]. """

    res = dA**2 * c * (1+z)**2 * lambda_z**4 * Tsys**2 / ( Ae**2 * nu21 * t1 *nk * H_z )
    return res


def Tsys_Mao2008(z):
    """This takes redshift and returns system temperature in [K], 
    which is dominated by sky temperature
    from synchrotron radiation. 
    Formula taken from Mao 2008, paragraph after eq 33."""
    lambda_z = lambda21 * ( 1. + z )
    res = 60. * (lambda_z/100.)**2.55
    return res

def Vpatch_factor(z, dA=3.1695653382881036e+28, H_z=1.1905643664441961e-16, Omega_patch=0.3):
    """This is the volume element in the Fisher integral, 
    *for a uniform magnetic field* B=B0*(1+z)^2. 
    Vpath_factor*dz = dVpatch.
    It takes z, dA[cm comoving], H_z[1/sec], 
    and Omega_patch[sr], and returns result in [(comoving Mpc)^3]."""
    
    res = c / H_z * dA**2 * Omega_patch / Mpc_in_cm**3
    return res

def FFTT_nk(k, Ntot=None, lambda_z=443.1, dA=3.1695653382881032e+28,
            sin_thetak_n=0.7, Lmax=None, Lmin=None,
            DeltaL=100000,Omega_beam=1.):
    """ This gives the UV coverage density in 
    baselines per k mode, for uniform tiling by dipoles of a square
    on the ground with surface area (\DeltaL)^2; 
    the result is unitless, and averaged over the phik_n angle, 
    going between 0 and 2\pi.
    It takes k[1/cm comoving], 21cm wavelength at z, 
    assumes effective area that is = lambda^2 [cm^2], 
    ang. diam. distance in [comoving cm], 
    sin of vector k wrt line of sight, 
    and its azimuthal angle wrt n, phik_n.
    """

    var1 = DeltaL / lambda_z
    #this is WRONG: var2 = k* dA * sin_thetak_n
    #this is WRONG: res = var1**2 - 4.*var1*var2/np.pi + var2**2/np.pi
    var2 = k* dA * sin_thetak_n / (2.*np.pi)
    res = var1**2 - 4./np.pi*var1*var2 + var2**2
    
    return res


##################
def integrand(x,#z=30,k_Mpc=0.1,thetak=0.1, phik=0.2,
              root=RESULTS_PATH, 
              val_nk=FFTT_nk,
              val_Ts=rf.Ts_21cmfast_interp, 
              val_Tk=rf.Tk_21cmfast_interp, 
              val_Jlya=rf.Jlya_21cmfast_interp, 
              val_x1s=rf.ones_x1s,
              val_Tsys=Tsys_Mao2008,
              val_Tg=rf.Tg_21cmfast_interp,
              phin=0., thetan=np.pi/2.,
              t_yr=1., Omega_patch=1.,
              DeltaL_km=2.,
              print_klims=False,
              debug=False,
              mode='B0'):
    """This takes k[1/Mpc comoving]. 
    Result is returned in CGS units."""
    
    z = x[0]
    k_Mpc = x[1]
    thetak = x[2]
    phik = x[3]
    
    thetak_n = np.arccos(np.sin(thetak)*np.cos(phik)*np.sin(thetan)*np.cos(phin) + np.sin(thetak)*np.sin(phik)*np.sin(thetan)*np.sin(phin) + np.cos(thetak)*np.cos(thetan))                  
    sint = np.sin(thetak_n)
    if np.isclose(sint, 0.):
        if debug:
            return 0,0,0
        return 0.

    
    DeltaL_cm = DeltaL_km * 1e5 # cm
    lambda_z = lambda21 * ( 1. + z ) # cm
    dA = cf.val_dA( z ) # cm 
    
    kmin = 2.*np.pi/(dA*sint/Mpc_in_cm) # 1/Mpc comoving
    kmax = kmin * DeltaL_cm / lambda_z # 1/Mpc comoving
    
    if print_klims:
        print('kmax={}, kmin={}'.format(kmax,kmin))
    if k_Mpc > kmax or k_Mpc < kmin:
        if debug:
            return 0,0,0
        return 0.
                   
    t_sec=365.*86400*t_yr # sec
    k_cm = k_Mpc/Mpc_in_cm # cm
    N_ants = ( DeltaL_cm / lambda_z )**2
    Ntot = N_ants * ( N_ants + 1. ) / 2.
    
    nk = val_nk(k_cm, Ntot=Ntot, 
                lambda_z=lambda_z, 
                dA=dA, 
                sin_thetak_n=sint,
                Lmax=DeltaL_cm, Lmin=lambda_z, 
                DeltaL=DeltaL_cm,
                Omega_beam=1.)

    t1 = t_sec
    if Omega_patch > 1.:
        t1 = t_sec / Omega_patch
        
    H_z = cf.H( z )
    Tsys = val_Tsys( z )
    Ts = val_Ts( z )
    Tg = val_Tg( z )
    Tk = val_Tk( z )
    Jlya = val_Jlya( z )        
    Salpha = cf.val_Salpha(Ts, Tk, z, 1., 0) 
    xalpha = rf.val_xalpha( Salpha=Salpha, Jlya=Jlya, Tg=Tg )
    xc = rf.val_xc(z, Tk=Tk, Tg=Tg)
    xBcoeff = ge * muB * Tstar / ( 2.*hbar * A * Tg ) 

    
    Vpatch = Vpatch_factor( z, dA=dA, H_z=H_z, 
                            Omega_patch=Omega_patch )
    Pnoise = P21_N( dA=dA, H_z=H_z, 
                    z=z, Tsys=Tsys, 
                    t1=t1, Ae=lambda_z**2, 
                    Lmax=DeltaL_cm, Lmin=lambda_z, 
                    lambda_z=lambda_z, nk=nk )
    if np.isnan( Pnoise ):
        raise ValueError( 'Pnoise is nan.' )

    
    Pdelta = cf.val_Pdelta( z, k_Mpc )  

    if mode=='B0':
        G = pt.calc_G(thetak=thetak, phik=phik, 
                      thetan=thetan, phin=phin,
                      Ts=Ts, Tg=Tg, z=z, 
                      verbose=False, 
                      xalpha=xalpha, xc=xc, xB=0., x1s=1.)

        dGdB = pt.calc_dGdB(thetak=thetak, phik=phik, 
                            thetan=thetan, phin=phin,
                            Ts=Ts, Tg=Tg, z=z, 
                            verbose=False, 
                            xalpha=xalpha, xc=xc, 
                            xBcoeff=xBcoeff, x1s=1.)

        Psignal = Pdelta*G**2
        Numerator = (2.*G*dGdB*Pdelta)**2
        Denominator = 2.*(Psignal + Pnoise)**2

    if mode=='zeta':
        G_Bzero = pt.calc_G_Bzero(thetak=thetak, phik=phik, 
                                  thetan=thetan, phin=phin,
                                  Ts=Ts, Tg=Tg, z=z, 
                                  verbose=False, 
                                  xalpha=xalpha, xc=xc, 
                                  x1s=1.)
        G_Binfinity = pt.calc_G_Binfinity(thetak=thetak, phik=phik, 
                                          thetan=thetan, phin=phin,
                                          Ts=Ts, Tg=Tg, z=z, 
                                          verbose=False, 
                                          xalpha=xalpha, xc=xc, 
                                          x1s=1.)

        Psignal_Bzero = Pdelta*G_Bzero**2
        Psignal_Binfinity = Pdelta*G_Binfinity**2
        Numerator = (Psignal_Binfinity - Psignal_Bzero)**2
        Denominator = 2.*(Psignal_Bzero + Pnoise)**2

    
    res = k_Mpc**2*np.sin(thetak)* Vpatch * Numerator/Denominator/(2.*np.pi)**3 

    if debug:
        return res,Pnoise,nk,Numerator,G,dGdB
    return res
 

def nest_integrator(neval=100, Nz=20,
                    DeltaL_km=2.,
                    kminmin=0.0001,kmaxmax=10.,
                    zmax=35,zmin=10,
                    Omega_survey=1.,
                    Omega_patch=1.,
                    thetan=np.pi/2.,phin=0.):

    DeltaL_cm = DeltaL_km * 1e5
    zs = np.linspace(zmin, zmax, Nz)
    thetas = np.linspace(0.,np.pi,neval)
    phis = np.linspace(0.,2.*np.pi,neval)
    ks = np.zeros(neval)

    samplesz = np.zeros(Nz)
    samplesp = np.zeros(neval)
    samplest = np.zeros(neval)
    samplesk = np.zeros(neval)
    for i,z in enumerate(zs):
        print(z)
        dA = cf.val_dA( z ) 
        lambda_z = lambda21 * ( 1. + z ) 
        for j,phik in enumerate(phis):
            for l,thetak in enumerate(thetas):
                thetak_n = np.arccos(np.sin(thetak)*np.cos(phik)*np.sin(thetan)*np.cos(phin) +
                      np.sin(thetak)*np.sin(phik)*np.sin(thetan)*np.sin(phin) +
                      np.cos(thetak)*np.cos(thetan))                  
                sint = np.sin(thetak_n)
                kmin = 2.*np.pi/(dA*sint/Mpc_in_cm) # 1/Mpc comoving
                kmax = kmin * DeltaL_cm / lambda_z # 1/Mpc comoving
                ks = np.linspace(kmin,kmax,neval)
                kVol = kmax - kmin
                samplesk = np.zeros(neval)
                for m,k in enumerate(ks):
                    x = np.array([z,k,thetak,phik])
                    samplesk[m] = integrand(x,

                    DeltaL_km=DeltaL_km,
                                           Omega_patch=Omega_patch)
                samplest[l] = samplesk.mean() * kVol
            tVol = np.pi
            samplesp[j] = samplest.mean() * tVol
        pVol = 2.*np.pi
        samplesz[i] = samplesp.mean() * pVol

    zVol = zmax - zmin
    res = samplesz.mean() * zVol

    
    alpha_survey = (Omega_survey)**0.5 
    result_all_survey = res / Omega_patch * np.pi * (alpha_survey + np.cos(alpha_survey)*np.sin(alpha_survey))

    result = 1./result_all_survey**0.5

    return result






def rand_integrator(neval=1000, DeltaL_km=2.,
                    kminmin=0.01,kmaxmax=1.,
                    zmax=35,zmin=15,
                    Omega_survey=1.,
                    Omega_patch=1.,
                    thetan=np.pi/2.,phin=0.,
                    mode='B0'):

    zs = np.random.random(size=neval)*(zmax - zmin) + zmin
    thetas = np.random.random(size=neval)*np.pi
    phis = np.random.random(size=neval)*2.*np.pi
    ks = np.random.random(size=neval)*(kmaxmax - kminmin) + kminmin

    Volume = (zmax - zmin) * (kmaxmax - kminmin) * 2.* np.pi**2

    xlist = []
    for i in np.arange(neval):
        xlist.append(np.array([zs[i],ks[i],thetas[i],phis[i]]))
    xs = np.array(xlist)

    samples = np.zeros(neval)
    for i,x in enumerate(xs):
        samples[i] = integrand(x,mode=mode,
                               DeltaL_km=DeltaL_km,
                               Omega_patch=Omega_patch)
        #print(samples[i])

    result = samples.mean()*Volume
    alpha_survey = (Omega_survey)**0.5 
    result_all_survey = result / Omega_patch * np.pi * (alpha_survey + np.cos(alpha_survey)*np.sin(alpha_survey))

    res = 1./result_all_survey**0.5
    return res


def vegas_integrator(neval=1000,nitn=10,
                     DeltaL_km=2.,
                     kmin=0.001,kmax=2.,
                     zmax=35,zmin=15,
                     Omega_survey=1.,
                     Omega_patch=1.,
                     mode='B0'):

    lims=[[zmin, zmax], [kmin, kmax], [0, np.pi], [0, 2.*np.pi]]
    integ = vegas.Integrator(lims)

    #fun = integrand(DeltaL_km=DeltaL_km,
    #                Omega_patch=Omega_patch)
    result = integ(integrand, nitn=nitn, neval=neval)
    print(result.summary())
    print('result = %s    Q = %.5f' % (result, result.Q))

    alpha_survey = (Omega_survey)**0.5 
    #result_all_survey = result / Omega_patch * np.pi * (alpha_survey + np.cos(alpha_survey)*np.sin(alpha_survey))

    #res = 1./result_all_survey**0.5
    
    #return res


def test_integrand_k(z=30,Nks=1000,
                   kmin=0.0001,kmax=10,
                   thetak=0.1,phik=0.,logplot=True,
                   mode='B0'):

    res = np.zeros(Nks)
    G = np.zeros(Nks)
    dGdB = np.zeros(Nks)
    ks = np.linspace(kmin,kmax,Nks)
    for i,k in enumerate(ks):
        x = np.array([z,k,thetak,phik])
        res[i] = integrand(x,mode=mode)

    if logplot:
        pl.semilogy(ks,res,label='thetak={:.2f},phik={:.2f},z={:.1f}'.format(thetak,phik,z))
    else:
        pl.plot(ks,res,label='thetak={:.2f},phik={:.2f},z={:.1f}'.format(thetak,phik,z))
    pl.xlabel('k')
    print(ks[res.argmax()],res.max())
    return ks,res

def test_integrand_theta(z=30,Ns=1000,
                         k=0.1,phik=0.,
                         mode='B0'):

    res = np.zeros(Ns)
    G = np.zeros(Ns)
    dGdB = np.zeros(Ns)
    thetas = np.linspace(0.,np.pi,Ns)
    for i,thetak in enumerate(thetas):
        x = np.array([z,k,thetak,phik])
        res[i] = integrand(x,mode=mode)

    #pl.figure()
    pl.plot(thetas,res,label='k={},phik={:.2f},z={:.1f}'.format(k,phik,z))
    pl.xlabel('thetak')
    print(thetas[res.argmax()],res.max())
    return thetas,res

def test_integrand_phi(z=30,Ns=1000,
                         k=0.1,thetak=0.2,
                         mode='B0'):

    res = np.zeros(Ns)
    G = np.zeros(Ns)
    dGdB = np.zeros(Ns)
    phis = np.linspace(0.,np.pi,Ns)
    for i,phik in enumerate(phis):
        x = np.array([z,k,thetak,phik])
        res[i] = integrand(x,mode=mode)

    #pl.figure()
    pl.plot(phis,res,label='k={},thetak={:.2f},z={:.1f}'.format(k,thetak,z))
    pl.xlabel('phik')
    print(phis[res.argmax()],res.max())
    return phis,res


def test_integrand_z(Ns=1000,
                     zmin=10,zmax=35,
                     k=0.1,thetak=0.2,phik=0.3,
                     mode='B0'):

    res = np.zeros(Ns)
    G = np.zeros(Ns)
    dGdB = np.zeros(Ns)
    num = np.zeros(Ns)
    den = np.zeros(Ns)
    pnoise = np.zeros(Ns)
    psig = np.zeros(Ns)
    ns = np.zeros(Ns)

    #print(pnoise,ns)
    zs = np.linspace(zmin,zmax,Ns)
    for i,z in enumerate(zs):
        x = np.array([z,k,thetak,phik])
        #res[i],G[i],dGdB[i],num[i],den[i],pnoise[i],psig[i] = integrand(x,debug=True)
        #res[i],pnoise[i],ns[i] = integrand(x,debug=True,mode=mode)3
        res[i] = integrand(x,mode=mode)

    #pl.figure()
    pl.semilogy(zs,res,label='k={},thetak={:.2f},phik={:.2f}'.format(k,thetak,phik))
    pl.xlabel('z')
    print(zs[res.argmax()],res.max())
    #return zs,res,pnoise,ns
    return zs,res
