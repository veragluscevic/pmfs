# This module contains 21-cm-cosmology-related functions, handeled in CGS units.

import numpy as np
from scipy.interpolate import UnivariateSpline as interpolate

import cosmo_functions as cf
#import pmfs_fisher as pf
reload(cf)
from constants import *
from globals import *

#set up kappas:
KAPPA = np.loadtxt(INPUTS_PATH + 'Kcoeffexpanded.txt')
TKAPPA = KAPPA[:,0]
KAPPAP = KAPPA[:,1]
KAPPAM = KAPPA[:,2]

val_kappam = interpolate(TKAPPA, KAPPAM, s=0)
val_kappap = interpolate(TKAPPA, KAPPAP, s=0)

#from Allison & Delgarno 1969:
kappa10_array = np.array([2.2e-14,4.2e-14,1.8e-13,5.1e-13,1.2e-12,2.3e-12,7.4e-12,1.5e-11,2.3e-11,3.0e-11,4.4e-11,5.6e-11,6.6e-11,7.4e-11,8.2e-11,8.9e-11,9.5e-11,1.4e-10,1.6e-10,1.8e-10,2.0e-10,2.1e-10,2.2e-10,2.3e-10,2.4e-10,2.5e-10])
T_kappa10_array = np.array([1,2,4,6,8,10,15,20,25,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000])
val_kappa10 = interpolate(T_kappa10_array, kappa10_array, s=0)

Pnp = np.array([1, 0, 0.2609, 0.3078, 0.3259, 0.3353, 0.3410, 0.3448, 0.3476, 0.3496,
       0.3512, 0.3524, 0.3535, 0.3543, 0.3550, 0.3556, 0.3561, 0.3565, 0.3569,
       0.3572, 0.3575, 0.3578, 0.3580, 0.3582, 0.3584, 0.3586, 0.3587, 0.3589, 0.3590])
Pnp_ns = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])

Pnp_tuple = ((2,1.), (3,0), (4,0.2609), (5,0.3078), (6,0.3259), (7,0.3353), (8,0.3410), (9,0.3448),
       (10,0.3476), (11,0.3496), (12,0.3512), (13,0.3524), (14,0.3535), (15,0.3543), (16,0.3550),
       (17,0.3556), (18,0.3561), (19,0.3565), (20,0.3569), (21,0.3572), (22,0.3575), (23,0.3578),
       (24,0.3580), (25,0.3582), (26,0.3584), (27,0.3586), (28,0.3587), (29,0.3589), (30,0.3590))


#file_21cmfast = np.loadtxt(INPUTS_PATH+'21cmfast_teja_nov2014.txt')
#file_21cmfast = np.loadtxt(INPUTS_PATH+'global_evolution_zetaIon31.50_Nsteps40_zprimestepfactor1.020_zetaX1.0e+56_alphaX1.2_TvirminX1.0e+04_Pop2_300_200Mpc___default')
file_21cmfast = np.loadtxt(INPUTS_PATH+'global_evolution_zetaIon31.50_Nsteps40_zprimestepfactor1.020_zetaX1.0e+56_alphaX1.2_TvirminX1.0e+04_Pop3_300_200Mpc')
file_21cmfast_noheat = np.loadtxt(INPUTS_PATH+'21cmfast_teja_nov2015_noheat.txt')
#file_21cmfast_hiheat = np.loadtxt(INPUTS_PATH+'21cmfast_teja_nov2015_hiheat.txt')
Tks_21cmfast = file_21cmfast[:,2][::-1]
Tgs_21cmfast = file_21cmfast[:,5][::-1]
Tss_21cmfast = file_21cmfast[:,4][::-1]
Jlyas_21cmfast = file_21cmfast[:,6][::-1]
Jlyas_21cmfast_noheat = file_21cmfast_noheat[:,6][::-1]
#Jlyas_21cmfast_hiheat = file_21cmfast_hiheat[:,6][::-1]
zs_21cmfast = file_21cmfast[:,0][::-1]
zs_21cmfast_noheat = file_21cmfast_noheat[:,0][::-1]
#zs_21cmfast_hiheat = file_21cmfast_hiheat[:,0][::-1]

Tk_21cmfast_interp = interpolate(zs_21cmfast, Tks_21cmfast, s=0)
Tg_21cmfast_interp = interpolate(zs_21cmfast, Tgs_21cmfast, s=0)
Ts_21cmfast_interp = interpolate(zs_21cmfast, Tss_21cmfast, s=0)

Jlya_21cmfast_interp = interpolate(zs_21cmfast, Jlyas_21cmfast, s=0)
Jlya_21cmfast_noheat_interp = interpolate(zs_21cmfast_noheat, Jlyas_21cmfast_noheat, s=0)
#Jlya_21cmfast_hiheat_interp = interpolate(zs_21cmfast_hiheat, Jlyas_21cmfast_hiheat, s=0)


def calc_simple_Jlya( z ):
    """ This returns Lyman-alpha flux as a function of redshift, Jlya(z) [# of photons/cm^2/sec/Hz/sr],
    as calculated using Chen and Miralda-Escude 2004 Eq. (16). """
    res = 1.65*1e-13 * ( 1 + z )**3 * ( Obaryonh2 / 0.02 )
    return res

def Tk_simple( z ):
    """This is the eyeballed kinetic temperature of the gas from the 21cmfast paper,
    for one of "extreme heating" models.
    It takes only z and returns Tk [K]."""

    k_l = -0.1872
    n_l = 4.6499 
    k_r = 0.0257
    n_r = 0.4994
    
    #zs = np.atleast_1d(z)
    #res = np.zeros(len(zs))
    if z  >= 19.5:
        n = n_r
        k = k_r
    else:
        n = n_l
        k = k_l
    res = 10**(k*z+n)
    #res[np.where(zs >= 19.5)] = 10**(k_r*zs[np.where(zs >= 19.5)] + n_r)
    #res[np.where(zs < 19.5)] = 10**(k_l*zs[np.where(zs >= 19.5)] + n_l)

    return res

def Tk_1000K( z ):
    """
    This just returns Tk=1000 K at any z.
    """
    return 1000.


def ones_x1s(z):
    """Neutral function as a function of redshift."""
    return z/z

def val_Tceff(Tk, Ts):
    """This returns Tc^{eff} in [K], from Eq. (42) in Hirata 2005, as a function of Tk[K] and Ts[K]."""

    res = 1. / ( 1./Tk + 0.405535/Tk*(1./Ts - 1./Tk) )
    return res


def Ts_Hirata05( z, Tk=10., Jlya=1.7e-9, x1s=1. ):
    """This takes z, Tk[K], and Jlya[# of photons/cm^2/sec/Hz/sr],
     and returns spin temperature in [K], according to Eq (39) of Hirata 2005.
     xc is calculated from eq 9, xalpha from eq 38, and Salpha from eq 40."""
    niterations = 1000
    
    Tg = Tcmb * ( 1 + z )
    Ts = Tg*1.

    xc = val_xc_tilde(z, Tk=Tk, Tg=Tg)
     
    for j in np.arange(niterations):
        
        Salpha = cf.val_Salpha(Ts, Tk, z, x1s, 0)
        xalpha = val_xalpha_tilde(Salpha=Salpha, Jlya=Jlya, Tg=Tg)
        Tceff = val_Tceff(Tk, Ts)

        Ts = ( 1. + xalpha + xc ) / ( 1./Tg + xalpha/Tceff + xc/Tk )

    return Ts


def val_xc_tilde(z, Tk=10., Tg=57.23):
    """For calculation of Ts, quantity from Hirata paper.
    This takes z, Tk[K], T_gamma at a given z also in [K], and returns xc [unitless]."""

    #kappap = val_kappap(Tk)
    #kappam = val_kappam(Tk)
    #res = 0.5*(kappap + kappam)*cf.nH(z,0)*Tstar/(A*Tg)
    kappa10 = val_kappa10(Tk)
    res = kappa10*cf.nH(z,0)*Tstar/(A*Tg)
    return res

def val_xc(z, Tk=10., Tg=57.23):
    """Quantity from our paper. 
    This takes z, Tk[K], T_gamma at a given z also in [K], and returns xc [unitless]."""

    #kappap = val_kappap(Tk)
    #kappam = val_kappam(Tk)
    #res =  2.*( val_kappap(Tk) + val_kappam(Tk) ) * cf.nH(z,0) * Tstar / ( A * Tg )
    res =  4.*val_kappa10(Tk) * cf.nH(z,0) * Tstar / ( A * Tg )  
    return res

def val_xalpha_tilde(Salpha=0.6, Jlya=1.7e-9, Tg=57.23):
    """For calculation of Ts, quantity from Hirata 2005 paper, eq 38.
    This takes tilde Salpha[unitless] from his eq 40, Jlya[# of photons/cm^2/sec/Hz/sr], and T_gamma[K] at z, and returns xalpha[unitless]"""

    res =  8./9.*np.pi * lambdaLya**2. * gamma * Tstar * Salpha * Jlya / ( A * Tg )
    return res


def val_xalpha(Salpha=0.6, Jlya=1.7e-9, Tg=57.23):
    """Quantity from our paper. This takes Salpha[unitless], Jlya[# of photons/cm^2/sec/Hz/sr], and T_gamma[K] at z, and returns xalpha[unitless]"""

    res =  3.607*np.pi * lambdaLya**2. * gamma * Tstar * Salpha * Jlya / ( A * Tg )
    return res




def df_sigma_dz(z, sigma=1.6, A0=0.186, Apow=-0.14, a0=1.47, apow=-0.06, alpha=0.48446733136638109, b0=2.57, c_coeff=1.19 ):
    """
    
    A = A0 * (1 + z)**Apow
    a = a0 * (1 + z)**apow
    b = b0 / (1 + z)**alpha
    """
    subexpression = alpha - apow * np.log(b0 / (1. + z)**alpha / sigma)
    parenths = Apow + (b0 / (1. + z)**alpha / sigma)**(a0 * (1. + z)**apow) * (Apow - a0 * (1. + z)**apow * subexpression)
    res = A0 / np.exp(c_coeff / sigma**2) * (1. + z)**(Apow - 1.) * parenths

    
    return res

def f_sigma(z, sigma=1.6, A0=0.186, Apow=-0.14, a0=1.47, apow=-0.06, alpha=0.48446733136638109, b0=2.57, c_coeff=1.19 ):
    """
    """
    A_coeff = A0 * (1 + z)**Apow
    a = a0 * (1 + z)**apow
    b = b0 / (1 + z)**alpha
    
    res = A_coeff / np.exp(c_coeff / sigma**2) * ( (b / sigma)**a + 1. )

    return res


def alphacoefficient_in_Jlya_Hirata2005(Delta_vir=200):
    """
    """
    exponent = -(0.75 / np.log(Delta_vir / 75.) )**(1.2)
    res = np.exp(exponent)
    return res


def rho_matter_z(z, rho_matter_0=Omatter*rho_critical):
    """
    """
    res = rho_matter_0 * (1. + z)**3
    return res

def drho_matter_dz(z, rho_matter_0=Omatter*rho_critical):
    """
    """
    res = 3.*rho_matter_0 * (1. + z)**2
    return res

def d_integral_dt(z, sigma=1.6, A0=0.186, Apow=-0.14, a0=1.47, apow=-0.06,
                    alpha=0.48446733136638109, b0=2.57, c_coeff=1.19,
                    rho_matter_0=Omatter*rho_critical, f_star=2.5e-4, ind=-1.5,
                    Mmin=4e6, Mmax=1e13):
    """
    This takes Mmin and Mmax halo masses in Msolar.
    """
    dlnsigma_dlnM = (ind + 3.) / 6.
    derivatives = df_sigma_dz(z, sigma=sigma, A0=A0, Apow=Apow, a0=a0, apow=apow, alpha=alpha, b0=b0, c_coeff=c_coeff) * rho_matter_z(z) + drho_matter_dz(z, rho_matter_0=rho_matter_0) * f_sigma(z, sigma=sigma, A0=A0, Apow=Apow, a0=a0, apow=apow, alpha=alpha, b0=b0, c_coeff=c_coeff)
    res = f_star * np.log(Mmax / Mmin) * dlnsigma_dlnM * derivatives * (1. + z)**(5./2.) * (2./3.) / t_universe
    #print derivatives
    return res
    
def epsilon_b(nu, T=1e5):
    """
    Result is in [sec]. Takes T[K].
    """
    res = 15.*h**3*nu**2 / (np.exp(h*nu / kb*T) - 1.) / (np.pi*kb*T)**4 * 5.4e6 / erg_in_ev
    return res


def epsilon_nu_z(z, nu, sigma=1.6, A0=0.186, Apow=-0.14, a0=1.47, apow=-0.06,
                    alpha=0.48446733136638109, b0=2.57, c_coeff=1.19,
                    rho_matter_0=Omatter*rho_critical, f_star=2.5e-4, ind=-1.5,
                    Mmin=4e6, Mmax=1e13, T=1e5, Omega_matter=Omatter, Omega_baryon=Obaryonh2/h0**2):
    """
    This takes Mmin and Mmax halo masses in Msolar, and T[K].
    """
    #print epsilon_b(nu, T=T) * Omega_baryon / Omega_matter / mp
    res = epsilon_b(nu, T=T) * Omega_baryon / Omega_matter / mp * d_integral_dt(z, sigma=sigma,
                                                                               A0=A0, Apow=Apow,
                                                                               a0=a0, apow=apow,
                                                                               alpha=alpha, b0=b0,
                                                                               c_coeff=c_coeff,
                                                                               rho_matter_0=rho_matter_0,
                                                                               f_star=f_star, ind=ind,
                                                                               Mmax=Mmax, Mmin=Mmin)
    return res


def zmax_in_Jlya(z, n):
    """
    """
    res = (1. + z) * (1. - 1./(n + 1.)**2) / (1. - 1./n**2) - 1.
    return res
        
def Jlya_Hirata2005_integrand(z, nu_n, sigma=1.6, A0=0.186, Apow=-0.14, a0=1.47, apow=-0.06,
                                alpha=0.48446733136638109, b0=2.57, c_coeff=1.19,
                                rho_matter_0=Omatter*rho_critical, f_star=2.5e-4, ind=-1.5,
                                Mmin=4e6, Mmax=1e13, T=1e5, Omega_matter=Omatter, Omega_baryon=Obaryonh2/h0**2):
    """
    """
    res = c / cf.H(z) * epsilon_nu_z(z, nu_n, sigma=sigma,A0=A0, Apow=Apow,a0=a0, apow=apow,alpha=alpha,
                                     b0=b0,c_coeff=c_coeff,rho_matter_0=rho_matter_0,f_star=f_star,
                                     ind=ind,Mmax=Mmax, Mmin=Mmin, T=T, Omega_matter=Omega_matter, Omega_baryon=Omega_baryon)
    return res        


"""
def calc_Jlya_Hirata2005(z, sigma=1.6, A0=0.186, Apow=-0.14, a0=1.47, apow=-0.06,
                        alpha=0.48446733136638109, b0=2.57, c_coeff=1.19,
                        rho_matter_0=Omatter*rho_critical, f_star=2.5e-4, ind=-1.5,
                        Mmin=4e6, Mmax=1e13, T=1e5, Omega_matter=Omatter, Omega_baryon=Obaryonh2/h0**2,
                        points=100):
    
    #This calculates Jlya [cgs] as a function of z, following Hirata 2005.
    #Assumptions: ...
    #References: ...
    
    suma = 0.
    for n,Pnp in Pnp_tuple:
        zmax = zmax_in_Jlya(z, n)
        zs = np.linspace(z, zmax, points)
        nu_n = nu_n_Lya(n,zs,z)#c/RH * (1. - 1./n**2) * (1. + zs) / (1. + z)
        #if n==2:
        #    print zmax
        #    print nu_n
        
        integrand_grid = Jlya_Hirata2005_integrand(zs, nu_n, sigma=sigma,A0=A0, Apow=Apow,a0=a0, apow=apow,alpha=alpha,
                                                b0=b0,c_coeff=c_coeff,rho_matter_0=rho_matter_0,f_star=f_star,
                                                ind=ind,Mmax=Mmax, Mmin=Mmin, T=T,
                                                Omega_matter=Omega_matter, Omega_baryon=Omega_baryon)
        integral = pf.trapznd(integrand_grid,zs)
        #print integral
        suma += Pnp*integral
    res = (1. + z)**2 / (4.*np.pi) * suma
    return res
"""

def nu_n_Lya(n, z_prime, z):
    """
    """
    res = c/RH * (1. - 1./n**2) * (1. + z_prime) / (1. + z)
    return res


def Gamma_X(z, sigma=1.6, A0=0.186, Apow=-0.14, a0=1.47, apow=-0.06,
                    alpha=0.48446733136638109, b0=2.57, c_coeff=1.19,
                    rho_matter_0=Omatter*rho_critical, f_star=2.5e-4, ind=-1.5,
                    Mmin=4e6, Mmax=1e13, Omega_matter=Omatter, Omega_baryon=Obaryonh2/h0**2,
                    f_Gamma=0.14, fxeEx=27.):
    """
    It takes fxeEx in [keV].

    """
    res = f_Gamma*(fxeEx/erg_in_ev*1e3)* Omega_baryon / Omega_matter / mp * d_integral_dt(z, sigma=sigma,
                                                            A0=A0, Apow=Apow,
                                                            a0=a0, apow=apow,
                                                            alpha=alpha, b0=b0,
                                                            c_coeff=c_coeff,
                                                            rho_matter_0=rho_matter_0,
                                                            f_star=f_star, ind=ind,
                                                            Mmax=Mmax, Mmin=Mmin)
    return res

def Tk_Hirata2005_integrand(zp, z, sigma=1.6, A0=0.186, Apow=-0.14, a0=1.47, apow=-0.06,
                    alpha=0.48446733136638109, b0=2.57, c_coeff=1.19,
                    rho_matter_0=Omatter*rho_critical, f_star=2.5e-4, ind=-1.5,
                    Mmin=4e6, Mmax=1e13, T=1e5, Omega_matter=Omatter, Omega_baryon=Obaryonh2/h0**2,
                    f_Gamma=0.14, fxeEx=27.):
    """
    It takes fxeEx in [keV].
    """
    res = (1. + z)**2 / (1. + zp)**3 / cf.H(zp) * Gamma_X(zp, sigma=sigma,
                                                    A0=A0, Apow=Apow,
                                                    a0=a0, apow=apow,
                                                    alpha=alpha, b0=b0,
                                                    c_coeff=c_coeff,
                                                    rho_matter_0=rho_matter_0,
                                                    f_star=f_star, ind=ind,
                                                    Mmax=Mmax, Mmin=Mmin,
                                                    Omega_matter=Omega_matter, Omega_baryon=Omega_baryon,
                                                    f_Gamma=f_Gamma, fxeEx=fxeEx)
    return res

"""
def calc_Tk_Hirata2005(z, z0, Tk0=100, sigma=1.6, A0=0.186, Apow=-0.14, a0=1.47, apow=-0.06,
                    alpha=0.48446733136638109, b0=2.57, c_coeff=1.19,
                    rho_matter_0=Omatter*rho_critical, f_star=2.5e-4, ind=-1.5,
                    Mmin=4e6, Mmax=1e13, Omega_matter=Omatter, Omega_baryon=Obaryonh2/h0**2,
                    f_Gamma=0.14, fxeEx=27., mu=1.22, rho_baryon_0=Obaryonh2/h0**2*rho_critical,points=1000):
    
    #It takes fxeEx in [keV], and Tk0 [K].

    

    zps = np.linspace(z, z0, points)
    integrand_grid = 2.*mu*mp / (3.*kb*rho_baryon_0) * Tk_Hirata2005_integrand(zps, z, sigma=sigma,A0=A0,
                                                                    Apow=Apow,a0=a0, apow=apow,alpha=alpha,
                                                                    b0=b0,c_coeff=c_coeff,rho_matter_0=rho_matter_0,f_star=f_star,
                                                                    ind=ind,Mmax=Mmax, Mmin=Mmin,
                                                                    Omega_matter=Omega_matter, Omega_baryon=Omega_baryon,
                                                                    f_Gamma=f_Gamma, fxeEx=fxeEx)
    
    integral = pf.trapznd(integrand_grid,zps)
    res = (1. + z)**2 / (1. + z0)**2 *Tk0 + integral
    return res
"""
