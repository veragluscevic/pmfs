#This module has the functions that calculate the transfer function, its derivatives, and the brightness temperature.

import numpy as np
import cosmo_functions as cf
from constants import *
from geometric_functions import *

def calc_Tb(thetak=np.pi/3., phik=np.pi/8., thetan=np.pi/3., phin=np.pi/4., 
            delta=0., Ts=11.1, Tg=57.23508, z=20, verbose=False,
            xalpha=34.247221, xc=0.004176, xB=0.365092, x1s=1.):
    
    """ Calculates brightness temperature Tb[K] from eq 138 of v3 draft of microphysics paper. 
    NOTE: Magnetic-field direction is along the z-axis!  It takes x's (all unitless), temperatures in [K], and angles in [rad]."""
    
    k_dot_n = np.cos(thetan)*np.cos(thetak) + np.sin(thetan)*np.sin(thetak)*np.cos(phin)*np.cos(phik) + np.sin(thetan)*np.sin(thetak)*np.sin(phin)*np.sin(phik)

    summ = 0.
    for i,m in enumerate( np.array([-2,-1,0,1,2]) ):
        summand = Y2( m,thetak,phik ) * np.conjugate( Y2( m,thetan,phin ) ) / (1. + xalpha + xc - 1j*m*xB)
        summ += summand.real

    first_term = 1 + delta + delta*k_dot_n**2
    second_term = 1 + 2.*delta + 2.*delta*k_dot_n**2 - delta*4.*np.pi/75.*summ
    
    res = x1s * ( 1 - Tg/Ts ) * np.sqrt( (1 + z)/10. ) * ( 26.4 * first_term - 0.128 * x1s * ( Tg/Ts ) * np.sqrt( (1 + z)/10. ) * second_term)
    
    if verbose:
        print '\n'
        print 'xalpha = %f' % xalpha
        print 'xc = %f' % xc
        print 'xB = %f' % xB
        print 'k_dot_n=%f' % k_dot_n
        print 'summ=%f' % summ
        print 'first=%f' % 26.4*first_term
        print 'second=%f' % second_term
        
    return res/1000. #this is to make it to K from mK.



    
def calc_deltaTb(thetak=np.pi/3., phik=np.pi/8., thetan=np.pi/3., phin=np.pi/4., 
            delta=0., Ts=11.1, Tg=57.23508, z=20, verbose=False,
            xalpha=34.247221, xc=0.004176, xB=0.365092, x1s=1.):
    
    """ Calculates the fluctuations in brightness temperature Tb[K] from eq 138 of v3 draft of microphysics paper. 
    NOTE: Magnetic-field direction is along the z-axis!  It takes x's (all unitless), temperatures in [K], and angles in [rad]."""
    
    res0 = calc_Tb(thetak=thetak, phik=phik, thetan=thetan, phin=phin, 
            delta=0., Ts=Ts, Tg=Tg, z=z, verbose=verbose,
            xalpha=xalpha, xc=xc, xB=xB)

    res = calc_Tb(thetak=thetak, phik=phik, thetan=thetan, phin=phin, 
            delta=delta, Ts=Ts, Tg=Tg, z=z, verbose=verbose,
            xalpha=xalpha, xc=xc, xB=xB)
    
    if verbose:
        print '\n\nTb(delta=0) = %f mK' % res0
        print 'Tb(delta) = %f mK' % res
        print 'Tb(delta) - Tb(delta=0) = %f mK' % ( res - res0 )
    return ( res - res0 )





def calc_G(thetak=np.pi/3., phik=np.pi/8., thetan=np.pi/3., phin=np.pi/4., 
            Ts=11.1, Tg=57.23508, z=20, verbose=False,
            xalpha=34.247221, xc=0.004176, xB=0.365092, x1s=1.):
    
    """Calculates the transfer function, which is the total derivative dTb/ddelta evaluated at delta=0, 
    from analytic derivative of eq 138 of Teja's draft v3. Result is in [K]. It takes x's (all unitless), temperatures in [K], and angles in [rad].
    B is along z!"""
    
    k_dot_n = np.cos(thetan)*np.cos(thetak) + np.sin(thetan)*np.sin(thetak)*np.cos(phin)*np.cos(phik) + np.sin(thetan)*np.sin(thetak)*np.sin(phin)*np.sin(phik)
      
    summ = 0.
    for i,m in enumerate( np.array([-2,-1,0,1,2]) ):
        summand = Y2( m,thetak,phik ) * np.conjugate( Y2( m,thetan,phin ) ) / ( 1. + xalpha + xc - 1j*m*xB )
        summ += summand.real
        
    first_term = 1 + k_dot_n**2
    second_term = 2. + 2.*k_dot_n**2 - 4.*np.pi/75.*summ

    res = x1s * ( 1 - Tg/Ts ) * np.sqrt( (1 + z)/10. ) * ( 26.4 * first_term - 0.128 * x1s * (Tg/Ts) * np.sqrt( (1 + z)/10. ) * second_term)
    
    return res/1000. #this is to make it to K from mK.



def calc_dGdB(thetak=np.pi/3., phik=np.pi/8., thetan=np.pi/3., phin=np.pi/4., 
            Ts=11.1, Tg=57.23508, z=20, verbose=False,
            xalpha=34.247221, xc=0.004176, xBcoeff=3.65092e18, x1s=1.):

    """Calculates the derivative of the transfer function wrt the magnitude of a homogeneous component
    of the magnetic field B0 *at present day* ( such that B(z)=B0*(1+z)^2 ), evaluated at B0=0, 
    from analytic derivative of eq 138 of Teja's draft v3. Result is in [K/Gauss].
    B is along z.  It takes x's (all unitless), temperatures in [K], and angles in [rad]."""

    
    k_dot_n = np.cos(thetan)*np.cos(thetak) + np.sin(thetan)*np.sin(thetak)*np.cos(phin)*np.cos(phik) + np.sin(thetan)*np.sin(thetak)*np.sin(phin)*np.sin(phik)
     
    summ = 0.
    for i,m in enumerate( np.array([-2,-1,0,1,2]) ):
        summand =  1j * m * xBcoeff * Y2( m,thetak,phik ) * np.conjugate( Y2(m,thetan,phin) ) / ( 1. + xalpha + xc )**2
        summ += summand.real

    #res = (0.128*4.*np.pi/75.) * x1s**2 * ( 1 - Tg/Ts ) * (Tg/Ts) * (1 + z)/10. * summ 
    res = (0.128*4.*np.pi/75.) * x1s**2 * ( 1 - Tg/Ts ) * (Tg/Ts) * (1 + z)**3/10. * summ 
    #!!!!!!!!! the deriv should be wrt B0 today, so there are factors of (1+z)**2 that might have been pushed to fisher...check this: thats why now (1+z)**3.
    
    return res/1000. #this is to make it to K from mK.
