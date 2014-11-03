#This module contains geometric functions, like the spherical harmonics etc.

import numpy as np
from constants import *
from globals import *
 
def Y2(m,theta,phi):
    """These are exactly Ylm's as defined on wikipaedia; seems to match Teja's."""
    if m==-2:
        return 0.25*(15./(2.*np.pi))**0.5 * np.sin(theta)**2. / np.exp(2.*1j*phi)
    elif m==2:
        return 0.25*(15./(2.*np.pi))**0.5 * np.sin(theta)**2. * np.exp(2.*1j*phi)
    elif m==-1:
        return 0.5*(15./(2.*np.pi))**0.5 * np.sin(theta)*np.cos(theta) / np.exp(1j*phi)
    elif m==1:
        return (-0.5)*(15./(2.*np.pi))**0.5 * np.sin(theta)*np.cos(theta) * np.exp(1j*phi)
    elif m==0:
        return 0.25*(5./np.pi)**0.5 * (3.*np.cos(theta)**2. - 1)
    
    
