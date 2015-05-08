
import numpy as np

import cosmo_functions as cf
reload(cf)
import reionization functions as rf
reload(rf)
import pmfs_transfer as pt
reload(pt)
import pmfs_fisher as pf

from constants import *
from globals import *
from geometric_functions import *


def saturation():
    """
    """
    pf.calc_G(thetak=np.pi/2., phik=0., thetan=np.pi/2., phin=np.pi/4., 
            Ts=11.1, Tg=57.23508, z=20, verbose=False,
            xalpha=34.247221, xc=0.004176, xB=0.365092, x1s=1.)
