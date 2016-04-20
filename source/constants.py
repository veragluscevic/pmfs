# This file contains all the physical and cosmological constants used in fisher.py
# All constants are in CGS units.

import numpy as np

c = 2.99792458e10         #[cm/sec] speed of light
amu=1.66053892e-24        #[g] Atomic mass unit
ge=2                      #[-] Lande factor for e-
hbar=1.05457173e-27       #[erg sec = cm^2 g/sec] Reduced Planck's constant
Tstar=68.16871899714079/1000.   #[K] temperature equivalent to singlet-triplet splitting
Tcmb=2.72548              #[K] CMB temperature
A=2.86888e-15             #[1/sec] einstein coeff for singlet-triplet transition, pg 15 of micronotes4
lambdaLya=1.215668*1e-5   #[cm] Lya wavelength
gamma=49853694.3741053    #[Hz] fwhm of lya ??? check ???
e=4.80320425e-10          #[esu] electron charge
me=9.1093829e-28          #[g]  electron mass
muB=9.27400968e-21        #[erg/G] Bohr magneton
Obaryonh2 = 0.022161    #[-] Omega_b h^2 from Planck 2013 +WP +highL+BAO
Mpc_in_cm = 3.08567758e+24      #[cm] 1 Mpc in cm
yHe = 0.24771             #[-] Helium abundance
Newton_G = 6.67259e-8     #[cm^3/g/sec^2] Newton's gravitational constant
mH = 1.00794*amu          #[g] Mass of Hydrogen atom
kb = 1.380658e-16         #[erg/K] Boltzmann's constant
nu21 = 1420.40575177*10**6#[1/sec] frequency of 21cm transition
lambda21 = 21.1            #[cm] wavelength of 21cm 
h = 6.6260755e-27         #[erg sec = cm^2 g/sec] Planck's constant
f_to_T = h*nu21/kb        #[K]
Omatter = 0.315           # Omega_m, from Planck 2013
Olambda = 0.6825          # Omega lambda, was olda ??? check ???
H0 = 2.1972484e-18        # [1/sec] Hubble rate at z=0, from Planck 2013
Neff = 3.046              # Effective neutrino number, theoretical value
rho_critical = 3. * H0**2. / (8. * np.pi * Newton_G) #critical density today.
h0 = 0.72                 #Hubble rate normalized???
mp = 1.67262178e-24       #proton mass in [grams]
erg_in_ev = 6.24150934e11 #erg in eV
RH = 109737.316           #Rydberg constant in [1/cm]
t_universe = 13.6*1e9*365.*24.*3600. #age of the universe in [sec] 

##set up cosmological parameters from cosmolopy:
#from cosmolopy.parameters import WMAP7_ML
#cop = WMAP7_ML(flat=False)
#Omega_m = cop['omega_M_0']
#Omega_b = cop['omega_b_0']
#h0 = cop['h']
