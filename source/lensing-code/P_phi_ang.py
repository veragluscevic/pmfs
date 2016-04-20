"""
This module calculates the lensing potential power spectrum at given l,
and then does the radial integral up to a list of redshifts.

---Xiao Fang, 2016.01
"""

import numpy as np
import ps2 as ps

h=ps.PowerSpectrum(0).h
c=ps.PowerSpectrum(0).c
omega_M_0=ps.PowerSpectrum(0).om_m0
omega_lambda_0=ps.PowerSpectrum(0).om_v0

l=6

outfile=open("P_phi_ang_l_%s.dat"%(str(l)),"w")

z=1091. #CMB redshift

dz=0.05
z1=dz #the initial value for redshift variable z1
while z1<=z:
	PSpec = ps.PowerSpectrum(z1)
	d_z1=PSpec.get_Dc(z1)
	H_z1=PSpec.get_H(z1)
	k=l/d_z1  #unit: 1/Mpc
	p=PSpec.D2_NL(k)
	pnl=p[0]  #nonlinear \Delta^2
	print(pnl)
	p_delta_z1 = pnl*2*(np.pi)**2/k**3 #matter power spectrum
	p_phi_z1=(3.0/2*omega_M_0*(1+z1)*(h/c)**2)**2/(k**4)*p_delta_z1 #Newtonian potential power spectrum
	f=p_phi_z1/(d_z1)**2/H_z1 *dz
	outfile.write("{0}\t{1}\t{2}\t{3}\n".format(z1, d_z1, H_z1, f))
	z1=z1+dz

outfile.close()
print("Congratulation!")
