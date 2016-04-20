'''
This module calculates the transverse shear power spectrum P_t
at z=10, 20, 30, respectively.
---Xiao Fang, 2015.01
'''

import numpy
import math
import ps2 as psnl

h=psnl.PowerSpectrum(0).h
c=psnl.PowerSpectrum(0).c
omega_M_0=psnl.PowerSpectrum(0).om_m0
omega_lambda_0=psnl.PowerSpectrum(0).om_v0

for i in range (1,4):
    outfile=open("power_spectrum_%d.dat"%(i),"w") #output file path
    z=10.*i
    dz=0.05
    d_z=psnl.PowerSpectrum(z).get_Dc(z) #get comoving distance for z
    print(d_z)
    lmax=3000
    l=2
    while l<=lmax:
        f=0
        z1=dz #the initial value for redshift variable z1
        #do integral over z given l
        while z1<=z:
            PSpec = psnl.PowerSpectrum(z1)
            d_z1=PSpec.get_Dc(z1)
            H_z1=PSpec.get_H(z1)
            k=l/d_z1  #unit: 1/Mpc
            p=PSpec.D2_NL(k)
            pnl=p[0]  #nonlinear \Delta^2
            p_delta_z1 = pnl*2*(math.pi)**2/k**3 #matter power spectrum
            p_phi_z1=(3.0/2*omega_M_0*(1+z1)*(h/c)**2)**2/(k**4)*p_delta_z1 #Newtonian potential power spectrum
            f=f+p_phi_z1/(d_z1)**2/H_z1
            z1=z1+dz
        f=f*dz*l**2*4/((d_z)**2) #this is the transverse power spectrum for the given l
        ff=l*(l+1)*f/(2*math.pi) #l(l+1)*P_t(l)/(2\pi)
        outfile.write("{0}\t{1}\t{2}\n".format(l,f,ff)) #output
        l=1.1*l #Since I will make a loglog plot, this is an efficient way to select l values
    outfile.close()
print("Congratulation!")
