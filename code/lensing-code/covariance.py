"""
This module construct the covariance matrix of the transverse shear power spectra, and implement
the delensing procedure using CMB lensing.
---Xiao Fang, 2016.01
"""

import numpy as np


xs=np.loadtxt('xs_table.txt')
z=np.asarray(xs[:,0])
N=z.size
cov=np.zeros((N+1,N+1))

ppa=np.loadtxt('P_phi_ang_l_6.dat')
z_ppa=np.asarray(ppa[:,0])
d_ppa=np.asarray(ppa[:,1])
f_ppa=np.asarray(ppa[:,3])
d_cmb=d_ppa[-1]

l=6
Pt_lensing = open('Pt_lensing.txt','w')
for i in range(N):
	index = np.where(z_ppa<=z[i])[0]
	integral=f_ppa[index]
	sum_integral=np.sum(integral)
	dz1=d_ppa[index]
	d_i=dz1[-1]
	for N_i in range(N-i):
		j=N_i+i
		index2=np.where(z_ppa<z[j])[0]
		dz2=d_ppa[index2]
		d_j=dz2[-1]
		cov[i][j]=4./d_i/d_j * l**2 *sum_integral
		cov[j][i]=cov[i][j]
		
	Pt_lensing.write("{0}\n".format(l*(l+1)/(2*np.pi)*cov[i][i]))
	cov[N][i]=cov[i,N]=2* l**3 /d_i * np.sum(integral * (1./dz1 - 1./d_cmb))
	# cov[i][N]=0.+2j*l**3 /d_i * sum_integral2



cov[N][N]= l**4 * np.sum( f_ppa * (1./d_ppa - 1./d_cmb)**2 )
np.savetxt('cov.dat',cov)

covinv = np.linalg.inv(cov)
covinv_xx=covinv[0:N,0:N]
cov_xx_post = np.linalg.inv(covinv_xx)
Pt_delensing= l*(l+1)/(2*np.pi)*np.real(cov_xx_post.diagonal())
print('Pt_delensing:', Pt_delensing)
np.savetxt('Pt_delensing.txt',Pt_delensing)

print("Congratulation!")
