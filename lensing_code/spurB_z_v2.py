"""
This module calculates the lensing contamination power spectrum and makes the plot
---Xiao Fang, 2015.12
"""

import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib import rc


def get_coeff_Bsat():
	xs=np.loadtxt('xs_table.txt')
	gl=np.loadtxt('global.txt')

	Pt_lensing=np.asarray(np.loadtxt('Pt_lensing.txt'))
	Pt_delensing=np.asarray(np.loadtxt('Pt_delensing.txt'))

	z=xs[:,0]
	xalpha2=xs[:,1]
	xc2=xs[:,2]
	xBcoeff=xs[:,3]

	x1s=gl[:,1]
	Tspin=gl[:,4]
	Tg=gl[:,5]
	x1s=x1s[::-1]
	Tspin=Tspin[::-1]
	Tg=Tg[::-1]

	ratio_TgammaTs=Tg/Tspin
	ns=-2.15

	C=0.128*ratio_TgammaTs*x1s*np.sqrt((1.+z)/10.)
	print(C/20.0/(1.+xalpha2+xc2),'C')
	lamda=13.2-C-C/20.0/(1.+xalpha2+xc2)
	q=3.*(13.2-C)-C/60.0/(1.+xalpha2+xc2)
	#print(-4*lamda/q)
	A=np.absolute(10*(1.+xalpha2+xc2)**2*(lamda+(ns)/4.*q*4./3./16.*11.)/C / xBcoeff /(1.+z)**2)/1.478*1.577
	Bsat =(1.+xalpha2+xc2)/xBcoeff/(1.+z)**2

	return z, A, Bsat, Pt_lensing, Pt_delensing

def spurB(z, A, Pt_lensing, Pt_delensing, l=6.):
	z=np.asarray(z)
	A=np.asarray(A)
	Delta_trans=np.sqrt(Pt_lensing)
	Delta_delensing=np.sqrt(Pt_delensing)

	Delta_B=A*Delta_trans
	Delta_B_de=A*Delta_delensing
	return Delta_B, Delta_trans, Delta_B_de

def main():
	recalculate=1
	if recalculate==1:
		z, A, Bsat, Pt_lensing, Pt_delensing= get_coeff_Bsat()
		Delta_B, Delta_trans, Delta_B_de = spurB(z, A, Pt_lensing, Pt_delensing)
	else:
		spurfile=np.loadtxt('delensing_l6.dat')
		z = spurfile[:,0]
		A = spurfile[:,1]
		Bsat = spurfile[:,2]
		Delta_B = spurfile[:,3]
		Delta_trans=spurfile[:,4]
		Delta_B_de=spurfile[:,5]


	matplotlib.use('Agg')

	fig=plt.figure()

	ax=plt.subplot(111)
	rc('font',**{'family':'serif','serif':['Times','Palatino']})
	rc('text', usetex=True)

	ax.tick_params(axis='both', which='major', length=12, width=2)
	ax.tick_params(axis='both', which='minor', length=8, width=1)
	ax.tick_params(axis='both', labelsize=22)



	ax.set_yscale('log')
	xlabel = ax.set_xlabel(r'$z$',fontsize=22)
	ylabel = ax.set_ylabel(r'comoving lensing $B$ [Gauss]',fontsize=22)
	ax.set_xlim(xmin=15,xmax=28)

	ax.plot(z, Delta_B, color='r',linestyle='solid',linewidth=3,label='lensing')

	ax.plot(z, Delta_B_de, color='b',linestyle='--', linewidth=3,label='after de-lensing')
	ymax=1e-19
	ceiling = np.ones(len(z)) * ymax
	plt.semilogy(z, Bsat,'--', color='gray',label='saturation ceiling')
	plt.fill_between(z, Bsat, ceiling, alpha=0.14, color='gray')


	ax.grid(which='both')
	plt.legend(fontsize=22)

	plt.gcf().subplots_adjust(left=0.15,bottom=0.15)
	# plt.gca().tight_layout()
	plt.show()
	fig.savefig('delensingB.eps',bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')
	fig.savefig('delensingB.png',bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')
	fig.savefig('delensingB.pdf',bbox_extra_artists=[xlabel, ylabel], 
                bbox_inches='tight')



	outfile=open("delensing_l6.dat","w")
	for i in range(z.size):
		outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(z[i], A[i], Bsat[i], Delta_B[i], Delta_trans[i], Delta_B_de[i]))
	outfile.close()
	print("Congratulation!")

if __name__ == '__main__':
	main()