"""
This file calculates the matter power spectrum according to the halofit code by Smith 2003.
Part of this code is adapted from a C code written by Christopher M. Hirata

------ Xiao Fang, 2015.03
"""

import numpy as np

class PowerSpectrum(object):
    def __init__(self,z,
                 om_m0 = 0.32,
                 om_v0 = 0.68,
                 sig8  = 0.83,
                 gams  = 0.1905,
                 h     = 0.67,
                 om_b  = 0.022/0.67**2,
                 T_cmb = 2.728,
                 alphas= 0,
                 lndelta_zeta=-9.9814,
                 ns    = 0.963,
                 c     = 2997.92458,
                 omh2  = 0.32*0.67**2):
        self.om_m0 = om_m0
        self.om_v0 = om_v0
        self.sig8 = sig8
        self.gams  = gams
        self.z_ = z
        self.h = h
        self.om_b = om_b
        self.T_cmb= T_cmb
        self.alphas=alphas
        self.lndelta_zeta=lndelta_zeta
        self.ns = ns
        self.c  = c
        self.omh2 = omh2

    def DNeh_trf_mpc(self, rk):
        """
        k in 1/Mpc. Numbers to re-compute
        """
        k=np.asarray(rk)
        
        th27 = self.T_cmb / 2.7
        omh2 = self.om_m0*self.h**2
        obh2 = self.om_b*self.h**2
        #redshift at decoupling, equality, sound horizon
        b1 = 0.313 * pow(omh2, -0.419) * ( 1 + 0.607*pow(omh2, 0.674) )
        b2 = 0.238 * pow(omh2, 0.223)
        zd = 1291. * pow(omh2, 0.251) / ( 1 + 0.659*pow(omh2, 0.828) ) * ( 1 + b1*pow(obh2,b2) )
        zeq = 25000. * omh2 / pow(th27,4.)
        keq = 0.0746 * omh2 / th27 / th27   # in Mpc^-1
        rd = 31500.*obh2 / pow(th27,4.) / zd
        req = zd*rd/zeq
        s = 1.632993161855/keq/np.sqrt(req) * np.log((np.sqrt(1+rd)+np.sqrt(rd+req))/(1+np.sqrt(req)))
        #EH parameters
        a1 = pow(46.9*omh2, 0.670) * ( 1 + pow(32.1*omh2, -0.532) )
        a2 = pow(12.0*omh2, 0.424) * ( 1 + pow(45.0*omh2, -0.582) )
        b1 = 0.944 / ( 1 + pow(458*omh2, -0.708) )
        b2 = pow( 0.395*omh2, -0.0266)
        bm = obh2/omh2
        alphac = pow(a1, -bm) * pow(a2, -bm*bm*bm)
        betac = 1./( 1 + b1*(pow(1-bm, b2)-1) )
        #k-independent baryon parameters
        ksilk = 1.6 * pow(obh2, 0.52) * pow(omh2, 0.73) * ( 1 + pow(10.4*omh2, -0.95) )
        y = (1.+zeq)/(1.+zd)
        alphab = 2.07*keq*s*pow(1+rd,-0.75) * y * ( -6.*np.sqrt(1+y) + (2.+3.*y)*np.log((np.sqrt(1.+y)+1.)/(np.sqrt(1.+y)-1.)) )
        betab = 0.5 + bm + (3.-2.*bm)*np.sqrt(295.84*omh2*omh2+1.)
        #More parameters
        xnumer = 8.41*pow(omh2,0.435)/s
        kmpc0 = omh2/(th27*th27)
        #wavenumber in Mpc^-1 comoving, no h
        kmpc = k
        #CDM piece
        q = kmpc/kmpc0
        f = 1./(1. + pow(kmpc*s/5.4, 4.))
        C0 = 386./(1+69.9*pow(q,1.08))
        xsupp = q*q/np.log(np.e + 1.8*betac*q )
        T0_kab = 1./(1. + (C0+14.2/alphac)*xsupp)
        T0_k1b = 1./(1. + (C0+14.2)*xsupp)
        Tc = f*T0_k1b + (1-f)*T0_kab
        #Baryonic piece
        betanode__ks = xnumer/kmpc
        betab__ks = betab/kmpc/s
        stilde = s*pow(1.+betanode__ks*betanode__ks*betanode__ks, -0.33333333)
        Tb = 1./(1. + (C0+14.2)*q*q/np.log(np.e + 1.8*q ))/(1+kmpc*kmpc*s*s/27.04)+ alphab/(1+betab__ks*betab__ks*betab__ks)*np.exp(-pow(kmpc/ksilk,1.4))
        Tb =Tb*np.sin(kmpc*stilde)/(kmpc*stilde)
        T = bm*Tb + (1.-bm)*Tc
        return T


    def get_H(self,z):
        a= 1./(1.+z)
        H=self.h*np.sqrt(self.om_v0+self.om_m0/a**3)/self.c
        return H

    def get_Dc(self,z):
        dz=0.01
        Dc=0.
        z1=0.
        while z1<z:
            Dc=Dc+1./self.get_H(z1)
            z1=z1+dz
        Dc=Dc*dz
        print(Dc)
        return Dc


    def DNgrowth_derivative(self,lny,X,flag):
        dXdy=[0,0]
        if (flag):
            self.DNgrowth_derivativey=np.exp(lny)
            self.DNgrowth_derivativez=self.DNgrowth_derivativey-1.
            self.DNgrowth_derivativeH=self.c*self.get_H(self.DNgrowth_derivativez)/self.h
            self.DNgrowth_derivativeom=self.omh2*self.DNgrowth_derivativey**3/(self.DNgrowth_derivativeH*self.h)**2
        dXdy[0]=X[1]/self.DNgrowth_derivativeH+X[0]
        dXdy[1]=3*X[1]+1.5*self.DNgrowth_derivativeom*self.DNgrowth_derivativeH*X[0]
        return dXdy
        

    def DNgrowth_step(self,lny,Xin,dlny,flag):
        Xout=[0,0]
        lnyh=lny+0.5*dlny
        X=[0,0]
        k1=self.DNgrowth_derivative(lny,Xin,flag)
        for i in range (0,2):X[i]=Xin[i]+0.5*dlny*k1[i]
        k2=self.DNgrowth_derivative(lnyh,X,1)
        for i in range (0,2):X[i]=Xin[i]+0.5*dlny*k2[i]
        k3=self.DNgrowth_derivative(lnyh,X,0)
        for i in range (0,2):X[i]=Xin[i]+dlny*k3[i]
        k4=self.DNgrowth_derivative(lny+dlny,X,1)
        for i in range (0,2):Xout[i]=Xin[i]+dlny/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i])
        return Xout

    

    def DNget_growth_normhiz(self,z):
        dlny=-0.025
        N=240
        lny=np.log(1.+z)-N*dlny
        Xold=Xnew=[0,0]
        Xold[0]=1.
        Xold[1]=-self.get_H(np.exp(lny)-1)*self.c/self.h
        for i in range (0,N):
            if (i==0):flag=1
            else:flag=0
            Xnew=self.DNgrowth_step(lny,Xold,dlny,flag)
            lny=lny+dlny
            Xold=Xnew
        return (Xnew[0]/(1.+z))

    
    def D2_L(self, rk):
        #rk in "1/Mpc"
        k=np.asarray(rk)
        kpivot = 0.05
        z=self.z_

        "Poisson eqn coefficient"
        val_4pga2r = 1.5*self.omh2*(1.+z)/self.c/self.c
        T = self.DNeh_trf_mpc(rk)

        "Potential growth function"
        G = (1.+z)*self.DNget_growth_normhiz(z)

        "Consider each wavenumber"
        zeta2delta = -0.6 * G * T / val_4pga2r * k*k
        D2L=zeta2delta * zeta2delta * np.exp(2.*self.lndelta_zeta)*pow(k/kpivot,self.ns-1)
        return D2L
    
    def get_linsigma_gauss(self, z, R, recomp):
        k= 10**( -4.0 + 6.01*np.linspace(0,1,601) )
        if (recomp):
            self.D2=self.D2_L(k)
        var=0.
        kR=k*R
        T=np.exp(-kR**2/2)
        for ik in range (0,601):
            var=var+(T[ik])**2*self.D2[ik]
        var=var*0.01*np.log(10)
        return (np.sqrt(var))

    def get_Om(self,z):
        H=self.c*self.get_H(z)
        om=self.omh2*(1.+z)**3/(H*H)
        return (om)


    def get_GG0ratio(self,z):
        z1=0.
        gamma=0.55
        ratio=0.
        dz=0.01
        while z1<z:
            om_z1=self.get_Om(z1)
            ratio=ratio+om_z1**gamma/(1.+z1)
            z1=z1+dz
        ratio=np.exp(-ratio*dz)
        return ratio

    def winFT_gauss(self, rk, R):
        k=np.asarray(rk)
        win=np.exp(-(k*R)**2/2.)
        return win

    def winFT_tophat(self, rk, R):
        k=np.asarray(rk)
        kR=k*R
        win=3.*(np.sin(kR)-kR*np.cos(kR))/(kR)**3
        return win

    def D2_Lu(self,rk):
        k=np.asarray(rk)
        D2Lu=k**(3.+self.ns)*(self.DNeh_trf_mpc(rk))**2
        return D2Lu

    def get_linsigma_square(self, R):
        k = 10**( -4.0 + 7.5*np.linspace(0,1,10000) )
        dlnk=7.5/10000*np.log(10)
        func_k=(self.winFT_tophat(k,R))**2*self.D2_Lu(k)
        var=0.
        for ik in range (0,10000):
            var=var+func_k[ik]
        var=var*dlnk
        return var

    def get_amp_square(self):
        sigma8u2=self.get_linsigma_square(8/self.h)
        sigma8=self.sig8
        amp2=sigma8/sigma8u2
        return amp2

    def D2_L2(self, rk):
        #rk in "1/Mpc"
        k=np.asarray(rk)
        z=self.z_
        D2L=self.get_amp_square()*self.D2_Lu(rk)*(self.get_GG0ratio(z))**2
        return D2L


    def D2_NL(self, rk):
        z=self.z_
        k=np.asarray(rk)
        D2lin=D2=D2H=D2Q=0*k
        if (z>6):
            D2lin=self.D2_L2(rk)
            D2=[D2lin,0,0,D2lin]
            return D2
        D2lin=self.D2_L2(rk)

        R_NL=10.
        self.get_linsigma_gauss(z,R_NL,1)
        while (self.get_linsigma_gauss(z,R_NL,0)>1.):R_NL=R_NL*2
        while (self.get_linsigma_gauss(z,R_NL,0)<1.):R_NL=R_NL/2
        factor_NL=np.sqrt(2)
        while (factor_NL>1.000001):
            sigma = self.get_linsigma_gauss(z,R_NL,0)
            if (sigma>1):
                R_NL = R_NL*factor_NL
            else:
                R_NL = R_NL/factor_NL
            factor_NL=np.sqrt(factor_NL)
        n = -3. + 2.*(self.get_linsigma_gauss(z,0.9*R_NL,0)-self.get_linsigma_gauss(z,1.1*R_NL,0))/0.2
        C = -np.log(self.get_linsigma_gauss(z,np.exp(0.2)*R_NL,0)*self.get_linsigma_gauss(z,np.exp(-0.2)*R_NL,0))/0.02

        an = pow(10., 1.4861+n*( 1.8369+n*( 1.6762+n*( 0.7940+n*0.1670)))-0.6206*C)
        bn = pow(10., 0.9463+n*( 0.9466+n*0.3084)-0.9400*C)
        cn = pow(10.,-0.2807+n*( 0.6669+n*0.3214)-0.0793*C)
        alpha = 1.3884+n*( 0.3700-n*0.1452)
        beta  = 0.8291+n*( 0.9854+n*0.3401)
        gamma = 0.8649 + 0.2989*n + 0.1631*C
        mu = pow(10., -3.5442 + 0.1908*n)
        nu = pow(10.,  0.9589 + 1.2857*n)

        Om = self.get_Om(z)
        f1 = pow(Om, -0.0307)
        f2 = pow(Om, -0.0585)
        f3 = pow(Om,  0.0743)
        y=k*R_NL
        D2Q=D2lin*pow(1+D2lin,beta)/(1+alpha*D2lin)*np.exp(-y/4*(1+y/2))
        D2H=an*pow(y,3.*f1)/(1.+bn*pow(y,f2)+pow(cn*f3*y,3.-gamma))/(1.+(mu+nu/y)/y)
        D2=D2Q+D2H
        return D2,D2Q,D2H,D2lin


        

if __name__ == '__main__':
    import pylab

    z = 20
    
    PSpec = PowerSpectrum(z=z)

    N = 1000
    rk = 10**( -5.0 + 12.0*np.linspace(0,1,N) )
    h=PSpec.h
    #pdiff=PSpec.D2_L2(rk)-PSpec.D2_L(rk)



    plin=PSpec.D2_L2(rk)
    pnl,pq,ph,plin = PSpec.D2_NL(rk)

    
    fig=pylab.figure()
    pylab.loglog(rk,pnl,'-k',label='nonlinear')
    # pylab.loglog(rk,pq,':k',label='quasi-linear')
    # pylab.loglog(rk,ph,':r',label='halo')
    pylab.loglog(rk,plin,'--r',label='linear')

    
    pylab.ylim(10**-1.5,3E3)
    pylab.xlim(10**-1.5,1E2)
    #pylab.loglog(rk,pdiff,'--r',label='diff')    
    pylab.xlabel(r'$k$')
    pylab.ylabel(r'$\Delta^2(k)$')
    pylab.title('z=%.2f Power Spectrum' % z)
    
    pylab.legend(loc=0)

    pylab.show()
    np.savetxt('matterpower.txt', np.c_[rk,pnl*2*(np.pi)**2/rk**3])
    #fig.savefig('pshalo_test_2.eps')
