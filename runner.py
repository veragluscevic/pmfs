#!/usr/bin/env python
import argparse
import matplotlib as plt
import pylab as pl
import numpy as np
from scipy.interpolate import UnivariateSpline as interpolate
import os,os.path,shutil
import cosmolopy.distance as cd
 
import reion_functions as rf
reload(rf)
import cosmo_functions as cf
reload(cf)
import geometric_functions as gf
reload(gf)
import pmfs_transfer as pt
reload(pt)
import pmfs_fisher as pf
reload(pf)

from constants import *
from globals import *
import fisher as f

parser = argparse.ArgumentParser()
parser.add_argument('--respath', default='/data/verag/pmfs/') 
parser.add_argument('--mode', default='B0') # 'B0' or 'zeta'
parser.add_argument('--forcegrid', action='store_true')
parser.add_argument('--plotgrid', action='store_true')
parser.add_argument('--tag', default=None)
parser.add_argument('--fastfile', default='21cmfast_teja_nov2014.txt')
parser.add_argument('--zmin', type=int, default=10)
parser.add_argument('--zmax', type=int, default=35)
parser.add_argument('--tobs', type=float, default=1.) # duration of the survey, in years.
parser.add_argument('--DeltaL', type=float, default=2.) # side of a square-shaped FFTT in km.
parser.add_argument('--Omegapatch', type=float, default=1.) # Omega_patch in degrees^2.
parser.add_argument('--Omegasurvey', type=float, default=1.) # Omega_survey in sr.
parser.add_argument('--resfile', default='B0_tobs_1_DeltaL_2.00_Omega_1.txt')
args = parser.parse_args()

mode = args.mode
force_grid = args.forcegrid
tag = args.tag

grid_path = RESULTS_PATH 
if tag is not None:
    grid_path += tag
else:
    grid_path += mode
grid_path += '/'

#noise-calculation parameters:
#Omega_survey = args.Omegasurvey
#tobs = args.tobs * 365. * 86400 #observation time in sec, for entire survey.
#DeltaL = args.DeltaL * 100000. #length of a square-shaped antenna in cm.
#Omega_patch = args.Omegapatch * (np.pi/180.)**2 # area coverage of the small patch in survey in sr.
##tobs_patch = tobs * Omega_patch / Omega_survey
#Ae = None #1000**2 #3500**2 #effective area of a single dish in cm.
#Lmax = None #1200000 #maximum baseline in cm.
#Lmin = None #1000 #minimum baseline in cm. 
#N_ant = None #1024 #number of antennas.
#kminmin = 0.0001 #minimum k to be analyzed in [1/Mpc comoving]
#kmaxmax = 10 #maximum k to be analyzed in [1/Mpc comoving]
#set up z-range of interest:
#z_lims = (args.zmin,args.zmax)# (15,29.9) 
#Nzs = int(z_lims[1]-z_lims[0])
#Nks = 100
#Nthetak = 51
#Nphik = 52

#file_21cmfast = np.loadtxt(args.fastfile)
#Tks_21cmfast = file_21cmfast[:,2][::-1]
#Tgs_21cmfast = file_21cmfast[:,5][::-1]
#Tss_21cmfast = file_21cmfast[:,4][::-1]
#Jlyas_21cmfast = file_21cmfast[:,6][::-1]
#zs_21cmfast = file_21cmfast[:,0][::-1]
#val_Tk = interpolate(zs_21cmfast, Tks_21cmfast, s=0)
#val_Tg = interpolate(zs_21cmfast, Tgs_21cmfast, s=0)
#val_Ts = interpolate(zs_21cmfast, Tss_21cmfast, s=0)
#val_Jlya = interpolate(zs_21cmfast, Jlyas_21cmfast, s=0)

##choose sky temperature function:
#val_Tsys = pf.Tsys_Mao2008 #pf.Tsys_zero #pf.Tsys_simple
##set up the choice of uv coverage:
#val_nk = pf.FFTT_nk #pf.one_over_u2_nk
##set evolution of ionized fraction:
#val_x1s = rf.ones_x1s
##write Fisher grid that can then be read from files, and integrated:
#if force_grid or (not(os.path.exists(grid_path))) or (not(os.path.exists(grid_path + 'fisher_grid.npy')))or (not(os.path.exists(grid_path + 'z_grid.npy')))or (not(os.path.exists(grid_path + 'phik_grid.npy')))or (not(os.path.exists(grid_path + 'thetak_grid.npy')))or (not(os.path.exists(grid_path + 'k_grid.npy'))):

if(os.path.exists(grid_path)):
    shutil.rmtree(grid_path)
os.makedirs(grid_path)
    
if mode == 'B0':
    res = f.rand_integrator(neval=100000, 
                          DeltaL_km=args.DeltaL,
                          kminmin=0.01,kmaxmax=1.,
                          zmax=35,zmin=15,
                          Omega_survey=1.,
                          Omega_patch=1.,
                          thetan=np.pi/2.,phin=0.)
        #pf.write_Fisher_grid(grid_path, 
        #                     val_nk=val_nk, val_Ts=val_Ts, 
        #                     val_Tk=val_Tk, val_x1s=val_x1s,
        #                     z_lims=z_lims, 
        #                     Nzs=Nzs, Nks=Nks, Nthetak=Nthetak, Nphik=Nphik, 
        #                     phin=0., thetan=np.pi/2.,
        #                     val_Tsys=val_Tsys, tobs=tobs, 
        #                     Ae=Ae, Lmax=Lmax, Lmin=Lmin, 
        #                     N_ant=N_ant, Omega_patch=Omega_patch,
        #                     kminmin=kminmin, kmaxmax=kmaxmax,
        #                     DeltaL=DeltaL)
        
    #if mode == 'zeta':
        #pf.write_Fisher_grid_zeta(grid_path, 
        #                          val_nk=val_nk, val_Ts=val_Ts, 
        #                          val_Tk=val_Tk, val_x1s=val_x1s,
        #                          z_lims=z_lims, 
        #                          Nzs=Nzs, Nks=Nks, Nthetak=Nthetak, Nphik=Nphik, 
        #                          phin=0., thetan=np.pi/2.,
        #                          val_Tsys=val_Tsys, tobs=tobs, 
        #                          Ae=Ae, Lmax=Lmax, Lmin=Lmin, 
        #                          N_ant=N_ant, Omega_patch=Omega_patch,
        #                          kminmin=kminmin, kmaxmax=kmaxmax)

##now read the fisher grid, and integrate it in all dimensions:
#fisher_grid = np.load(grid_path + 'fisher_grid.npy')
#zs = np.load(grid_path + 'z_grid.npy')
#phiks = np.load(grid_path + 'phik_grid.npy')
#thetaks = np.load(grid_path + 'thetak_grid.npy')
#ks = np.load(grid_path + 'k_grid.npy')
#result = pf.trapznd(fisher_grid,zs,phiks,thetaks,ks)
#alpha_survey = (Omega_survey)**0.5 
#result_all_survey = result / Omega_patch * np.pi * (alpha_survey + np.cos(alpha_survey)*np.sin(alpha_survey))
##the final result, in Gauss:
#res = 1./result_all_survey**0.5
#print res 

#print to output file:
fout = open(grid_path + args.resfile, 'w')
fout.write('B0[Gauss]  Omega_survey[sr]  DeltaL^2[km^2]  tobs[yr]\n')
fout.write('{}  {}  {} {}\n'.format(res, args.Omegasurvey, args.DeltaL**2, args.tobs ))
fout.close()

