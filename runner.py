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

from constants import *
from globals import *
import fisher as f

parser = argparse.ArgumentParser()
parser.add_argument('--respath', default='/data/verag/pmfs/') 
parser.add_argument('--folder', default=None)
parser.add_argument('--resfile', default='test.txt') # e.g.: B0_tyr_1_DeltaL_2.00_Omega_1.txt
#parser.add_argument('--fastfile', default=INPUTS_PATH+'21cmfast_teja_nov2014.txt')
parser.add_argument('--neval', type=int, default=100000) # number of integrand evaluations
parser.add_argument('--nevalPBi', type=int, default=10000) # number of integrand evaluations

parser.add_argument('--mode', default='B0') # 'B0' or 'xi' or 'SI' 

parser.add_argument('--zmin', type=int, default=10)
parser.add_argument('--zmax', type=int, default=25)
parser.add_argument('--kminmin', type=float, default=0.01)
parser.add_argument('--kmaxmax', type=float, default=1.)

parser.add_argument('--tyr', type=float, default=1.) # duration of the survey, in years.
parser.add_argument('--DeltaL', type=float, default=2.) # side of a square-shaped FFTT in km.
parser.add_argument('--Omegapatch', type=float, default=1.) # Omega_patch in degrees^2.
parser.add_argument('--Omegasurvey', type=float, default=1.) # Omega_survey in sr.


args = parser.parse_args()

resfile = RESULTS_PATH 
if args.folder is not None:
    resfile += args.folder + '/'
if not os.path.exists(resfile):
    os.mkdir(resfile)
resfile += args.resfile
fout = open(resfile, 'w')

# compute Fisher integral:
if args.mode=='B0' or args.mode=='xi':
    res = f.rand_integrator(neval=args.neval, 
                            t_yr=args.tyr,
                            DeltaL_km=args.DeltaL,
                            kminmin=args.kminmin,kmaxmax=args.kmaxmax,
                            zmax=args.zmax,zmin=args.zmin,
                            Omega_survey=args.Omegasurvey,
                            Omega_patch=args.Omegapatch,
                            thetan=np.pi/2.,phin=0.,
                            mode=args.mode)
if args.mode=='SI':
    res = f.calc_SNR(neval=args.neval, 
                     neval_PBi=args.nevalPBi,
                     t_yr=args.tyr,
                     DeltaL_km=args.DeltaL,
                     kminmin=args.kminmin,kmaxmax=args.kmaxmax,
                     zmax=args.zmax,zmin=args.zmin,
                     Omega_survey=args.Omegasurvey,
                     thetan=np.pi/2.,phin=0.)
#print result to output file:

if args.mode=='B0':
    header = 'B0[Gauss]'
if args.mode=='xi':
    header = 'xi'
if args.mode=='SI':
    header = 'sigma(sqrt[SI amplitude])'
fout.write('{}  Omega_survey[sr]  DeltaL^2[km^2]  tobs[yr]\n'.format(header))
fout.write('{}  {}  {} {}\n'.format(res, args.Omegasurvey, args.DeltaL**2, args.tyr ))
fout.close()

