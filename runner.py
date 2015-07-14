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
parser.add_argument('--tag', default=None)
parser.add_argument('--resfile', default='test.txt') # e.g.: B0_tobs_1_DeltaL_2.00_Omega_1.txt
parser.add_argument('--fastfile', default='21cmfast_teja_nov2014.txt')
parser.add_argument('--neval', type=int, default=100000) # number of integrand evaluations

parser.add_argument('--mode', default='B0') # 'B0' or 'zeta'

parser.add_argument('--zmin', type=int, default=15)
parser.add_argument('--zmax', type=int, default=35)
parser.add_argument('--kminmin', type=float, default=0.01)
parser.add_argument('--kmaxmax', type=float, default=1.)

parser.add_argument('--tobs', type=float, default=1.) # duration of the survey, in years.
parser.add_argument('--DeltaL', type=float, default=2.) # side of a square-shaped FFTT in km.
parser.add_argument('--Omegapatch', type=float, default=1.) # Omega_patch in degrees^2.
parser.add_argument('--Omegasurvey', type=float, default=1.) # Omega_survey in sr.


args = parser.parse_args()

# create results directory:
grid_path = RESULTS_PATH 
if args.tag is not None:
    grid_path += args.tag
else:
    grid_path += args.mode
grid_path += '/'
if(os.path.exists(grid_path)):
    shutil.rmtree(grid_path)
os.makedirs(grid_path)

# compute Fisher integral:
res = f.rand_integrator(neval=args.neval, 
                          DeltaL_km=args.DeltaL,
                          kminmin=args.kminmin,kmaxmax=args.kmaxmax,
                          zmax=args.zmax,zmin=args.zmin,
                          Omega_survey=args.Omegasurvey,
                          Omega_patch=args.Omegapatch,
                          thetan=np.pi/2.,phin=0.,
                          mode=args.mode)
#print result to output file:
fout = open(grid_path + args.resfile, 'w')
if args.mode=='B0':
    header = 'B0[Gauss]'
if args.mode=='zeta':
    header = 'zeta'
fout.write('{}  Omega_survey[sr]  DeltaL^2[km^2]  tobs[yr]\n'.format(header))
fout.write('{}  {}  {} {}\n'.format(res, args.Omegasurvey, args.DeltaL**2, args.tobs ))
fout.close()

