#!/usr/bin/env python
import subprocess as sp
import os,sys,fnmatch
import argparse
import numpy as np
from globals import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode',default='B0')
parser.add_argument('--zmin', type=int, default=15)
parser.add_argument('--zmax', type=int, default=25)
parser.add_argument('--folder',default='midFSTAR') # midFSTAR, loFSTAR, or hiFSTAR
parser.add_argument('--tyr', type=float, default=1.) # duration of the survey, in years.
parser.add_argument('--DeltaLmin', type=float, default=1) # min side of a square-shaped FFTT in km.
parser.add_argument('--DeltaLmax', type=float, default=10) # max side of a square-shaped FFTT in km.
parser.add_argument('--Omegasurvey', type=float, default=1.) # Omega_survey in sr.
#parser.add_argument('--neval', type=int, default=100000) # number of integrand evaluations; should be 100 for mode=='SI' #this choice is just causing trouble
parser.add_argument('--nevalPBi', type=int, default=5000) # number of integrand evaluations

parser.add_argument('--NDeltaL', type=int, default=256) # DeltaL sample points.
parser.add_argument('--ngroups', type=int, default=512)
args = parser.parse_args()

if args.mode=='B0' or args.mode=='xi':
    neval = 100000
if args.mode=='SI':
    neval = 100

Omegasurvey = args.Omegasurvey

NGROUPS = args.ngroups
DeltaLs = np.linspace(args.DeltaLmin,args.DeltaLmax,args.NDeltaL)
filename = '/DeltaLs_{}_tyr_{:.2f}_Omega_{:.2f}.npy'.format(args.mode,args.tyr,args.Omegasurvey)
np.save(RESULTS_PATH + args.folder + filename, DeltaLs)

cmds = []
count = 0

for DeltaL in DeltaLs:
    resfile = '{}_tyr_{:.2f}_DeltaL_{:.2f}_Omega_{:.2f}.txt'.format(args.mode,
                                                                args.tyr,
                                                                DeltaL,
                                                                Omegasurvey)


    cmd = '../runner.py --mode {} --folder {} --zmin {} --zmax {} --tyr {} --DeltaL {} --Omegasurvey {} --resfile {} --neval {} --nevalPBi {}'.format(args.mode,args.folder,args.zmin,args.zmax,args.tyr,DeltaL,Omegasurvey,resfile,neval,args.nevalPBi)
    cmds.append(cmd)
    count += 1

print  '\n There will be {} calls to runner.py.\n'.format(count)
if count < NGROUPS:
    NGROUPS = count

for i in range(NGROUPS):
    fout=open('runs_pmfs/{}_{}_{}.sh'.format(args.mode,args.folder, i+1), 'w')
    for cmd in cmds[i::NGROUPS]:
        fout.write('{}\n'.format(cmd))
    fout.close()

fout = open('runs_pmfs/go_{}_{}.sh'.format(args.mode,args.folder), 'w')
fout.write('#! /bin/bash\n')
fout.write('#$ -l h_rt=1:00:00\n')
fout.write('#$ -cwd\n')
fout.write('#$ -t 1-{}\n'.format(NGROUPS))
fout.write('#$ -V\n')
fout.write('bash {}_{}_$SGE_TASK_ID.sh\n'.format(args.mode,args.folder))
fout.close()
