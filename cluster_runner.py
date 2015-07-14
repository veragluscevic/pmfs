#!/usr/bin/env python
import subprocess as sp
import os,sys,fnmatch
import argparse
import numpy as np
from globals import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode',default='B0')
parser.add_argument('--zmin', type=int, default=15)
parser.add_argument('--zmax', type=int, default=35)
parser.add_argument('--Jmode',default='default') # 'default', 'noheat', or 'hiheat'
parser.add_argument('--tobs', type=float, default=1.) # duration of the survey, in years.
parser.add_argument('--DeltaLmin', type=float, default=1) # min side of a square-shaped FFTT in km.
parser.add_argument('--DeltaLmax', type=float, default=10) # max side of a square-shaped FFTT in km.
parser.add_argument('--Omegasurvey', type=float, default=1.) # Omega_survey in sr.
parser.add_argument('--NDeltaL', type=int, default=300) # DeltaL sample points.
parser.add_argument('--ngroups', type=int, default=512)
args = parser.parse_args()

Omegasurvey = args.Omegasurvey

RUNNER_PATH = '/home/verag/Projects/Repositories/'
if args.Jmode == 'default':
    fastfile = INPUTS_PATH+ '21cmfast_teja_nov2014.txt'

NGROUPS = args.ngroups
DeltaLs = np.linspace(args.DeltaLmin,args.DeltaLmax,args.NDeltaL)
np.save(RESULTS_PATH + 'DeltaLs_{}_{}_tobs_{:.1f}.npy'.format(args.mode,args.Jmode,args.tobs), DeltaLs)

cmds = []
count = 0

for DeltaL in DeltaLs:
    tag = '{}_{}_tobs_{:.2f}_DeltaL_{:.2f}_Omega_{:.2f}'.format(args.mode,
                                                                args.Jmode,
                                                                args.tobs,
                                                                DeltaL,
                                                                Omegasurvey)
    resfile = tag + '.txt'

    cmd = '../runner.py --mode {} --tag {} --fastfile {} --zmin {} --zmax {} --tobs {:.1f} --DeltaL {} --Omegasurvey {} --resfile {}'.format(args.mode,tag,fastfile,args.zmin,args.zmax,args.tobs,DeltaL,Omegasurvey,resfile)
    cmds.append(cmd)
    count += 1

print  '\n There will be {} calls to runner.py.\n'.format(count)
if count < NGROUPS:
    NGROUPS = count

for i in range(NGROUPS):
    fout=open('runs_pmfs/{}_{}_{}.sh'.format(args.mode,args.Jmode, i+1), 'w')
    for cmd in cmds[i::NGROUPS]:
        fout.write('{}\n'.format(cmd))
    fout.close()

fout = open('runs_pmfs/go_{}_{}.sh'.format(args.mode,args.Jmode), 'w')
fout.write('#! /bin/bash\n')
fout.write('#$ -l h_rt=2:00:00\n')
fout.write('#$ -cwd\n')
fout.write('#$ -t 1-{}\n'.format(NGROUPS))
fout.write('#$ -V\n')
fout.write('bash {}_{}_$SGE_TASK_ID.sh\n'.format(args.mode,args.Jmode))
fout.close()
