# This file contains global variables related to paths, for use in fisher.py and plots_pmfs.py

import os.path

#MAIN_PATH = '/home/verag/Projects/Repositories/pmfs/' #imac110
#MAIN_PATH = '/Users/verag/Research/Repositories/pmfs/' #laptop

MAIN_PATH = os.path.dirname(os.path.realpath(__file__)) + '/' #sets the main path to the directory where this file is

    
RESULTS_PATH = MAIN_PATH+'results/'
INPUTS_PATH = MAIN_PATH + 'inputs/'
MATTER_POWER_PATH = MAIN_PATH + 'matter_power/'




