#!/usr/bin/env python
# coding: utf-8


import numpy as np
import subprocess
import os

interv_type = 'smv1'
epochs = 20000
epochs_MMD = 2000
n = 5000

for tag_DA in ['baseline']:
    for myseed in range(100):
        subprocess.call(['bsub', '-W 03:50', '-n 4', "./sim_linearSCM_var_shift_exp9_scat_run.py --interv_type=%s --tag_DA=%s --epochs=%d --seed=%d" %(interv_type, tag_DA, 20000, myseed)])
        

# You don't have to run all the lamMatch choices!!!
lamMatches = [10.**(k) for k in (np.arange(10)-5)]
for tag_DA in ['DAmean', 'DAstd']:
    for myseed in range(100):
        for lam in lamMatches:
            subprocess.call(['bsub', '-W 03:50', '-n 4', "./sim_linearSCM_var_shift_exp9_scat_run.py --interv_type=%s --lamMatch=%f --tag_DA=%s --epochs=%d --seed=%d" %(interv_type, lam, tag_DA, epochs, myseed)]) 

for tag_DA in ['DAMMD']:
    for myseed in range(100):
        for lam in lamMatches:
            subprocess.call(['bsub', '-W 119:50', '-n 4', "./sim_linearSCM_var_shift_exp9_scat_run.py --interv_type=%s --lamMatch=%f --tag_DA=%s --epochs=%d --seed=%d" %(interv_type, lam, tag_DA, epochs_MMD, myseed)]) 


# You don't have to run all the lamCIP choices!!!
lamCIPs = [10.**(k) for k in (np.arange(10)-5)]
for tag_DA in ['DACIP']:
    for myseed in range(100):
        for lam in lamCIPs:
            subprocess.call(['bsub', '-W 03:50', '-n 4', "./sim_linearSCM_var_shift_exp9_scat_run.py --interv_type=%s --lamCIP=%f --tag_DA=%s --epochs=%d --seed=%d" %(interv_type, lam, tag_DA, epochs, myseed)]) 

for tag_DA in ['DACIPMMD']:        
    for myseed in range(100):
        for lam in lamCIPs:
            subprocess.call(['bsub', '-W 23:50', '-n 4', "./sim_linearSCM_var_shift_exp9_scat_run.py --interv_type=%s --lamCIP=%f --tag_DA=%s --epochs=%d --seed=%d" %(interv_type, lam, tag_DA, epochs_MMD, myseed)]) 
