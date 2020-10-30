#!/usr/bin/env python
# coding: utf-8

import numpy as np
import subprocess

interv_type = 'sv1'
epochs = 20000
epochs_MMD = 2000
n = 5000

for tag_DA in ['baseline']:
    subprocess.call(['bsub', '-W 03:50', '-n 4', "./sim_linearSCM_var_shift_exp8_box_run.py --interv_type=%s --tag_DA=%s --n=%d --epochs=%d" %(interv_type, tag_DA, n, epochs)])

lamMatches = [10.**(k) for k in (np.arange(10)-5)]
for tag_DA in ['DAmean', 'DAstd']:
    for lam in lamMatches:
        subprocess.call(['bsub', '-W 03:50', '-n 4', "./sim_linearSCM_var_shift_exp8_box_run.py --interv_type=%s --lamMatch=%f --tag_DA=%s --n=%d --epochs=%d" %(interv_type, lam, tag_DA, n, epochs)])

for tag_DA in ['DAMMD']:
    for lam in lamMatches:
        subprocess.call(['bsub', '-W 23:50', '-n 4', "./sim_linearSCM_var_shift_exp8_box_run.py --interv_type=%s --lamMatch=%f --tag_DA=%s --n=%d --epochs=%d" %(interv_type, lam, tag_DA, n, epochs_MMD)])