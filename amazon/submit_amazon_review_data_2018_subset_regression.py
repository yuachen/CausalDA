#!/usr/bin/env python
# coding: utf-8


import subprocess
import numpy as np

tag_DA = 'baseline'
lamL2s = [10.**(k) for k in (np.arange(10)-5)]
for target in range(16):
    for lamL2 in lamL2s:
        for myseed in range(10):
            subprocess.call(['bsub', '-W 03:50', '-n 8', '-R', "rusage[mem=4096]", "./amazon_review_data_2018_subset_regression.py --target=%d --tag_DA=%s --lamL2=%s --seed=%d" %(target, tag_DA, lamL2, myseed)]) 

tag_DA = 'DAmean'
lamMatches = [10.**(k) for k in (np.arange(10)-5)]
for target in range(16):
    for lam in lamMatches:
        for myseed in range(10):
            subprocess.call(['bsub', '-W 03:50', '-n 8', '-R', "rusage[mem=4096]", "./amazon_review_data_2018_subset_regression.py --target=%d --tag_DA=%s --lamL2=%s --lamMatch=%s --epochs=%d --seed=%d" %(target, tag_DA, 1.0, lam, 20000, myseed)]) 


# tag_DA = 'DAstd'
# for target in range(16):
#     for lam in lamMatches:
#         for myseed in range(10):
#             subprocess.call(['bsub', '-W 23:50', '-n 8', '-R', "rusage[mem=4096]", "./amazon_review_data_2018_subset_regression.py --target=%d --tag_DA=%s --lamMatch=%s --epochs=%d --seed=%d" %(target, tag_DA, lam, 20000, myseed)]) 
