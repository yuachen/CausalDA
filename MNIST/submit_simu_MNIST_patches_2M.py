#!/usr/bin/env python
# coding: utf-8

import numpy as np
import subprocess


#perturb = 'whitepatch2M'
#perturb = 'rotation2M'
perturb = 'rotation2Ma'
#perturb = 'translation2M'
tag_DA = 'baseline'
epochs = 100
for seed in range(10):
    for k in [2]:
        subprocess.call(['bsub', '-W 3:50', '-n 4', '-R', "rusage[ngpus_excl_p=1,mem=2048]",
                         "./simu_MNIST_patches_2M.py --perturb=%s --subset_prop=%.1f --seed=%d --epochs=%d --tag_DA=%s" %(perturb, k/10, seed, epochs, tag_DA)])

tag_DA = 'DIP'
lamMatches = [10.**(k) for k in (np.arange(10)-5)]
for lam in lamMatches:
    for seed in range(10):
        for k in [2]:
            subprocess.call(['bsub', '-W 3:50', '-n 4', '-R', "rusage[ngpus_excl_p=1,mem=2048]",
                "./simu_MNIST_patches_2M.py --perturb=%s --subset_prop=%.1f --seed=%d --lamMatch=%f --lamMatchMMD=%f --epochs=%d --tag_DA=%s" %(perturb, k/10, seed, lam, lam, epochs, tag_DA)])

