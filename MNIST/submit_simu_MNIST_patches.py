#!/usr/bin/env python
# coding: utf-8

import numpy as np
import subprocess


# perturb = 'whitepatch'
perturb = 'rotation'
epochs=100
M = 5
lamMatches = [10.**(k) for k in (np.arange(10)-5)]
lamCIPs = [10.**(k) for k in (np.arange(10)-5)]

tag_DA = 'baseline'
for seed in range(10):
    for k in [2]:
        subprocess.call(['bsub', '-W 3:50', '-n 4', '-R', "rusage[ngpus_excl_p=1,mem=2048]",
                         "./simu_MNIST_patches.py --perturb=%s --M=%d --subset_prop=%.1f --seed=%d --epochs=%d --tag_DA=%s" %(perturb, M, k/10, seed, epochs, tag_DA)])

for tag_DA in ['DACIPmean']:
    for lam in lamCIPs:
        for seed in range(10):
            for k in [2]:
                subprocess.call(['bsub', '-W 3:50', '-n 8', '-R', "rusage[ngpus_excl_p=1,mem=2048]",
                    "./simu_MNIST_patches.py --perturb=%s --M=%d --subset_prop=%.1f --seed=%d --lamMatch=%f --lamCIP=%f --lamMatchMMD=%f --lamCIPMMD=%f --epochs=%d --tag_DA=%s" %(perturb, M, k/10, seed, 1., lam, 1., lam, epochs, tag_DA)])

for tag_DA in ['DACIPMMD']:
    for lam in lamCIPs:
        for seed in range(10):
            for k in [2]:
                subprocess.call(['bsub', '-W 23:50', '-n 8', '-R', "rusage[ngpus_excl_p=1,mem=2048]",
                    "./simu_MNIST_patches.py --perturb=%s --M=%d --subset_prop=%.1f --seed=%d --lamMatch=%f --lamCIP=%f --lamMatchMMD=%f --lamCIPMMD=%f --epochs=%d --tag_DA=%s" %(perturb, M, k/10, seed, 1., lam, 1., lam, epochs, tag_DA)])

# pick lamCIP after looking at CIP source results
lamCIP = 1.
for tag_DA in ['DAmean']:
    for lam in lamMatches:
        for seed in range(10):
            for k in [2]:
                subprocess.call(['bsub', '-W 23:50', '-n 4', '-R', "rusage[ngpus_excl_p=1,mem=2048]",
                "./simu_MNIST_patches.py --perturb=%s --M=%d --subset_prop=%.1f --seed=%d --lamMatch=%f --lamCIP=%f --lamMatchMMD=%f --lamCIPMMD=%f --epochs=%d --tag_DA=%s" %(perturb, M, k/10, seed, lam, lamCIP, lam, lamCIP, epochs, tag_DA)])


lamCIP = 1.
for tag_DA in ['DAMMD']:
    for lam in lamMatches:
        for seed in range(10):
            for k in [2]:
                subprocess.call(['bsub', '-W 23:50', '-n 8', '-R', "rusage[ngpus_excl_p=1,mem=2048]",
                "./simu_MNIST_patches.py --perturb=%s --M=%d --subset_prop=%.1f --seed=%d --lamMatch=%f --lamCIP=%f --lamMatchMMD=%f --lamCIPMMD=%f --epochs=%d --tag_DA=%s" %(perturb, M, k/10, seed, lam, lamCIP, lam, lamCIP, epochs, tag_DA)])
