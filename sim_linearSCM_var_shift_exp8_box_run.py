#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import sys
import argparse

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import plot_basic
# local packages
import semiclass
import semitorchclass
import semitorchstocclass
import util
import simudata

# check gpu avail
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--interv_type", type=str, default="sv1", help="type of intervention")
parser.add_argument("--lamMatch", type=float, default=1., help="DIP matching penalty")
parser.add_argument("--epochs", type=int, default=4000, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--tag_DA", type=str, default="baseline", help="choose whether to run baseline methods or DA methods")
parser.add_argument("--n", type=int, default=5000, help="sample size")
myargs = parser.parse_args()
print(myargs)

lamL2 = 0.
lamL1 = 0.

if myargs.tag_DA == "baseline":
    methods = [
               semiclass.Tar(lamL2=lamL2),
               semiclass.SrcPool(lamL2=lamL2),
               semitorchclass.Tar(lamL2=lamL2, lamL1=lamL1, lr=myargs.lr, epochs=myargs.epochs),
               semitorchclass.SrcPool(lamL2=lamL2, lamL1=lamL1, lr=myargs.lr, epochs=myargs.epochs),
              ]
elif myargs.tag_DA == "DAmean":
    methods = [
               semiclass.DIP(lamMatch=myargs.lamMatch, lamL2=lamL2, sourceInd=0), 
               semiclass.DIPOracle(lamMatch=myargs.lamMatch, lamL2=lamL2, sourceInd=0), 
              ]
elif myargs.tag_DA == "DAstd":
    methods = [
               semitorchclass.DIP(lamMatch=myargs.lamMatch, lamL2=lamL2, lamL1=lamL1, sourceInd=0, lr=myargs.lr, epochs=myargs.epochs, wayMatch='std'),
               semitorchclass.DIP(lamMatch=myargs.lamMatch, lamL2=lamL2, lamL1=lamL1, sourceInd=0, lr=myargs.lr, epochs=myargs.epochs, wayMatch='mean+std+25p'),
              ]
else: # "DAMMD"
    methods = [
               semitorchstocclass.DIP(lamMatch=myargs.lamMatch, lamL2=lamL2, lamL1=lamL1, sourceInd = 0, lr=myargs.lr,
                  epochs=myargs.epochs, wayMatch='mmd', sigma_list=[1.]),
              ]


names = [str(m) for m in methods]
print(names)
names_short = [str(m).split('_')[0] for m in methods]
print(names_short)

seed1 = int(123456 + np.exp(1) * 1000)

def simple_run_sem(sem_nums, ds, methods, i2n_ratios=[1.], n=2000, repeats=10):
    res_all = {}
    for i, sem_num in enumerate(sem_nums): 
        for j, inter2noise_ratio_local in enumerate(i2n_ratios):
            print("Number of envs M=%d, inter2noise_ratio=%.1f" %(2, inter2noise_ratio_local), flush=True)
            params = {'M': 2, 'inter2noise_ratio': inter2noise_ratio_local, 'd': ds[i]}

            sem1 = simudata.pick_sem(sem_num, 
                                     params=params, 
                                     seed=seed1)


            # run methods on data generated from sem
            results_src_all, results_tar_all = util.run_all_methods(sem1, methods, n=n, repeats=repeats)
            res_all[(i, j)] = results_src_all, results_tar_all  
    return res_all

repeats = 10 
res_all = simple_run_sem(sem_nums=['r0%sd3x1' %myargs.interv_type,
                                   'r0%sd?x1' %myargs.interv_type,
                                   'r0%sd?x1' %myargs.interv_type],
                         ds=[3, 10, 20],
                         i2n_ratios=[1.],
                         methods=methods,
                         n=myargs.n,
                         repeats=repeats)

np.save("simu_results/sim_exp8_box_r0%sd31020_%s_lamMatch%s_n%d_epochs%d_repeats%d.npy" %(myargs.interv_type,
         myargs.tag_DA, myargs.lamMatch, myargs.n, myargs.epochs, repeats), res_all)