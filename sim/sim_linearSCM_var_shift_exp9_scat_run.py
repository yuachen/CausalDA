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

# local packages
import sys
sys.path.append('../')
import semiclass
import semitorchclass
import semitorchstocclass
import util
import simudata

# check gpu avail
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

parser = argparse.ArgumentParser()
parser.add_argument("--interv_type", type=str, default="smv1", help="type of intervention")
parser.add_argument("--lamMatch", type=float, default=1., help="DIP matching penalty")
parser.add_argument("--lamCIP", type=float, default=0.1, help="CIP penalty")
parser.add_argument("--epochs", type=int, default=4000, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--tag_DA", type=str, default="baseline", help="choose whether to run baseline methods or DA methods")
parser.add_argument("--n", type=int, default=5000, help="sample size")
parser.add_argument("--seed", type=int, default=0, help="seed of experiment")
myargs = parser.parse_args()
print(myargs)


lamL2 = 0.
lamL1 = 0.

if myargs.tag_DA == 'baseline':
    methods = [
               semiclass.Tar(lamL2=lamL2),
               semiclass.SrcPool(lamL2=lamL2),
               semitorchclass.Tar(lamL2=lamL2, lamL1=lamL1, lr=myargs.lr, epochs=myargs.epochs),
               semitorchclass.SrcPool(lamL2=lamL2, lamL1=lamL1, lr=myargs.lr, epochs=myargs.epochs),
              ]
elif myargs.tag_DA == 'DAmean':
    methods = [
               semiclass.DIP(lamMatch=myargs.lamMatch, lamL2=lamL2, sourceInd=0),
               semiclass.DIPOracle(lamMatch=myargs.lamMatch, lamL2=lamL2, sourceInd=0),
               semiclass.DIPweigh(lamMatch=myargs.lamMatch, lamL2=lamL2),
               semiclass.CIP(lamCIP=myargs.lamCIP, lamL2=lamL2),
               semiclass.CIRMweigh(lamCIP=myargs.lamCIP, lamMatch=myargs.lamMatch, lamL2=lamL2),
              ]
elif myargs.tag_DA == 'DAstd':
    methods = [
               semitorchclass.DIP(lamMatch=myargs.lamMatch, lamL2=lamL2, lamL1=lamL1, sourceInd=0, lr=myargs.lr,
                              epochs=myargs.epochs, wayMatch='mean+std+25p'),
               semitorchclass.DIPweigh(lamMatch=myargs.lamMatch, lamL2=lamL2, lamL1=lamL1, lr=myargs.lr,
                                   epochs=myargs.epochs, wayMatch='mean+std+25p'),
               semitorchclass.CIP(lamCIP=myargs.lamCIP, lamL2=lamL2, lamL1=lamL1, lr=myargs.lr,
                               epochs=myargs.epochs, wayMatch='mean+std+25p'),
               semitorchclass.CIRMweigh(lamMatch=myargs.lamMatch, lamL2=lamL2, lamL1=lamL1, lr=myargs.lr,
                                    epochs=myargs.epochs, wayMatch='mean+std+25p'),
              ]
elif myargs.tag_DA == 'DAMMD':
    methods = [
               semitorchstocclass.DIP(lamMatch=myargs.lamMatch, lamL2=lamL2, lamL1=lamL1, sourceInd = 0, lr=myargs.lr,
                              epochs=myargs.epochs, wayMatch='mmd', sigma_list=[1.]),
               semitorchstocclass.DIPweigh(lamMatch=myargs.lamMatch, lamL2=lamL2, lamL1=lamL1, lr=myargs.lr,
                                       epochs=myargs.epochs, wayMatch='mmd', sigma_list=[1.]),
               semitorchstocclass.CIP(lamCIP=myargs.lamCIP, lamL2=lamL2, lamL1=lamL1, lr=myargs.lr,
                                  epochs=myargs.epochs, wayMatch='mmd', sigma_list=[1.]),
               semitorchstocclass.CIRMweigh(lamMatch=myargs.lamMatch, lamCIP=myargs.lamCIP, lamL2=lamL2, lamL1=lamL1, lr=myargs.lr,
                                        epochs=myargs.epochs, wayMatch='mmd', sigma_list=[1.])
              ]
elif myargs.tag_DA == 'DACIP':
    methods = [
               semiclass.CIP(lamCIP=myargs.lamCIP, lamL2=lamL2),
               semitorchclass.CIP(lamCIP=myargs.lamCIP, lamL2=lamL2, lamL1=lamL1, lr=myargs.lr,
                               epochs=myargs.epochs, wayMatch='mean+std+25p'),
              ]
elif myargs.tag_DA == 'DACIPMMD':
    methods = [
               semitorchstocclass.CIP(lamCIP=myargs.lamCIP, lamL2=lamL2, lamL1=lamL1, lr=myargs.lr,
                                  epochs=myargs.epochs, wayMatch='mmd', sigma_list=[1.]),
              ]


names = [str(m) for m in methods]
print(names)
names_short = [str(m).split('_')[0] for m in methods]
print(names_short)

seed1 = int(123456 + np.exp(2) * 1000)

params = {'M': 15, 'inter2noise_ratio': 1., 'd': 20, 'cicnum': 10,'interY': 1.}

sem1 = simudata.pick_sem('r0%sd?x4' %myargs.interv_type,
                         params=params,
                         seed=seed1+myargs.seed)

# run methods on data generated from sem
results_src_all, results_tar_all, results_minDiffIndx = util.run_all_methods(sem1,
                                                                             methods,
                                                                             n=myargs.n,
                                                                             repeats=1,
                                                                             returnMinDiffIndx=True,
                                                                             tag_DA=myargs.tag_DA)
res_all = {}
res_all['src'] = results_src_all
res_all['tar'] = results_tar_all
res_all['minDiffIndx'] = results_minDiffIndx


np.save("simu_results/sim_exp9_scat_r0%sd20x4_%s_lamMatch%s_lamCIP%s_n%d_epochs%d_seed%d.npy" %(myargs.interv_type,
        myargs.tag_DA, myargs.lamMatch, myargs.lamCIP, myargs.n, myargs.epochs, myargs.seed), res_all)


