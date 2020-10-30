#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
import argparse

np.set_printoptions(precision=3)

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import time

import simudata_MNIST
import semitorchMNISTclass


# In[2]:


# check gpu avail
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

# print(device)


# In[3]:

parser = argparse.ArgumentParser()
parser.add_argument("--perturb", type=str, default="whitepatch2M", help="type of perturbation")
parser.add_argument("--subset_prop", type=float, default=0.2, help="proportion of data points to be used for each env")
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--lamMatch", type=float, default=1., help="DIP matching penalty")
parser.add_argument("--lamMatchMMD", type=float, default=1., help="DIP matching penalty with MMD")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--tag_DA", type=str, default="baseline", help="whether to run baseline methods or DA methods")
myargs = parser.parse_args()
print(myargs)

M = 2 
train_batch_size = 64
test_batch_size = 500
save_model = False
np.random.seed(123456+myargs.seed)

trainloaders, testloaders = simudata_MNIST.generate_MNIST_envs(perturb=myargs.perturb, subset_prop=myargs.subset_prop, 
                                                               M=M, interY=False, 
                                                               train_batch_size=train_batch_size, 
                                                               test_batch_size=test_batch_size)

lamL2 = 0.
lamL1 = 0.
lr = 1e-4

source = list(np.arange(M))
target = M-1 
source.remove(target)

savefilename_prefix = 'results_MNIST/report_v8_%s_M%d_subsetprop%s_%s_lamMatch%s_lamMatchMMD%s_epochs%d_seed%d' % (myargs.perturb, M,
   str(myargs.subset_prop), myargs.tag_DA, myargs.lamMatch, myargs.lamMatchMMD, myargs.epochs, myargs.seed)
savefilename = '%s.txt' % savefilename_prefix 
savefile = open(savefilename, 'w')

if myargs.tag_DA == 'baseline':
    methods = [
               semitorchMNISTclass.Original(),
               semitorchMNISTclass.Tar(lamL2=lamL2, lamL1=lamL1, lr=lr, epochs=myargs.epochs),
               semitorchMNISTclass.SrcPool(lamL2=lamL2, lamL1=lamL1, lr=lr, epochs=myargs.epochs),
              ]
else: # DIP
    methods = [
               semitorchMNISTclass.DIP(lamMatch=myargs.lamMatch, lamL2=0., lamL1=0., 
                                         sourceInd = 0, lr=lr, epochs=myargs.epochs, wayMatch='mean'),
               semitorchMNISTclass.DIPOracle(lamMatch=myargs.lamMatch, lamL2=0., lamL1=0., 
                                         sourceInd = 0, lr=lr, epochs=myargs.epochs, wayMatch='mean'),
               semitorchMNISTclass.DIP_MMD(lamMatch=myargs.lamMatchMMD, lamL2=0., lamL1=0., 
                                         sourceInd = 0, lr=lr, epochs=myargs.epochs, wayMatch='mmd', sigma_list=[1., 10.]),
              ]

names = [str(m) for m in methods]
print(names, file=savefile)



trained_methods = [None]*len(methods)
results_src_all = np.zeros((M-1, len(methods), 2))
results_tar_all = np.zeros((len(methods), 2))

def compute_accuracy(loader, env, me):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader[env]:
            images, labels = data[0].to(device), data[1].to(device)
            predicted = me.predict(images)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

for i, me in enumerate(methods):
    starttime = time.time()
    print("fitting %s" %names[i], file=savefile)
    me = me.fit(trainloaders, source=source, target=target)
    trained_methods[i] = me
    # evaluate the methods
    # target train and test accuracy
    results_tar_all[i, 0] = compute_accuracy(trainloaders, target, me)
    results_tar_all[i, 1] = compute_accuracy(testloaders, target, me)
    # source train and test accuracy
    for j, sourcej in enumerate(source):
        results_src_all[j, i, 0] = compute_accuracy(trainloaders, sourcej, me)
        results_src_all[j, i, 1] = compute_accuracy(testloaders, sourcej, me)
    
            
    print('Method %-30s, Target %d, Source accuracy: %.3f %%, Target accuracy: %.3f %%' % (names[i], target, 
        100 * results_tar_all[i, 0], 100 * results_tar_all[i, 1]), file=savefile)
    endtime = time.time()
    print("time elapsed: %.1f s" % (endtime - starttime), file=savefile)
    print("\n", file=savefile)
    
results_all = {}
results_all['src'] = results_src_all
results_all['tar'] = results_tar_all
np.save("%s.npy" %savefilename_prefix, results_all)



