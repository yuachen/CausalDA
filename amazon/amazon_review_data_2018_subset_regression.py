#!/usr/bin/env python
# coding: utf-8

import numpy as np

import pandas as pd
import os
import h5py
import sys
import argparse

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

np.set_printoptions(precision=3)

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# local packages
import sys
sys.path.append('../')
import semiclass
import semitorchclass
import semitorchstocclass
import util

# check gpu avail
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

parser = argparse.ArgumentParser()
parser.add_argument("--lamMatch", type=float, default=1., help="DIP matching penalty")
parser.add_argument("--lamCIP", type=float, default=0.1, help="CIP matching penalty")
parser.add_argument("--lamL2", type=float, default=1., help="L2 penalty")
parser.add_argument("--tag_DA", type=str, default="baseline", help="choose whether to run baseline methods or DA methods")
parser.add_argument("--seed", type=int, default=0, help="seed of experiment")
parser.add_argument("--target", type=int, default=0, help="target category")
parser.add_argument("--minDf", type=float, default=0.008, help="minimum term frequency")
parser.add_argument("--epochs", type=int, default=2000, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
myargs = parser.parse_args()
print(myargs)

data_folder = '../../data/amazon_review_data_2018_subset'

# 'All_Beauty_5', 'AMAZON_FASHION_5', 'Appliances_5', 'Gift_Cards_5', 'Magazine_Subscriptions_5'
# are removed for now for not enough number of data points
categories = [
              'Arts_Crafts_and_Sewing_5', 'Automotive_5', 'CDs_and_Vinyl_5',
              'Cell_Phones_and_Accessories_5', 'Digital_Music_5',
              'Grocery_and_Gourmet_Food_5', 'Industrial_and_Scientific_5', 'Luxury_Beauty_5',
              'Musical_Instruments_5', 'Office_Products_5',
              'Patio_Lawn_and_Garden_5', 'Pet_Supplies_5', 'Prime_Pantry_5',
              'Software_5', 'Tools_and_Home_Improvement_5', 'Toys_and_Games_5']

nb_reviews = 10000
dfs = {}
for i, cate in enumerate(categories):
    df = pd.read_csv('%s/%s_%d.csv' %(data_folder, cate, nb_reviews))
    dfs[i] = df
    print(cate, dfs[i].shape)

allReviews = pd.concat([dfs[i]['reviewText'] for i in range(len(categories))])
ngramMin = 1
ngramMax = 2
stop_words = 'english'
vectTF = TfidfVectorizer(min_df = myargs.minDf, stop_words=stop_words, ngram_range=(ngramMin, ngramMax)).fit(allReviews.values.astype('U'))
print("Number of reviews=%d, feature size=%d" %(allReviews.shape[0], len(vectTF.get_feature_names())))
print(vectTF.vocabulary_)

lamL1 = 0.

if myargs.tag_DA == 'baseline':
    methods = [
               semiclass.Tar(lamL2=myargs.lamL2),
               semiclass.SrcPool(lamL2=myargs.lamL2),
              ]
elif myargs.tag_DA == 'DAmean':
    methods = [
               semiclass.DIP(lamMatch=myargs.lamMatch, lamL2=myargs.lamL2, sourceInd=0),
               semiclass.DIPOracle(lamMatch=myargs.lamMatch, lamL2=myargs.lamL2, sourceInd=0),
               semiclass.DIPweigh(lamMatch=myargs.lamMatch, lamL2=myargs.lamL2),
               semiclass.CIP(lamCIP=myargs.lamCIP, lamL2=myargs.lamL2),
               semiclass.CIRMweigh(lamCIP=myargs.lamCIP, lamMatch=myargs.lamMatch, lamL2=myargs.lamL2),
              ]
elif myargs.tag_DA == 'DAstd':
    methods = [
               semitorchclass.DIP(lamMatch=myargs.lamMatch, lamL2=myargs.lamL2, lamL1=lamL1, sourceInd=0, lr=myargs.lr,
                              epochs=myargs.epochs, wayMatch='mean+std+25p'),
               semitorchclass.DIPweigh(lamMatch=myargs.lamMatch, lamL2=myargs.lamL2, lamL1=lamL1, lr=myargs.lr,
                                   epochs=myargs.epochs, wayMatch='mean+std+25p'),
               semitorchclass.CIP(lamCIP=myargs.lamCIP, lamL2=myargs.lamL2, lamL1=lamL1, lr=myargs.lr,
                               epochs=myargs.epochs, wayMatch='mean+std+25p'),
               semitorchclass.CIRMweigh(lamMatch=myargs.lamMatch, lamL2=myargs.lamL2, lamL1=lamL1, lr=myargs.lr,
                                    epochs=myargs.epochs, wayMatch='mean+std+25p'),
              ]
elif myargs.tag_DA == 'DAMMD':
    methods = [
               semitorchstocclass.DIP(lamMatch=myargs.lamMatch, lamL2=myargs.lamL2, lamL1=lamL1, sourceInd = 0, lr=myargs.lr,
                              epochs=myargs.epochs, wayMatch='mmd', sigma_list=[1.]),
               semitorchstocclass.DIPweigh(lamMatch=myargs.lamMatch, lamL2=myargs.lamL2, lamL1=lamL1, lr=myargs.lr,
                                       epochs=myargs.epochs, wayMatch='mmd', sigma_list=[1.]),
               semitorchstocclass.CIP(lamCIP=myargs.lamCIP, lamL2=myargs.lamL2, lamL1=lamL1, lr=myargs.lr,
                                  epochs=myargs.epochs, wayMatch='mmd', sigma_list=[1.]),
               semitorchstocclass.CIRMweigh(lamMatch=myargs.lamMatch, lamCIP=myargs.lamCIP, lamL2=myargs.lamL2, lamL1=lamL1, lr=myargs.lr,
                                        epochs=myargs.epochs, wayMatch='mmd', sigma_list=[1.])
              ]

names = [str(m) for m in methods]
print(names)

# random data split
random_state = 123456 + myargs.seed
datasets = {}
datasets_test = {}
for i, cate in enumerate(categories):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(dfs[i]['reviewText'].astype('U'), dfs[i]['overall'].astype('U'), test_size=0.10, random_state = random_state)


    X_train = vectTF.transform(X_train_raw)
    X_test = vectTF.transform(X_test_raw)


    datasets[i] = np.array(X_train.todense()), np.array(y_train, dtype=np.float32)
    datasets_test[i] = np.array(X_test.todense()), np.array(y_test, dtype=np.float32)

    print(cate, X_train.shape, X_test.shape)

# normalize
Xmean = 0
N = 0
for i, cate in enumerate(categories):
    Xmean += np.sum(datasets[i][0], axis=0)
    N += datasets[i][0].shape[0]
Xmean /= N


Xvar = 0
for i, cate in enumerate(categories):
    Xvar += np.sum((datasets[i][0] - Xmean.reshape(1, -1))**2, axis=0)
    N += datasets[i][0].shape[0]

Xvar /= N
Xstd = np.sqrt(Xvar)

for i, cate in enumerate(categories):
    x, y = datasets[i]
    x_test, y_test = datasets_test[i]
    datasets[i] = (x - Xmean)/Xstd, y
    datasets_test[i] = (x_test - Xmean)/Xstd, y_test

# create torch format data
dataTorch = {}
dataTorchTest = {}

for i in range(len(categories)):
    dataTorch[i] = [torch.from_numpy(datasets[i][0].astype(np.float32)).to(device),
                torch.from_numpy(datasets[i][1].astype(np.float32)).to(device)]
    dataTorchTest[i] = [torch.from_numpy(datasets_test[i][0].astype(np.float32)).to(device),
                torch.from_numpy(datasets_test[i][1].astype(np.float32)).to(device)]

train_batch_size = 500
test_batch_size = 500

trainloaders = {}
testloaders = {}

for i in range(len(categories)):
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(datasets[i][0]),
                                   torch.Tensor(datasets[i][1]))
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(datasets_test[i][0]),
                                   torch.Tensor(datasets_test[i][1]))
    trainloaders[i] = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)
    testloaders[i] = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size)


M = len(categories)
source = [i for i in range(M)]
source.remove(myargs.target)

print("source =", source, "target =", myargs.target, flush=True)
results_src_all = np.zeros((M-1, len(methods), 2))
results_tar_all = np.zeros((len(methods), 2))
results_minDiffIndx = {}
labeledsize_list = np.arange(1, 11) * 20
results_tar_sub_all = np.zeros((len(methods), len(labeledsize_list)))
for i, m in enumerate(methods):
    if m.__module__ == 'semiclass':
        me = m.fit(datasets, source=source, target=myargs.target)
        if hasattr(me, 'minDiffIndx'):
            print("best index="+str(me.minDiffIndx))
            results_minDiffIndx[(myargs.tag_DA, i)] = me.minDiffIndx
        xtar, ytar= datasets[myargs.target]
        xtar_test, ytar_test= datasets_test[myargs.target]
        targetE = util.MSE(me.ypred, ytar)
        targetNE = util.MSE(me.predict(xtar_test), ytar_test)
        for j, sourcej in enumerate(source):
            results_src_all[j, i, 0] = util.MSE(me.predict(datasets[sourcej][0]), datasets[sourcej][1])
            results_src_all[j, i, 1] = util.MSE(me.predict(datasets_test[sourcej][0]), datasets_test[sourcej][1])
        # obtain target error for each labeledsize
        for k, labeledsize in enumerate(labeledsize_list):
             xtar_sub = xtar[:labeledsize, :]
             ytar_sub = ytar[:labeledsize]
             results_tar_sub_all[i, k] = util.MSE(me.predict(xtar_sub), ytar_sub)
    elif  m.__module__ == 'semitorchclass':
        me = m.fit(dataTorch, source=source, target=myargs.target)
        if hasattr(me, 'minDiffIndx'):
            print("best index="+str(me.minDiffIndx))
            results_minDiffIndx[(myargs.tag_DA, i)] = me.minDiffIndx
        xtar, ytar= dataTorch[myargs.target]
        xtar_test, ytar_test= dataTorchTest[myargs.target]
        targetE = util.torchMSE(me.ypred, ytar)
        targetNE = util.torchMSE(me.predict(xtar_test), ytar_test)
        for j, sourcej in enumerate(source):
            results_src_all[j, i, 0] = util.torchMSE(me.predict(dataTorch[sourcej][0]), dataTorch[sourcej][1])
            results_src_all[j, i, 1] = util.torchMSE(me.predict(dataTorchTest[sourcej][0]), dataTorchTest[sourcej][1])
        for k, labeledsize in enumerate(labeledsize_list):
             xtar_sub = xtar[:labeledsize, :]
             ytar_sub = ytar[:labeledsize]
             results_tar_sub_all[i, k] = util.torchMSE(me.predict(xtar_sub), ytar_sub)
    elif m.__module__ == 'semitorchstocclass':
        me = m.fit(trainloaders, source=source, target=myargs.target)
        targetE = util.torchloaderMSE(me, trainloaders[myargs.target], device)
        targetNE = util.torchloaderMSE(me, testloaders[myargs.target], device)
        for j, sourcej in enumerate(source):
            results_src_all[j, i, 0] = util.torchloaderMSE(me, trainloaders[sourcej], device)
            results_src_all[j, i, 1] = util.torchloaderMSE(me, testloaders[sourcej], device)
    else:
        raise ValueError('error')
    results_tar_all[i, 0] = targetE
    results_tar_all[i, 1] = targetNE

res_all = {}
res_all['src'] = results_src_all
res_all['tar'] = results_tar_all
res_all['minDiffIndx'] = results_minDiffIndx
res_all['tar_sub'] = results_tar_sub_all
res_all['labeledsize_list'] = labeledsize_list

np.save('results_amazon/amazon_review_data_2018_N%d_%s_minDf%s_lamL2%s_lamMatch%s_lamCIP%s_target%d_seed%d.npy' %(
           nb_reviews, myargs.tag_DA, myargs.minDf, myargs.lamL2, myargs.lamMatch, myargs.lamCIP, myargs.target, myargs.seed), res_all)


