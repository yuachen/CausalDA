import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import time


# This file generates MNIST perturbed environments
datapath = '/cluster/home/chenyua/Code/causal/data'

def get_actual_data_idx(all_labels, subset_prop, interY=False):
    num_classes = 10
    # mask a certain proportion of data
    data_idx_mask_prop = torch.rand_like(all_labels.float()) < subset_prop
    
    if not interY:
        data_idx_list = torch.where(data_idx_mask_prop)[0]
    else:
        # intervention on Y
        # the following digits will only have 50% data
        mod_digits = [3, 4, 5, 6, 8, 9]
        data_idx_mask_mod = torch.rand_like(all_labels.float()) < 0.8
        idx_trainset_mod = (all_labels == mod_digits[0])
        for digit in mod_digits:
            idx_trainset_mod = idx_trainset_mod | (all_labels == digit)
        
        data_idx_mask_final = data_idx_mask_prop & (~(data_idx_mask_mod & idx_trainset_mod))
        data_idx_list = torch.where(data_idx_mask_final)[0]
    return data_idx_list
    

def generate_MNIST_envs(perturb='noisepatch', subset_prop=0.1, M=12, interY=False, train_batch_size=64, test_batch_size=1000):
    trainset_original = torchvision.datasets.MNIST(root=datapath, train=True,
                                          download=False)
    testset_original = torchvision.datasets.MNIST(root=datapath, train=False,
                                               download=False)
    
    # to be commented
#    idx_trainsetlist = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=interY)
#    idx_testsetlist = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=interY)

    
    
    trainloaders = {}
    testloaders = {}
    if perturb == 'whitepatch':
        # prepare the noise patces
        # noise_patches = [torch.zeros(1, 28, 28)]
        noise_patches = []
        offset = 10
        initpos = 2
        # sqsize 12 works to make CIRM better than CIP
        # offset = 4, sqsize = 12, initpos = 6, interY = 0.3 works for CIRM better than CIP, DIP
        sqsizesmall = 12
        sqsizelarge = 16
        if M == 12:
            for m in range(M-2):
                a = torch.zeros(1, 28, 28)
                a[0, (initpos-m+offset):initpos+sqsizelarge-m+offset, (initpos-m+offset):initpos+sqsizelarge-m+offset] = 3.25
                noise_patches.append(a)

            for m in [M-2, M-1]:
                a = torch.zeros(1, 28, 28)
                a[0, (initpos-m+offset):initpos+sqsizesmall-m+offset, (initpos-m+offset):initpos+sqsizesmall-m+offset] = 3.25
                noise_patches.append(a)
        elif M == 6:
            for m in range(M):
                a = torch.zeros(1, 28, 28)
                a[0, (initpos-2*m+offset):initpos+sqsizelarge-2*m+offset, (initpos-2*m+offset):initpos+sqsizelarge-2*m+offset] = 3.25
                noise_patches.append(a)

            
        # now transform the data
        for m in range(M):
            # load MNIST data
            transformer = torchvision.transforms.Compose(
                [
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.1306,), (0.3081,)),
                 torchvision.transforms.Lambda((lambda y: lambda x: torch.max(x, noise_patches[y]))(m)),
                 ])

            trainset = torchvision.datasets.MNIST(root=datapath, train=True,
                                                  download=False, transform=transformer)

            testset = torchvision.datasets.MNIST(root=datapath, train=False,
                                                   download=False, transform=transformer)
            
            if m != M-1:
                idx_trainsetlist_loc = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=False)
                idx_testsetlist_loc = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=False)
            else:
                idx_trainsetlist_loc = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=interY)
                idx_testsetlist_loc = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=interY)
            print('actual env=%d trainsize %s, testsize %s' %(m, idx_trainsetlist_loc.shape, idx_testsetlist_loc.shape))
            
            trainloaders[m] = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, num_workers=0, 
                                             sampler = torch.utils.data.sampler.SubsetRandomSampler(idx_trainsetlist_loc))
            testloaders[m] = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, num_workers=0, 
                                             sampler = torch.utils.data.sampler.SubsetRandomSampler(idx_testsetlist_loc))
            trainloaders[m].dataset.targets_mod = trainloaders[m].dataset.targets[idx_trainsetlist_loc]
            testloaders[m].dataset.targets_mod = testloaders[m].dataset.targets[idx_testsetlist_loc]
    elif perturb == 'whitepatch2M':
        # prepare the noise patces
        # noise_patches = [torch.zeros(1, 28, 28)]
        noise_patches = []
        offset = 10
        initpos = 2
        # sqsize 12 works to make CIRM better than CIP
        # offset = 4, sqsize = 12, initpos = 6, interY = 0.3 works for CIRM better than CIP, DIP
        sqsizesmall = 12
        for m in range(M):
            a = torch.zeros(1, 28, 28)
            # this will make the pixel white, 3.25 is because of normalization
            a[0, (initpos-m*5+offset):initpos+sqsizesmall-m+offset, (initpos-m+offset):initpos+sqsizesmall-m*5+offset] = 3.25
            noise_patches.append(a)
            
        # now transform the data
        for m in range(M):
            # load MNIST data
            transformer = torchvision.transforms.Compose(
                [
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.1306,), (0.3081,)),
                 torchvision.transforms.Lambda((lambda y: lambda x: torch.max(x, noise_patches[y]))(m)),
                 ])

            trainset = torchvision.datasets.MNIST(root=datapath, train=True,
                                                  download=True, transform=transformer)

            testset = torchvision.datasets.MNIST(root=datapath, train=False,
                                                   download=True, transform=transformer)
            
            if m != M-1:
                idx_trainsetlist_loc = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=False)
                idx_testsetlist_loc = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=False)
            else:
                idx_trainsetlist_loc = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=interY)
                idx_testsetlist_loc = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=interY)
            print('actual env=%d trainsize %s, testsize %s' %(m, idx_trainsetlist_loc.shape, idx_testsetlist_loc.shape))
            
            trainloaders[m] = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, num_workers=0, 
                                             sampler = torch.utils.data.sampler.SubsetRandomSampler(idx_trainsetlist_loc))
            testloaders[m] = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, num_workers=0, 
                                             sampler = torch.utils.data.sampler.SubsetRandomSampler(idx_testsetlist_loc))
            trainloaders[m].dataset.targets_mod = trainloaders[m].dataset.targets[idx_trainsetlist_loc]
            testloaders[m].dataset.targets_mod = testloaders[m].dataset.targets[idx_testsetlist_loc]
    elif perturb == 'noisepatch':
        # prepare the noise patces
        noise_patches = []
        offset = 10
        initpos = 2
        # sqsize 12 works to make CIRM better than CIP
        # offset = 4, sqsize = 12, initpos = 6, interY = 0.3 works for CIRM better than CIP, DIP
        sqsizesmall = 12
        sqsizelarge = 16
        for m in range(M-2):
            a = torch.zeros(1, 28, 28)
            a[0, (initpos-m+offset):initpos+sqsizelarge-m+offset, (initpos-m+offset):initpos+sqsizelarge-m+offset] = 3.25 * (torch.rand(1, sqsizelarge, sqsizelarge) > 0.5)
            noise_patches.append(a)

        for m in [M-2, M-1]:
            a = torch.zeros(1, 28, 28)
            a[0, (initpos-m+offset):initpos+sqsizesmall-m+offset, (initpos-m+offset):initpos+sqsizesmall-m+offset] = 3.25 * (torch.rand(1, sqsizesmall, sqsizesmall) > 0.5)
            noise_patches.append(a)
            
        # now transform the data
        for m in range(M):
            # load MNIST data
            transformer = torchvision.transforms.Compose(
                [
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.1306,), (0.3081,)),
                 torchvision.transforms.Lambda((lambda y: lambda x: torch.max(x, noise_patches[y]))(m)),
                 ])

            trainset = torchvision.datasets.MNIST(root=datapath, train=True,
                                                  download=True, transform=transformer)

            testset = torchvision.datasets.MNIST(root=datapath, train=False,
                                                   download=True, transform=transformer)
            
            if m != M-1:
                idx_trainsetlist_loc = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=False)
                idx_testsetlist_loc = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=False)
            else:
                idx_trainsetlist_loc = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=interY)
                idx_testsetlist_loc = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=interY)
            print('actual env=%d trainsize %s, testsize %s' %(m, idx_trainsetlist_loc.shape, idx_testsetlist_loc.shape))
            
            trainloaders[m] = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, num_workers=0, 
                                             sampler = torch.utils.data.sampler.SubsetRandomSampler(idx_trainsetlist_loc))
            testloaders[m] = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, num_workers=0, 
                                             sampler = torch.utils.data.sampler.SubsetRandomSampler(idx_testsetlist_loc))
            trainloaders[m].dataset.targets_mod = trainloaders[m].dataset.targets[idx_trainsetlist_loc]
            testloaders[m].dataset.targets_mod = testloaders[m].dataset.targets[idx_testsetlist_loc]
            
    elif perturb == 'rotation':
        angles = np.zeros(M)
        if M == 12:
            angles = np.arange(M) * 10 - 45
        elif M == 10:
            angles = np.arange(M) * 10 - 35
            angles[M-1] = 50
        elif M == 5:
            angles = np.arange(M) * 15 - 30 
        # now transform the data
        for m in range(M):
            # load MNIST data
            transformer = torchvision.transforms.Compose(
                    [torchvision.transforms.RandomRotation((angles[m], angles[m])),
                     torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize((0.1306,), (0.3081,))
                     ])
            trainset = torchvision.datasets.MNIST(root=datapath, train=True,
                                                      download=True, transform=transformer)

            testset = torchvision.datasets.MNIST(root=datapath, train=False,
                                                       download=True, transform=transformer)
        
            if m != M-1:
                idx_trainsetlist_loc = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=False)
                idx_testsetlist_loc = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=False)
            else:
                idx_trainsetlist_loc = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=interY)
                idx_testsetlist_loc = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=interY)
            print('actual env=%d trainsize %s, testsize %s' %(m, idx_trainsetlist_loc.shape, idx_testsetlist_loc.shape))

            trainloaders[m] = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, num_workers=0, 
                                             sampler = torch.utils.data.sampler.SubsetRandomSampler(idx_trainsetlist_loc))
            testloaders[m] = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, num_workers=0, 
                                             sampler = torch.utils.data.sampler.SubsetRandomSampler(idx_testsetlist_loc))
            trainloaders[m].dataset.targets_mod = trainloaders[m].dataset.targets[idx_trainsetlist_loc]
            testloaders[m].dataset.targets_mod = testloaders[m].dataset.targets[idx_testsetlist_loc]
    elif perturb == 'rotation2M' or perturb == 'rotation2Ma':
        if perturb == 'rotation2M':
            angles = [30, 45]
        else:
            angles = [10, 45]
        # now transform the data
        for m in range(M):
            # load MNIST data
            transformer = torchvision.transforms.Compose(
                    [torchvision.transforms.RandomRotation((angles[m], angles[m])),
                     torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize((0.1306,), (0.3081,))
                     ])
            trainset = torchvision.datasets.MNIST(root=datapath, train=True,
                                                      download=True, transform=transformer)

            testset = torchvision.datasets.MNIST(root=datapath, train=False,
                                                       download=True, transform=transformer)
        
            if m != M-1:
                idx_trainsetlist_loc = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=False)
                idx_testsetlist_loc = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=False)
            else:
                idx_trainsetlist_loc = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=interY)
                idx_testsetlist_loc = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=interY)
            print('actual env=%d trainsize %s, testsize %s' %(m, idx_trainsetlist_loc.shape, idx_testsetlist_loc.shape))

            trainloaders[m] = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, num_workers=0, 
                                             sampler = torch.utils.data.sampler.SubsetRandomSampler(idx_trainsetlist_loc))
            testloaders[m] = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, num_workers=0, 
                                             sampler = torch.utils.data.sampler.SubsetRandomSampler(idx_testsetlist_loc))
            trainloaders[m].dataset.targets_mod = trainloaders[m].dataset.targets[idx_trainsetlist_loc]
            testloaders[m].dataset.targets_mod = testloaders[m].dataset.targets[idx_testsetlist_loc]
            
    elif perturb == 'translation2M':
        translates = [(0.2, 0), (0, 0.2)]
        # now transform the data
        for m in range(M):
            # load MNIST data
            transformer = torchvision.transforms.Compose(
                    [torchvision.transforms.RandomAffine(0, translates[m]),
                     torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize((0.1306,), (0.3081,))
                     ])
            trainset = torchvision.datasets.MNIST(root=datapath, train=True,
                                                      download=True, transform=transformer)

            testset = torchvision.datasets.MNIST(root=datapath, train=False,
                                                       download=True, transform=transformer)
        
            if m != M-1:
                idx_trainsetlist_loc = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=False)
                idx_testsetlist_loc = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=False)
            else:
                idx_trainsetlist_loc = get_actual_data_idx(trainset_original.targets, subset_prop=subset_prop, interY=interY)
                idx_testsetlist_loc = get_actual_data_idx(testset_original.targets, subset_prop=subset_prop, interY=interY)
            print('actual env=%d trainsize %s, testsize %s' %(m, idx_trainsetlist_loc.shape, idx_testsetlist_loc.shape))

            trainloaders[m] = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, num_workers=0, 
                                             sampler = torch.utils.data.sampler.SubsetRandomSampler(idx_trainsetlist_loc))
            testloaders[m] = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, num_workers=0, 
                                             sampler = torch.utils.data.sampler.SubsetRandomSampler(idx_testsetlist_loc))
            trainloaders[m].dataset.targets_mod = trainloaders[m].dataset.targets[idx_trainsetlist_loc]
            testloaders[m].dataset.targets_mod = testloaders[m].dataset.targets[idx_testsetlist_loc]
            
    return trainloaders, testloaders
    
