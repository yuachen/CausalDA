"""
``semitorchMNISTclass`` implements domain adaptation on MNIST
"""

from abc import abstractmethod
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

import sys
sys.path.append('../')
import mmd

# check gpu avail
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseDANet(nn.Module):
    """Basec Net class for the network for domain adaptation, only last layer should be learnt"""
    def __init__(self):
        super(BaseDANet, self).__init__()


    @abstractmethod
    def forward(self, x):
        """usual forward pass of the net"""

    @abstractmethod
    def get_feat_bf(self, x):
        """a getter for the features before last layer"""

    @abstractmethod
    def forward_fc2(self, x):
        """only forward the last layer without touching other layers"""


class LeNet(BaseDANet):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_feat_bf(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        return x

    def forward_fc2(self, x):
        x = self.fc2(x)
        return x


class BaseEstimator():
    """Base class for domain adaptation"""
    @abstractmethod
    def fit(self, data, source, target):
        """Fit model.
        Arguments:
            data (dict of (X, y) pairs): maps env index to the (X, y) pair in that env
            source (list of indexes): indexes of source envs
            target (int): single index of the target env
        """
        self.source = source
        self.target = target

        return self

    @abstractmethod
    def predict(self, X):
        """Use the learned estimator to predict labels on fresh target data X
        """

    def __str__(self):
        """For easy name printing
        """
        return self.__class__.__name__


def create_fixed_feat_LeNet():
    ORIGINAL_PATH = 'mycheckpoints/MNIST_2C2F.pt'
    model_LeNet = LeNet()
    model_LeNet.load_state_dict(torch.load(ORIGINAL_PATH, map_location=device))

    # make all layers fixed
    for param in model_LeNet.parameters():
        param.requires_grad = False

    # create a new last layer that is trainable
    num_ftrs = model_LeNet.fc2.in_features
    num_classes = model_LeNet.fc2.out_features

    model_LeNet.fc2 = nn.Linear(num_ftrs, num_classes)

    return model_LeNet


class Original(BaseEstimator):
    """Take the pretrained LeNet without modifying it"""
    def __init__(self):
        pass


    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        ORIGINAL_PATH = 'mycheckpoints/MNIST_2C2F.pt'
        model_LeNet = LeNet()
        model_LeNet.load_state_dict(torch.load(ORIGINAL_PATH, map_location=device))
        self.model = model_LeNet.to(device)

        return self

    def predict(self, X):
        output = self.model(X)
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def __str__(self):
        return self.__class__.__name__


class Tar(BaseEstimator):
    """Oracle Linear regression on the last layer (with l1 or l2 penalty) trained on the target domain"""
    def __init__(self, lamL2=0.0, lamL1=0.0, lr=3e-4, epochs=10):
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        model_LeNet = create_fixed_feat_LeNet().to(device)
        loss_fn = F.nll_loss
        opt = optim.SGD(model_LeNet.fc2.parameters(), lr=self.lr, momentum=0.9)
        discrepancy = torch.nn.MSELoss()

        self.losses = np.zeros(self.epochs)
        # oracle estimator uses target labels
        for epoch in range(self.epochs): # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(dataloaders[target]):
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                opt.zero_grad()
                # forward + backward + optimize
                outputs = model_LeNet(inputs)
                loss = loss_fn(outputs, labels) + \
                    self.lamL2 * torch.sum(model_LeNet.fc2.weight ** 2) + \
                    self.lamL1 * torch.sum(torch.abs(model_LeNet.fc2.weight))
                # Perform gradient descent

                loss.backward()
                opt.step()

                running_loss += loss.item()

            self.losses[epoch] = running_loss
        self.model = model_LeNet

        return self

    def predict(self, X):
        output = self.model(X)
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


class SrcPool(BaseEstimator):
    """Pool all source data together and then run linear regression
       with l1 or l2 penalty """
    def __init__(self, lamL2=0.0, lamL1=0.0, lr=3e-4, epochs=10):
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        model_LeNet = create_fixed_feat_LeNet().to(device)
        loss_fn = F.nll_loss
        opt = optim.SGD(model_LeNet.fc2.parameters(), lr=self.lr, momentum=0.9)
        discrepancy = torch.nn.MSELoss()

        self.losses = np.zeros(self.epochs)
        # oracle estimator uses target labels
        for epoch in range(self.epochs): # loop over the dataset multiple times
            running_loss = 0.0

            for i, data in enumerate(zip(*[dataloaders[m] for m in source])):
                opt.zero_grad()
                loss = 0
                for mindex, m in enumerate(source):
                    inputs, labels = data[mindex][0].to(device), data[mindex][1].to(device)
                    loss += loss_fn(model_LeNet(inputs), labels) / len(source)

                loss += self.lamL2 * torch.sum(model_LeNet.fc2.weight ** 2) + \
                    self.lamL1 * torch.sum(torch.abs(model_LeNet.fc2.weight))


                loss.backward()
                opt.step()

                running_loss += loss.item()

            self.losses[epoch] = running_loss
        self.model = model_LeNet

        return self

    def predict(self, X):
        output = self.model(X)
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


# def wayMatchSelector(wayMatch='mean'):
#     if wayMatch == 'mean':
#         return [lambda x: torch.mean(x, dim=0)]
#     elif wayMatch == 'std':
#         return [lambda x: torch.std(x, dim=0)]
#     elif wayMatch == '25p':
#         return [lambda x: torch.kthvalue(x, (1 + round(.25 * (x.shape[0] - 1))), dim=0)]
#     elif wayMatch == '75p':
#         return [lambda x: torch.kthvalue(x, (1 + round(.75 * (x.shape[0] - 1))), dim=0)]
#     elif wayMatch == 'mean+std':
#         return [lambda x: torch.mean(x, dim=0), lambda x: torch.std(x, dim=0)]
#     elif wayMatch == 'mean+std+25p':
#         return [lambda x: torch.mean(x, dim=0), lambda x: torch.std(x, dim=0), lambda x: torch.kthvalue(x, (1 + round(.25 * (x.shape[0] - 1))), dim=0)[0]]
#     elif wayMatch == 'mean+std+25p+75p':
#         return [lambda x: torch.mean(x, dim=0), lambda x: torch.std(x, dim=0), lambda x: torch.kthvalue(x, (1 + round(.25 * (x.shape[0] - 1))), dim=0)[0],
#             lambda x: torch.kthvalue(x, (1 + round(.75 * (x.shape[0] - 1))), dim=0)[0]]
#     else:
#         print("Error: wayMatch not specified correctly, using mean")
#         return [lambda x: torch.mean(x, 0)]

def matchPenaltyStats(dataloader1, model, wayMatch='mean'):
    """The penalty term for a model"""
    if wayMatch == 'mean':
        mean1 = 0
        nb1 = 0
        for i, data in enumerate(dataloader1):
            inputs1, _ = data[0].to(device), data[1].to(device)
            mean1 += torch.sum(model.get_feat_bf(inputs1), dim=0)
            nb1 += inputs1.shape[0]
        mean1 /= nb1
        return mean1
    else:
        return None



class DIP(BaseEstimator):
    """Pick one source, match mean of X * beta between source and target"""
    def __init__(self, lamMatch=10., lamL2=0., lamL1=0., sourceInd = 0, lr=1e-4, epochs=10, wayMatch='mean'):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.sourceInd = sourceInd
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatch

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        model_LeNet = create_fixed_feat_LeNet().to(device)
        loss_fn = F.nll_loss
        opt = optim.SGD(model_LeNet.fc2.parameters(), lr=self.lr, momentum=0.9)
        discrepancy = torch.nn.MSELoss()

        self.losses = np.zeros(self.epochs)

        feat_bf1 = matchPenaltyStats(dataloaders[source[self.sourceInd]],
                                     model_LeNet, wayMatch='mean')
        feat_bf2 = matchPenaltyStats(dataloaders[target], model_LeNet, wayMatch='mean')
        # oracle estimator uses target labels
        for epoch in range(self.epochs): # loop over the dataset multiple times
            running_loss = 0.0

            for i, data in enumerate(zip(dataloaders[source[self.sourceInd]], dataloaders[target])):
                opt.zero_grad()
                loss = 0
                inputs, labels = data[0][0].to(device), data[0][1].to(device)
                loss += loss_fn(model_LeNet(inputs), labels)
                loss += self.lamMatch * discrepancy(model_LeNet.forward_fc2(feat_bf1), model_LeNet.forward_fc2(feat_bf2))
                loss += self.lamL2 * torch.sum(model_LeNet.fc2.weight ** 2) + \
                    self.lamL1 * torch.sum(torch.abs(model_LeNet.fc2.weight))


                loss.backward()
                opt.step()

                running_loss += loss.item()

            self.losses[epoch] = running_loss
        self.model = model_LeNet

        return self

    def predict(self, X):
        output = self.model(X)
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


class DIP_MMD(BaseEstimator):
    """Pick one source, match generic feature of X * beta between source and target,
       generic match only matches on a batch, so needs the batch size to be large to avoid finite sample error
    """
    def __init__(self, lamMatch=10., lamL2=0., lamL1=0., sourceInd = 0, lr=1e-4, epochs=10, wayMatch='mmd', sigma_list=[0.1, 1.0, 10]):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.sourceInd = sourceInd
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatch
        self.sigma_list = sigma_list 

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        model_LeNet = create_fixed_feat_LeNet().to(device)
        loss_fn = F.nll_loss
        opt = optim.SGD(model_LeNet.fc2.parameters(), lr=self.lr, momentum=0.9)
        discrepancy = torch.nn.MSELoss()

        self.losses = np.zeros(self.epochs)

        # oracle estimator uses target labels
        for epoch in range(self.epochs): # loop over the dataset multiple times
            running_loss = 0.0

            for i, data in enumerate(zip(dataloaders[source[self.sourceInd]], dataloaders[target])):
                opt.zero_grad()
                loss = 0
                inputs, labels = data[0][0].to(device), data[0][1].to(device)
                inputstar = data[1][0].to(device)
                loss += loss_fn(model_LeNet(inputs), labels)
                #loss += self.lamMatch * mmd.mix_rbf_mmd2(model_LeNet(inputs), model_LeNet(inputstar), self.sigma_list)
                loss += self.lamMatch * mmd.mix_rbf_mmd2(model_LeNet.forward_fc2(model_LeNet.get_feat_bf(inputs)),
                                                         model_LeNet.forward_fc2(model_LeNet.get_feat_bf(inputstar)), self.sigma_list)
                loss += self.lamL2 * torch.sum(model_LeNet.fc2.weight ** 2) + \
                    self.lamL1 * torch.sum(torch.abs(model_LeNet.fc2.weight))


                loss.backward()
                opt.step()

                running_loss += loss.item()

            self.losses[epoch] = running_loss
        self.model = model_LeNet

        return self

    def predict(self, X):
        output = self.model(X)
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)



class DIPOracle(BaseEstimator):
    """use the target label directly, match mean of X * beta between source and target"""
    def __init__(self, lamMatch=10., lamL2=0., lamL1=0., sourceInd = 0, lr=1e-4, epochs=10, wayMatch='mean'):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.sourceInd = sourceInd
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatch

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        model_LeNet = create_fixed_feat_LeNet().to(device)
        loss_fn = F.nll_loss
        opt = optim.SGD(model_LeNet.fc2.parameters(), lr=self.lr, momentum=0.9)
        discrepancy = torch.nn.MSELoss()

        self.losses = np.zeros(self.epochs)

        feat_bf1 = matchPenaltyStats(dataloaders[source[self.sourceInd]],
                                     model_LeNet, wayMatch='mean')
        feat_bf2 = matchPenaltyStats(dataloaders[target], model_LeNet, wayMatch='mean')
        # oracle estimator uses target labels
        for epoch in range(self.epochs): # loop over the dataset multiple times
            running_loss = 0.0

            for i, data in enumerate(zip(dataloaders[source[self.sourceInd]], dataloaders[target])):
                opt.zero_grad()
                loss = 0
                inputs, labels = data[1][0].to(device), data[1][1].to(device)
                loss += loss_fn(model_LeNet(inputs), labels)
                loss += self.lamMatch * discrepancy(model_LeNet.forward_fc2(feat_bf1), model_LeNet.forward_fc2(feat_bf2))
                loss += self.lamL2 * torch.sum(model_LeNet.fc2.weight ** 2) + \
                    self.lamL1 * torch.sum(torch.abs(model_LeNet.fc2.weight))


                loss.backward()
                opt.step()

                running_loss += loss.item()

            self.losses[epoch] = running_loss
        self.model = model_LeNet

        return self

    def predict(self, X):
        output = self.model(X)
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


class DIPOracle_MMD(BaseEstimator):
    """use the target label directly, match mean of X * beta between source and target"""
    def __init__(self, lamMatch=10., lamL2=0., lamL1=0., sourceInd = 0, lr=1e-4, epochs=10, wayMatch='mmd', sigma_list=[0.1, 1., 10]):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.sourceInd = sourceInd
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatch
        self.sigma_list = sigma_list

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        model_LeNet = create_fixed_feat_LeNet().to(device)
        loss_fn = F.nll_loss
        opt = optim.SGD(model_LeNet.fc2.parameters(), lr=self.lr, momentum=0.9)
        discrepancy = torch.nn.MSELoss()

        self.losses = np.zeros(self.epochs)

        # oracle estimator uses target labels
        for epoch in range(self.epochs): # loop over the dataset multiple times
            running_loss = 0.0

            for i, data in enumerate(zip(dataloaders[source[self.sourceInd]], dataloaders[target])):
                opt.zero_grad()
                loss = 0
                inputs, labels = data[1][0].to(device), data[1][1].to(device)
                inputsrc = data[0][0].to(device)
                loss += loss_fn(model_LeNet(inputs), labels)
                loss += self.lamMatch * mmd.mix_rbf_mmd2(model_LeNet(inputs), model_LeNet(inputsrc), self.sigma_list)
                loss += self.lamL2 * torch.sum(model_LeNet.fc2.weight ** 2) + \
                    self.lamL1 * torch.sum(torch.abs(model_LeNet.fc2.weight))


                loss.backward()
                opt.step()

                running_loss += loss.item()

            self.losses[epoch] = running_loss
        self.model = model_LeNet

        return self

    def predict(self, X):
        output = self.model(X)
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


class DIPweigh(BaseEstimator):
    '''loop throught all source envs, match the mean of X * beta between source env i and target, weigh the final prediction based loss of env i'''
    def __init__(self, lamMatch=10., lamL2=0., lamL1=0., lr=1e-4, epochs=10, wayMatch='mean'):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatch

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        models = {}
        diffs = {}
        losses_all = {}
        self.total_weight = 0

        for m in source:
            model_LeNet = create_fixed_feat_LeNet().to(device)
            models[m] = model_LeNet
            loss_fn = F.nll_loss
            opt = optim.SGD(model_LeNet.fc2.parameters(), lr=self.lr, momentum=0.9)
            discrepancy = torch.nn.MSELoss()

            losses_all[m] = np.zeros(self.epochs)

            feat_bf1 = matchPenaltyStats(dataloaders[m],
                                         model_LeNet, wayMatch='mean')
            feat_bf2 = matchPenaltyStats(dataloaders[target], model_LeNet, wayMatch='mean')
            # oracle estimator uses target labels
            for epoch in range(self.epochs): # loop over the dataset multiple times
                running_loss = 0.0

                for i, data in enumerate(zip(dataloaders[m], dataloaders[target])):
                    opt.zero_grad()
                    loss = 0
                    inputs, labels = data[0][0].to(device), data[0][1].to(device)
                    loss += loss_fn(model_LeNet(inputs), labels)
                    loss += self.lamMatch * discrepancy(model_LeNet.forward_fc2(feat_bf1), model_LeNet.forward_fc2(feat_bf2))
                    loss += self.lamL2 * torch.sum(model_LeNet.fc2.weight ** 2) + \
                        self.lamL1 * torch.sum(torch.abs(model_LeNet.fc2.weight))


                    loss.backward()
                    opt.step()

                    running_loss += loss.item()

                losses_all[m][epoch] = running_loss
            diffs[m] = discrepancy(model_LeNet.forward_fc2(feat_bf1), model_LeNet.forward_fc2(feat_bf2))
            self.total_weight += torch.exp(-1000.*diffs[m])

        self.models = models
        self.diffs = diffs

        minDiff = diffs[source[0]]
        minDiffIndx = source[0]
        for m in source:
            if diffs[m] < minDiff:
                minDiff = diffs[m]
                minDiffIndx = m
        print(minDiffIndx)
        self.minDiffIndx = minDiffIndx
        self.losses = losses_all[minDiffIndx]

        return self

    def predict(self, X):
        output = 0
        for m in self.source:
            output += torch.exp(-1000.*self.diffs[m]) * self.models[m](X)
        output /= self.total_weight
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


class DIPweigh_MMD(BaseEstimator):
    '''loop throught all source envs, match the mean of X * beta between source env i and target, weigh the final prediction based loss of env i
    generic
    '''
    def __init__(self, lamMatch=10., lamL2=0., lamL1=0., lr=1e-4, epochs=10, wayMatch='mmd', sigma_list=[0.1, 1., 10]):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatch
        self.sigma_list = sigma_list

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        models = {}
        diffs = {}
        losses_all = {}
        self.total_weight = 0

        for m in source:
            model_LeNet = create_fixed_feat_LeNet().to(device)
            models[m] = model_LeNet
            loss_fn = F.nll_loss
            opt = optim.SGD(model_LeNet.fc2.parameters(), lr=self.lr, momentum=0.9)
            discrepancy = torch.nn.MSELoss()

            losses_all[m] = np.zeros(self.epochs)

            # oracle estimator uses target labels
            for epoch in range(self.epochs): # loop over the dataset multiple times
                running_loss = 0.0

                for i, data in enumerate(zip(dataloaders[m], dataloaders[target])):
                    opt.zero_grad()
                    loss = 0
                    inputs, labels = data[0][0].to(device), data[0][1].to(device)
                    inputstar = data[1][0].to(device)
                    loss += loss_fn(model_LeNet(inputs), labels)
                    # loss += self.lamMatch * mmd.mix_rbf_mmd2(model_LeNet(inputs), model_LeNet(inputstar), self.sigma_list)
                    loss += self.lamMatch * mmd.mix_rbf_mmd2(model_LeNet.forward_fc2(model_LeNet.get_feat_bf(inputs)),
                                                         model_LeNet.forward_fc2(model_LeNet.get_feat_bf(inputstar)), self.sigma_list)
                    loss += self.lamL2 * torch.sum(model_LeNet.fc2.weight ** 2) + \
                        self.lamL1 * torch.sum(torch.abs(model_LeNet.fc2.weight))


                    loss.backward()
                    opt.step()

                    running_loss += loss.item()

                losses_all[m][epoch] = running_loss


            # need to calculate the diffs
            diffs[m] = 0
            with torch.no_grad():
                for i, data in enumerate(zip(dataloaders[m], dataloaders[target])):
                    inputs, labels = data[0][0].to(device), data[0][1].to(device)
                    inputstar = data[1][0].to(device)
                    # local_match_res_mmd =  mmd.mix_rbf_mmd2(model_LeNet(inputs), model_LeNet(inputstar), self.sigma_list)
                    local_match_res_mmd = mmd.mix_rbf_mmd2(model_LeNet.forward_fc2(model_LeNet.get_feat_bf(inputs)),
                                                         model_LeNet.forward_fc2(model_LeNet.get_feat_bf(inputstar)), self.sigma_list) 
                    diffs[m] += local_match_res_mmd / self.epochs / (len(dataloaders[m].dataset)/dataloaders[m].batch_size)
            self.total_weight += torch.exp(-1000.*diffs[m])

        self.models = models
        self.diffs = diffs

        minDiff = diffs[source[0]]
        minDiffIndx = source[0]
        for m in source:
            if diffs[m] < minDiff:
                minDiff = diffs[m]
                minDiffIndx = m
        print(minDiffIndx)
        self.minDiffIndx = minDiffIndx
        self.losses = losses_all[minDiffIndx]

        return self

    def predict(self, X):
        output = 0
        for m in self.source:
            output += torch.exp(-1000.*self.diffs[m]) * self.models[m](X)
        output /= self.total_weight
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


class CIP(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, no target env is needed"""
    def __init__(self, lamCIP=10., lamL2=0., lamL1=0., lr=1e-4, epochs=10, wayMatch='mean'):
        self.lamCIP = lamCIP
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatch

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        model_LeNet = create_fixed_feat_LeNet().to(device)
        loss_fn = F.nll_loss
        opt = optim.SGD(model_LeNet.fc2.parameters(), lr=self.lr, momentum=0.9)
        discrepancy = torch.nn.MSELoss()


        # First, get before last layer feature mean from source and target
        num_classes = 10
        feat_bf_fc2 = {}
        nb_samples_per_class = {}
        for m in source:
            for k in range(num_classes):
                feat_bf_fc2[(m, k)] = 0
                nb_samples_per_class[(m, k)] = 0

        for m in source:
            for i, data in enumerate(dataloaders[m]):
                inputs, labels = data[0].to(device), data[1].to(device)
                for k in range(num_classes):
                    a = torch.sum(labels==k)
                    if a != 0:
                        feat_bf_fc2[(m, k)] += torch.sum(model_LeNet.get_feat_bf(inputs[labels==k]), 0)
                        nb_samples_per_class[(m, k)] += a

        for m in source:
            for k in range(num_classes):
                feat_bf_fc2[(m, k)] /= nb_samples_per_class[(m, k)].float()

        self.losses = np.zeros(self.epochs)

        # Second, minimize loss with conditional invariance penalty
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(zip(*[dataloaders[m] for m in source])):
                loss = 0
                for mindex, m in enumerate(source):
                    inputs, labels = data[mindex][0].to(device), data[mindex][1].to(device)
                    loss += loss_fn(model_LeNet(inputs), labels)/float(len(source))

                    # conditional invariance penalty
                    for j in source:
                        if j > m:
                            for k in range(num_classes):
                                loss += self.lamCIP/float(len(source)**2) *  \
                                           discrepancy(model_LeNet.forward_fc2(feat_bf_fc2[(m, k)]),
                                                       model_LeNet.forward_fc2(feat_bf_fc2[(j, k)]))
                loss += self.lamL2 * torch.sum(model_LeNet.fc2.weight ** 2) + \
                        self.lamL1 * torch.sum(torch.abs(model_LeNet.fc2.weight))
                # Perform gradient descent
                loss.backward()
                opt.step()
                opt.zero_grad()
                running_loss += loss.item()
            self.losses[epoch] = running_loss
        self.model = model_LeNet

        return self

    def predict(self, X):
        output = self.model(X)
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_CIP{:.1f}".format(self.lamCIP) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)



class CIP_MMD(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, no target env is needed"""
    def __init__(self, lamCIP=10., lamL2=0., lamL1=0., lr=1e-4, epochs=10, wayMatch='mmd', sigma_list=[0.1, 1., 10]):
        self.lamCIP = lamCIP
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatch
        self.sigma_list = sigma_list

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        model_LeNet = create_fixed_feat_LeNet().to(device)
        loss_fn = F.nll_loss
        opt = optim.SGD(model_LeNet.fc2.parameters(), lr=self.lr, momentum=0.9)
        discrepancy = torch.nn.MSELoss()

        num_classes = 10

        self.losses = np.zeros(self.epochs)

        # Second, minimize loss with conditional invariance penalty
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(zip(*[dataloaders[m] for m in source])):
                loss = 0
                for mindex, m in enumerate(source):
                    inputs, labels = data[mindex][0].to(device), data[mindex][1].to(device)
                    # loss for one source env
                    loss += loss_fn(model_LeNet(inputs), labels)/float(len(source))

                    # conditional invariance penalty
                    for jindex, j in enumerate(source):
                        inputsj, labelsj = data[jindex][0].to(device), data[jindex][1].to(device)
                        if j > m:
                            for k in range(num_classes):
                                num_mk = torch.sum(labels==k)
                                num_jk = torch.sum(labelsj==k)
                                if num_mk > 0 and num_jk > 0:
                                    #loss += self.lamCIP/float(len(source)**2) *  \
                              # mmd.mix_rbf_mmd2(model_LeNet(inputs[labels==k]), model_LeNet(inputsj[labelsj==k]), self.sigma_list)
                                    loss += self.lamCIP/float(len(source)**2) *  \
                               mmd.mix_rbf_mmd2(model_LeNet.forward_fc2(model_LeNet.get_feat_bf(inputs[labels==k])),
                                                  model_LeNet.forward_fc2(model_LeNet.get_feat_bf(inputsj[labelsj==k])), self.sigma_list)
                loss += self.lamL2 * torch.sum(model_LeNet.fc2.weight ** 2) + \
                        self.lamL1 * torch.sum(torch.abs(model_LeNet.fc2.weight))
                # Perform gradient descent
                loss.backward()
                opt.step()
                opt.zero_grad()
                running_loss += loss.item()
            self.losses[epoch] = running_loss
        self.model = model_LeNet

        return self

    def predict(self, X):
        output = self.model(X)
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_CIP{:.1f}".format(self.lamCIP) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)



class CIRMweigh(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, use Yhat as proxy of Y to remove the Y parts in X.
    Match on the residual between one source env and target env"""
    def __init__(self, lamMatch=10., lamCIP=10., lamL2=0., lamL1=0., lr=1e-4, epochs=10, wayMatch='mean'):
        self.lamMatch = lamMatch
        self.lamCIP = lamCIP
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatch

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        models1_LeNet = create_fixed_feat_LeNet().to(device)
        loss_fn = F.nll_loss
        opt = optim.SGD(models1_LeNet.fc2.parameters(), lr=self.lr, momentum=0.9)
        discrepancy = torch.nn.MSELoss()


        # Step 1: use source envs to match the conditional mean
        # get before last layer feature mean from source and target
        num_classes = 10
        feat_bf_fc2 = {}
        nb_samples_per_class = {}
        for m in source:
            for k in range(num_classes):
                feat_bf_fc2[(m, k)] = 0
                nb_samples_per_class[(m, k)] = 0

        for m in source:
            for i, data in enumerate(dataloaders[m]):
                inputs, labels = data[0].to(device), data[1].to(device)
                for k in range(num_classes):
                    a = torch.sum(labels==k)
                    if a != 0:
                        feat_bf_fc2[(m, k)] += torch.sum(models1_LeNet.get_feat_bf(inputs[labels==k]), 0)
                        nb_samples_per_class[(m, k)] += a

        for m in source:
            for k in range(num_classes):
                feat_bf_fc2[(m, k)] /= nb_samples_per_class[(m, k)].float()

        losses1 = np.zeros(self.epochs)

        # minimize loss with conditional invariance penalty
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(zip(*[dataloaders[m] for m in source])):
                loss = 0
                for mindex, m in enumerate(source):
                    inputs, labels = data[mindex][0].to(device), data[mindex][1].to(device)
                    loss += loss_fn(models1_LeNet(inputs), labels)/float(len(source))

                    # conditional invariance penalty
                    for j in source:
                        if j > m:
                            for k in range(num_classes):
                                loss += self.lamCIP/float(len(source)**2) *  \
                                           discrepancy(models1_LeNet.forward_fc2(feat_bf_fc2[(m, k)]),
                                                       models1_LeNet.forward_fc2(feat_bf_fc2[(j, k)]))
                loss += self.lamL2 * torch.sum(models1_LeNet.fc2.weight ** 2) + \
                        self.lamL1 * torch.sum(torch.abs(models1_LeNet.fc2.weight))
                # Perform gradient descent
                loss.backward()
                opt.step()
                opt.zero_grad()
                running_loss += loss.item()
            losses1[epoch] = running_loss
        self.models1 = models1_LeNet

        # fix grads now
        for param in models1_LeNet.fc2.parameters():
            param.requires_grad = False

        # Step 2: remove the invariant part on all source envs, so that everything is independent of Y
        # get that coefficient b
        with torch.no_grad():
            meanYY = 0
            meanYX = 0
            for m in source:
                sizeofdatam = dataloaders[m].dataset.targets_mod.shape[0]
                Ym_onehot = torch.zeros((sizeofdatam, num_classes), dtype=torch.float)
                Ym_onehot = Ym_onehot.scatter_(1, dataloaders[m].dataset.targets_mod.reshape(-1, 1), 1)
                Ym_onehot_sum = Ym_onehot.sum(0).reshape(1, -1).to(device)

                for i, data in enumerate(dataloaders[m]):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    labels_onehot = torch.zeros((labels.shape[0], num_classes), dtype=torch.float).to(device)
                    labels_onehot = labels_onehot.scatter_(1, labels.reshape(-1, 1), 1)
                    labels_onehot_normalized = (labels_onehot/Ym_onehot_sum).to(device)
                    labels_onehot_demean = labels_onehot_normalized - \
                labels_onehot_normalized.mm((torch.ones(num_classes, num_classes)/num_classes).to(device))

                    outputs_bf_fc2 = models1_LeNet.get_feat_bf(inputs)
                    outputs_at_fc2 = torch.nn.functional.softmax(models1_LeNet.forward_fc2(outputs_bf_fc2), 1)
                    outputs_at_fc2_normalized = outputs_at_fc2[:, 0:(num_classes-1)] -  outputs_at_fc2[:, (num_classes-1):num_classes]

                    meanYYlocal = torch.mm(labels_onehot_demean[:, 0:(num_classes-1)].t(), outputs_at_fc2_normalized)
                    meanYXlocal = torch.mm(labels_onehot_demean[:, 0:(num_classes-1)].t(), outputs_bf_fc2)
                    meanYY += meanYYlocal/sizeofdatam
                    meanYX += meanYXlocal/sizeofdatam

            b, _ = torch.solve(meanYX, meanYY)
            print(b.shape)
            self.b = b

        # Step 3: mean match between source and target on the residual, after  transforming the covariates X - (X * beta_invariant) * b_invariant
        feat_bf_fc2_mod = {}
        with torch.no_grad():
            for m in (list(source) + [target]):
            # first get modified before-fc2 feature mean from source and target
                feat_bf_fc2_mod[m] = 0

                for i, data in enumerate(dataloaders[m]):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    labels_onehot = torch.zeros((labels.shape[0], num_classes), dtype=torch.float).to(device)
                    labels_onehot = labels_onehot.scatter_(1, labels.reshape(-1, 1), 1)

                    outputs_bf_fc2 = models1_LeNet.get_feat_bf(inputs)
                    outputs_at_fc2 = torch.nn.functional.softmax(models1_LeNet.forward_fc2(outputs_bf_fc2), 1)

                    outputs_at_fc2_sum = torch.sum(outputs_at_fc2, 0).reshape(1, -1)
                    outputs_at_fc2_sum_normalized = outputs_at_fc2_sum[:, 0:(num_classes-1)] - outputs_at_fc2_sum[:, (num_classes-1):num_classes]
                    feat_bf_fc2_mod[m] += torch.sum(outputs_bf_fc2, 0) - 1.* torch.mm(outputs_at_fc2_sum_normalized, b)


                feat_bf_fc2_mod[m] /= dataloaders[m].dataset.targets_mod.shape[0]

        # second run mean match on the modified before-fc2 feature
        models = {}
        diffs = {}
        losses_all = {}
        self.total_weight = 0

        for m in source:
            models[m] = create_fixed_feat_LeNet().to(device)
            loss_fn = F.nll_loss
            opt = optim.SGD(models[m].fc2.parameters(), lr=self.lr, momentum=0.9)
            discrepancy = torch.nn.MSELoss()

            losses_all[m] = np.zeros(self.epochs)

            feat_bf1_mod = feat_bf_fc2_mod[m]
            feat_bf2_mod = feat_bf_fc2_mod[target]
            # oracle estimator uses target labels
            for epoch in range(self.epochs): # loop over the dataset multiple times
                running_loss = 0.0

                for i, data in enumerate(zip(dataloaders[m], dataloaders[target])):
                    opt.zero_grad()
                    loss = 0
                    inputs, labels = data[0][0].to(device), data[0][1].to(device)
                    loss += loss_fn(models[m](inputs), labels)
                    loss += self.lamMatch * discrepancy(models[m].forward_fc2(feat_bf1_mod), models[m].forward_fc2(feat_bf2_mod))
                    loss += self.lamL2 * torch.sum(models[m].fc2.weight ** 2) + \
                        self.lamL1 * torch.sum(torch.abs(models[m].fc2.weight))

                    loss.backward()
                    opt.step()

                    running_loss += loss.item()

                losses_all[m][epoch] = running_loss
            diffs[m] = discrepancy(models[m].forward_fc2(feat_bf1_mod), models[m].forward_fc2(feat_bf2_mod))
            self.total_weight += torch.exp(-1000.*diffs[m])

        self.models = models
        self.diffs = diffs
        # print min diff
        minDiff = diffs[source[0]]
        minDiffIndx = source[0]
        for m in source:
            if diffs[m] < minDiff:
                minDiff = diffs[m]
                minDiffIndx = m
        print(minDiffIndx)
        self.minDiffIndx = minDiffIndx
        self.losses = losses_all[minDiffIndx]

        return self

    def predictSingle(self, X):
        output = self.models[m](X)
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def predict(self, X):
        output = 0
        for m in self.source:
            output += torch.exp(-1000.*self.diffs[m]) * self.models[m](X)
        output /= self.total_weight
        _, ypredX = torch.max(output.data, 1)
        return ypredX


    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)



class CIRMweigh_MMD(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, use Yhat as proxy of Y to remove the Y parts in X.
    Match on the residual between one source env and target env"""
    def __init__(self, lamMatch=10., lamCIP=10., lamL2=0., lamL1=0., lr=1e-4, epochs=10, wayMatch='mmd', sigma_list=[0.1, 1., 10]):
        self.lamMatch = lamMatch
        self.lamCIP = lamCIP
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatch
        self.sigma_list = sigma_list

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        models1_LeNet = create_fixed_feat_LeNet().to(device)
        loss_fn = F.nll_loss
        opt = optim.SGD(models1_LeNet.fc2.parameters(), lr=self.lr, momentum=0.9)
        discrepancy = torch.nn.MSELoss()


        # Step 1: use source envs to match the conditional mean
        num_classes = 10

        losses1 = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(zip(*[dataloaders[m] for m in source])):
                loss = 0
                for mindex, m in enumerate(source):
                    inputs, labels = data[mindex][0].to(device), data[mindex][1].to(device)
                    # loss for one source env
                    loss += loss_fn(models1_LeNet(inputs), labels)/float(len(source))

                    # conditional invariance penalty
                    for jindex, j in enumerate(source):
                        inputsj, labelsj = data[jindex][0].to(device), data[jindex][1].to(device)
                        if j > m:
                            for k in range(num_classes):
                                num_mk = torch.sum(labels==k)
                                num_jk = torch.sum(labelsj==k)
                                if num_mk > 0 and num_jk > 0:
                                  #  loss += self.lamCIP/float(len(source)**2) *  \
                             # mmd.mix_rbf_mmd2(models1_LeNet(inputs[labels==k]), models1_LeNet(inputsj[labelsj==k]), self.sigma_list)
                                    loss += self.lamCIP/float(len(source)**2) *  \
                               mmd.mix_rbf_mmd2(models1_LeNet.forward_fc2(models1_LeNet.get_feat_bf(inputs[labels==k])),
                                                  models1_LeNet.forward_fc2(models1_LeNet.get_feat_bf(inputsj[labelsj==k])), self.sigma_list)
                loss += self.lamL2 * torch.sum(models1_LeNet.fc2.weight ** 2) + \
                        self.lamL1 * torch.sum(torch.abs(models1_LeNet.fc2.weight))
                # Perform gradient descent
                loss.backward()
                opt.step()
                opt.zero_grad()
                running_loss += loss.item()
            losses1[epoch] = running_loss

        self.models1 = models1_LeNet

        # fix grads now
        for param in models1_LeNet.fc2.parameters():
            param.requires_grad = False

        # Step 2: remove the invariant part on all source envs, so that everything is independent of Y
        # get that coefficient b
        with torch.no_grad():
            meanYY = 0
            meanYX = 0
            for m in source:
                sizeofdatam = dataloaders[m].dataset.targets_mod.shape[0]
                Ym_onehot = torch.zeros((sizeofdatam, num_classes), dtype=torch.float)
                Ym_onehot = Ym_onehot.scatter_(1, dataloaders[m].dataset.targets_mod.reshape(-1, 1), 1)
                Ym_onehot_sum = Ym_onehot.sum(0).reshape(1, -1).to(device)

                for i, data in enumerate(dataloaders[m]):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    labels_onehot = torch.zeros((labels.shape[0], num_classes), dtype=torch.float).to(device)
                    labels_onehot = labels_onehot.scatter_(1, labels.reshape(-1, 1), 1)
                    labels_onehot_normalized = (labels_onehot/Ym_onehot_sum).to(device)
                    labels_onehot_demean = labels_onehot_normalized - \
                labels_onehot_normalized.mm((torch.ones(num_classes, num_classes)/num_classes).to(device))

                    outputs_bf_fc2 = models1_LeNet.get_feat_bf(inputs)
                    outputs_at_fc2 = torch.nn.functional.softmax(models1_LeNet.forward_fc2(outputs_bf_fc2), 1)
                    outputs_at_fc2_normalized = outputs_at_fc2[:, 0:(num_classes-1)] -  outputs_at_fc2[:, (num_classes-1):num_classes]

                    meanYYlocal = torch.mm(labels_onehot_demean[:, 0:(num_classes-1)].t(), outputs_at_fc2_normalized)
                    meanYXlocal = torch.mm(labels_onehot_demean[:, 0:(num_classes-1)].t(), outputs_bf_fc2)
                    meanYY += meanYYlocal/sizeofdatam
                    meanYX += meanYXlocal/sizeofdatam

            b, _ = torch.solve(meanYX, meanYY)
            print(b.shape)
            self.b = b

        # Step 3: mean match between source and target on the residual, after  transforming the covariates X - (X * beta_invariant) * b_invariant

        models = {}
        diffs = {}
        losses_all = {}
        self.total_weight = 0

        for m in source:
            models[m] = create_fixed_feat_LeNet().to(device)
            loss_fn = F.nll_loss
            opt = optim.SGD(models[m].fc2.parameters(), lr=self.lr, momentum=0.9)

            losses_all[m] = np.zeros(self.epochs)

            for epoch in range(self.epochs): # loop over the dataset multiple times
                running_loss = 0.0

                for i, data in enumerate(zip(dataloaders[m], dataloaders[target])):
                    opt.zero_grad()
                    loss = 0
                    inputs, labels = data[0][0].to(device), data[0][1].to(device)
                    outputs_bf_fc2 = models1_LeNet.get_feat_bf(inputs)
                    outputs_at_fc2 = torch.nn.functional.softmax(models1_LeNet.forward_fc2(outputs_bf_fc2), 1)
                    outputs_at_fc2_normalized = outputs_at_fc2[:, 0:(num_classes-1)] - outputs_at_fc2[:, (num_classes-1):num_classes]
                    inputs_bf_fc2_mod = outputs_bf_fc2 - torch.mm(outputs_at_fc2_normalized, b)
                    inputstar = data[1][0].to(device)
                    outputstar_bf_fc2 = models1_LeNet.get_feat_bf(inputstar)
                    outputstar_at_fc2 = torch.nn.functional.softmax(models1_LeNet.forward_fc2(outputstar_bf_fc2), 1)
                    outputstar_at_fc2_normalized = outputstar_at_fc2[:, 0:(num_classes-1)] - outputstar_at_fc2[:, (num_classes-1):num_classes]
                    inputstar_bf_fc2_mod = outputstar_bf_fc2 - torch.mm(outputstar_at_fc2_normalized, b)


                    loss += loss_fn(models[m](inputs), labels)
                    loss += self.lamMatch * mmd.mix_rbf_mmd2(models[m].forward_fc2(inputs_bf_fc2_mod),
                                                             models[m].forward_fc2(inputstar_bf_fc2_mod),
                                                             self.sigma_list)
                    loss += self.lamL2 * torch.sum(models[m].fc2.weight ** 2) + \
                        self.lamL1 * torch.sum(torch.abs(models[m].fc2.weight))

                    loss.backward()
                    opt.step()

                    running_loss += loss.item()

                losses_all[m][epoch] = running_loss

            # need to calculate the diffs
            diffs[m] = 0
            with torch.no_grad():
                for i, data in enumerate(zip(dataloaders[m], dataloaders[target])):
                    inputs, labels = data[0][0].to(device), data[0][1].to(device)
                    outputs_bf_fc2 = models1_LeNet.get_feat_bf(inputs)
                    outputs_at_fc2 = torch.nn.functional.softmax(models1_LeNet.forward_fc2(outputs_bf_fc2), 1)
                    outputs_at_fc2_normalized = outputs_at_fc2[:, 0:(num_classes-1)] - outputs_at_fc2[:, (num_classes-1):num_classes]
                    inputs_bf_fc2_mod = outputs_bf_fc2 - torch.mm(outputs_at_fc2_normalized, b)
                    inputstar = data[1][0].to(device)
                    outputstar_bf_fc2 = models1_LeNet.get_feat_bf(inputstar)
                    outputstar_at_fc2 = torch.nn.functional.softmax(models1_LeNet.forward_fc2(outputstar_bf_fc2), 1)
                    outputstar_at_fc2_normalized = outputstar_at_fc2[:, 0:(num_classes-1)] - outputstar_at_fc2[:, (num_classes-1):num_classes]
                    inputstar_bf_fc2_mod = outputstar_bf_fc2 - torch.mm(outputstar_at_fc2_normalized, b)

                    local_match_res_mmd = mmd.mix_rbf_mmd2(models[m].forward_fc2(inputs_bf_fc2_mod),
                                                 models[m].forward_fc2(inputstar_bf_fc2_mod),
                                                 self.sigma_list)
                    diffs[m] += local_match_res_mmd / self.epochs / (len(dataloaders[m].dataset)/dataloaders[m].batch_size)
            self.total_weight += torch.exp(-1000.*diffs[m])

        self.models = models
        self.diffs = diffs
        # print min diff
        minDiff = diffs[source[0]]
        minDiffIndx = source[0]
        for m in source:
            if diffs[m] < minDiff:
                minDiff = diffs[m]
                minDiffIndx = m
        print(minDiffIndx)
        self.minDiffIndx = minDiffIndx
        self.losses = losses_all[minDiffIndx]

        return self

    def predictSingle(self, X):
        output = self.models[m](X)
        _, ypredX = torch.max(output.data, 1)
        return ypredX

    def predict(self, X):
        output = 0
        for m in self.source:
            output += torch.exp(-1000.*self.diffs[m]) * self.models[m](X)
        output /= self.total_weight
        _, ypredX = torch.max(output.data, 1)
        return ypredX


    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)
