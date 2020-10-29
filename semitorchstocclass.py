"""
``semitorchstocclass`` provides classes implementing various domain adaptation methods using torch and stochastic gradient method.
All domain adaptation methods have to be subclass of BaseEstimator.
This implementation takes advantage of gradient method to optimize covariance match or MMD match in addition to mean match.
"""

from abc import abstractmethod
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import mmd

# check gpu avail
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# simple linear model in torch
class LinearModel(nn.Module):
    def __init__(self, d):
        super(LinearModel, self).__init__()
        self.lin1 = nn.Linear(d, 1, bias=True)

    def forward(self, x):
        x = self.lin1(x)
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


class Tar(BaseEstimator):
    """Oracle Linear regression (with l1 or l2 penalty) trained on the target domain"""
    def __init__(self, lamL2=0.0, lamL1=0.0, lr=1e-4, epochs=10):
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        # get the input dimension
        # assume it is a TensorDataset
        d = dataloaders[target].dataset[0][0].shape[0]
        model = LinearModel(d).to(device)
        with torch.no_grad():
            model.lin1.bias.data = torch.zeros_like(model.lin1.bias)
        torch.nn.init.xavier_normal_(model.lin1.weight, gain=0.01)
        # Define loss function
        loss_fn = F.mse_loss
        opt = optim.Adam(model.parameters(), lr=self.lr)

        self.losses = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloaders[target]):
                xtar, ytar = data[0].to(device), data[1].to(device)
                opt.zero_grad()
                loss = loss_fn(model(xtar).view(-1), ytar) + \
                    self.lamL2 * torch.sum(model.lin1.weight ** 2) + \
                    self.lamL1 * torch.sum(torch.abs(model.lin1.weight))
                # Perform gradient descent
                loss.backward()
                opt.step()
                running_loss += loss.item()

            self.losses[epoch] = running_loss

        self.model = model

        return self

    def predict(self, X):
        ypredX = self.model(X)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)



class Src(BaseEstimator):
    """Src Linear regression (with l1 or l2 penalty) trained on the source domain"""
    def __init__(self, lamL2=0.0, lamL1=0.0, sourceInd = 0, lr=1e-4, epochs=10):
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.sourceInd = sourceInd
        self.lr = lr
        self.epochs = epochs

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        # get the input dimension
        # assume it is a TensorDataset
        d = dataloaders[target].dataset[0][0].shape[0]
        model = LinearModel(d).to(device)
        with torch.no_grad():
            model.lin1.bias.data = torch.zeros_like(model.lin1.bias)
        torch.nn.init.xavier_normal_(model.lin1.weight, gain=0.01)
        # Define loss function
        loss_fn = F.mse_loss
        opt = optim.Adam(model.parameters(), lr=self.lr)

        self.losses = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloaders[source[self.sourceInd]]):
                x, y = data[0].to(device), data[1].to(device)
                opt.zero_grad()
                loss = loss_fn(model(x).view(-1), y) + \
                    self.lamL2 * torch.sum(model.lin1.weight ** 2) + \
                    self.lamL1 * torch.sum(torch.abs(model.lin1.weight))
                # Perform gradient descent
                loss.backward()
                opt.step()
                running_loss += loss.item()

            self.losses[epoch] = running_loss

        self.model = model

        return self

    def predict(self, X):
        ypredX = self.model(X)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)



class SrcPool(BaseEstimator):
    """Pool all source data together and then run linear regression
       with l1 or l2 penalty """
    def __init__(self, lamL2=0.0, lamL1=0.0, lr=1e-4, epochs=10):
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        # get the input dimension
        # assume it is a TensorDataset
        d = dataloaders[target].dataset[0][0].shape[0]
        model = LinearModel(d).to(device)
        # custom initialization
        with torch.no_grad():
            model.lin1.bias.data = torch.zeros_like(model.lin1.bias)
        torch.nn.init.xavier_normal_(model.lin1.weight, gain=0.01)
        # Define loss function
        loss_fn = F.mse_loss
        opt = optim.Adam(model.parameters(), lr=self.lr)

        self.losses = np.zeros(self.epochs)

        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(zip(*[dataloaders[m] for m in source])):
                opt.zero_grad()
                loss = 0
                for mindex, m in enumerate(source):
                    x, y = data[mindex][0].to(device), data[mindex][1].to(device)
                    loss += loss_fn(model(x).view(-1), y) / len(source)

                loss += self.lamL2 * torch.sum(model.lin1.weight ** 2)
                loss += self.lamL1 * torch.sum(torch.abs(model.lin1.weight))

                loss.backward()
                opt.step()

                running_loss += loss.item()

            self.losses[epoch] = running_loss
        self.model = model

        return self

    def predict(self, X):
        ypredX = self.model(X)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


class DIP(BaseEstimator):
    """Pick one source, match mean of X * beta between source and target"""
    def __init__(self, lamMatch=10., lamL2=0., lamL1=0., sourceInd = 0, lr=1e-4, epochs=10,
                 wayMatch='mean', sigma_list=[0.1, 1, 10, 100]):
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

        d = dataloaders[target].dataset[0][0].shape[0]
        model = LinearModel(d).to(device)
        # custom initialization
        with torch.no_grad():
            model.lin1.bias.data = torch.zeros_like(model.lin1.bias)
        torch.nn.init.xavier_normal_(model.lin1.weight, gain=0.01)
        # Define loss function
        loss_fn = F.mse_loss
        opt = optim.Adam(model.parameters(), lr=self.lr)
        self.losses = np.zeros(self.epochs)

        for epoch in range(self.epochs):
            running_loss = 0.0

            for i, data in enumerate(zip(dataloaders[source[self.sourceInd]], dataloaders[target])):
                opt.zero_grad()
                loss = 0
                x, y = data[0][0].to(device), data[0][1].to(device)
                xtar = data[1][0].to(device)
                loss += loss_fn(model(x).view(-1), y)
                if self.wayMatch == 'mean':
                    discrepancy = torch.nn.MSELoss()
                    loss += self.lamMatch * discrepancy(model(x), model(xtar))
                elif self.wayMatch == 'mmd':
                    loss += self.lamMatch * mmd.mix_rbf_mmd2(model(x), model(xtar), self.sigma_list)
                else:
                    print('error discrepancy')
                loss += self.lamL2 * torch.sum(model.lin1.weight ** 2) + \
                    self.lamL1 * torch.sum(torch.abs(model.lin1.weight))


                loss.backward()
                opt.step()

                running_loss += loss.item()

            self.losses[epoch] = running_loss
        self.model = model

        return self

    def predict(self, X):
        ypredX = self.model(X)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + self.wayMatch + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


class DIPweigh(BaseEstimator):
    '''loop throught all source envs, match the mean of X * beta between source env i and target, weigh the final prediction based loss of env i'''
    def __init__(self, lamMatch=10., lamL2=0., lamL1=0., lr=1e-4,
        epochs=10, wayMatch='mean', sigma_list=[0.1, 1, 10, 100]):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatch
        self.sigma_list = sigma_list

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        d = dataloaders[target].dataset[0][0].shape[0]
        models = {}
        diffs = {}
        ypreds = {}
        losses_all = {}
        self.total_weight = 0
        for m in source:
            model = LinearModel(d).to(device)
            models[m] = model
            # custom initialization
            with torch.no_grad():
                model.lin1.bias.data = torch.zeros_like(model.lin1.bias)
            torch.nn.init.xavier_normal_(model.lin1.weight, gain=0.01)
            # Define loss function
            loss_fn = F.mse_loss
            opt = optim.Adam(model.parameters(), lr=self.lr)

            losses_all[m] = np.zeros(self.epochs)

            for epoch in range(self.epochs):
                running_loss = 0.0

                for i, data in enumerate(zip(dataloaders[m], dataloaders[target])):
                    opt.zero_grad()
                    loss = 0
                    x, y = data[0][0].to(device), data[0][1].to(device)
                    xtar = data[1][0].to(device)
                    loss += loss_fn(model(x).view(-1), y)
                    if self.wayMatch == 'mean':
                        discrepancy = torch.nn.MSELoss()
                        loss += self.lamMatch * discrepancy(model(x), model(xtar))
                    elif self.wayMatch == 'mmd':
                        loss += self.lamMatch * mmd.mix_rbf_mmd2(model(x), model(xtar), self.sigma_list)
                    else:
                        raise('error discrepancy')
                    loss += self.lamL2 * torch.sum(model.lin1.weight ** 2) + \
                        self.lamL1 * torch.sum(torch.abs(model.lin1.weight))


                    loss.backward()
                    opt.step()

                    running_loss += loss.item()


                losses_all[m][epoch] = running_loss

            # need to calculate the diffs
            diffs[m] = 0
            with torch.no_grad():
                for i, data in enumerate(zip(dataloaders[m], dataloaders[target])):
                    x, y = data[0][0].to(device), data[0][1].to(device)
                    xtar = data[1][0].to(device)
                    if self.wayMatch == 'mean':
                        discrepancy = torch.nn.MSELoss()
                        local_match_res = discrepancy(model(x), model(xtar))
                    elif self.wayMatch == 'mmd':
                        local_match_res = mmd.mix_rbf_mmd2(model(x), model(xtar), self.sigma_list)
                    else:
                        raise('error discrepancy')
                    diffs[m] += local_match_res / self.epochs / (len(dataloaders[m].dataset)/dataloaders[m].batch_size)
            self.total_weight += torch.exp(-100.*diffs[m])

        self.models = models
        self.diffs = diffs

        minDiff = diffs[source[0]]
        minDiffIndx = source[0]
        for m in source:
            if diffs[m] < minDiff:
                minDiff = diffs[m]
                minDiffIndx = m
        self.minDiffIndx = minDiffIndx
        print(minDiffIndx)
        self.losses = losses_all[minDiffIndx]

        return self

    def predict(self, X):
        ypredX1 = 0
        for m in self.source:
            ypredX1 += torch.exp(-100.*self.diffs[m]) * self.models[m](X)
        ypredX1 /= self.total_weight
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + self.wayMatch + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)




class CIP(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, no target env is needed"""
    def __init__(self, lamCIP=10., lamL2=0., lamL1=0., lr=1e-4, epochs=10,
                 wayMatch='mean', sigma_list = [0.1, 1, 10, 100]):
        self.lamCIP = lamCIP
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatch
        self.sigma_list = sigma_list

    def fit(self, dataloaders, source, target):
        super().fit(dataloaders, source, target)

        d = dataloaders[target].dataset[0][0].shape[0]
        model = LinearModel(d).to(device)
        # custom initialization
        with torch.no_grad():
            model.lin1.bias.data = torch.zeros_like(model.lin1.bias)
        torch.nn.init.xavier_normal_(model.lin1.weight, gain=0.01)

        # Define loss function
        loss_fn = F.mse_loss
        opt = optim.Adam(model.parameters(), lr=self.lr)

        self.losses = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(zip(*[dataloaders[m] for m in source])):
                opt.zero_grad()
                loss = 0
                for mindex, m in enumerate(source):
                    x, y = data[mindex][0].to(device), data[mindex][1].to(device)
                    loss += loss_fn(model(x).view(-1), y)/float(len(source))
                    xmod = x - torch.mm(y.view(-1, 1), torch.mm(y.view(1, -1), x))/torch.sum(y**2)

                    # conditional invariance penalty
                    for jindex, j in enumerate(source):
                        if j > m:
                            xj, yj = data[jindex][0].to(device), data[jindex][1].to(device)
                            xmodj = xj - torch.mm(yj.view(-1, 1), torch.mm(yj.view(1, -1), xj))/torch.sum(yj**2)
                            if self.wayMatch == 'mean':
                                discrepancy = torch.nn.MSELoss()
                                loss += self.lamCIP/float(len(source)**2) * discrepancy(model(xmod), model(xmodj))
                            elif self.wayMatch == 'mmd':
                                loss += self.lamCIP/float(len(source)**2) * \
                                  mmd.mix_rbf_mmd2(model(xmod), model(xmodj), self.sigma_list)
                            else:
                                raise('error discrepancy')

                loss += self.lamL2 * torch.sum(model.lin1.weight ** 2) + \
                        self.lamL1 * torch.sum(torch.abs(model.lin1.weight))
                # Perform gradient descent
                loss.backward()
                opt.step()
                running_loss += loss.item()

            self.losses[epoch] = running_loss
        self.model = model

        return self

    def predict(self, X):
        ypredX = self.model(X)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + self.wayMatch + "_CIP{:.1f}".format(self.lamCIP) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)



class CIRMweigh(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, use Yhat as proxy of Y to remove the Y parts in X.
    Match on the residual between one source env and target env"""
    def __init__(self, lamMatch=10., lamCIP=10., lamL2=0., lamL1=0., lr=1e-4, epochs=10,
                 wayMatch='mean', sigma_list=[0.1, 1, 10, 100]):
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

        d = dataloaders[target].dataset[0][0].shape[0]
        # Step 1: use source envs to match the conditional mean
        # find beta_invariant
        models1 = LinearModel(d).to(device)
        # custom initialization
        with torch.no_grad():
            models1.lin1.bias.data = torch.zeros_like(models1.lin1.bias)
        torch.nn.init.xavier_normal_(models1.lin1.weight, gain=0.01)

        # Define loss function
        loss_fn = F.mse_loss
        opt = optim.Adam(models1.parameters(), lr=self.lr)

        losses1 = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(zip(*[dataloaders[m] for m in source])):
                loss = 0
                for mindex, m in enumerate(source):
                    x, y = data[mindex][0].to(device), data[mindex][1].to(device)
                    loss += loss_fn(models1(x).view(-1), y)/float(len(source))
                    xmod = x - torch.mm(y.view(-1, 1), torch.mm(y.view(1, -1), x))/torch.sum(y**2)

                    # conditional invariance penalty
                    for jindex, j in enumerate(source):
                        xj, yj = data[jindex][0].to(device), data[jindex][1].to(device)
                        if j > m:
                            xmodj = xj - torch.mm(yj.view(-1, 1), torch.mm(yj.view(1, -1), xj))/torch.sum(yj**2)
                            if self.wayMatch == 'mean':
                                discrepancy = torch.nn.MSELoss()
                                loss += self.lamCIP/float(len(source)**2) * discrepancy(models1(xmod), models1(xmodj))
                            elif self.wayMatch == 'mmd':
                                loss += self.lamCIP/float(len(source)**2) * \
                                  mmd.mix_rbf_mmd2(models1(xmod), models1(xmodj), self.sigma_list)
                            else:
                                raise('error discrepancy')
                loss += self.lamL2 * torch.sum(models1.lin1.weight ** 2) + \
                        self.lamL1 * torch.sum(torch.abs(models1.lin1.weight))
                # Perform gradient descent
                loss.backward()
                opt.step()
                opt.zero_grad()
                running_loss += loss.item()
            losses1[epoch] = running_loss

        self.models1 = models1

        # fix grads now
        for param in models1.lin1.parameters():
            param.requires_grad = False

        # Step 2: remove the invariant part on all source envs, so that everything is independent of Y
        # get that coefficient b
        YsrcMean = 0
        ntotal = 0
        for m in source:
            YsrcMean += torch.sum(dataloaders[m].dataset.tensors[1])
            ntotal += dataloaders[m].dataset.tensors[1].shape[0]
        YsrcMean /= ntotal

        YTX = 0
        YTY = 0
        for m in source:
            for i, data in enumerate(dataloaders[m]):
                x, y = data[0].to(device), data[1].to(device)
                yguess = self.models1(x)
                yCentered = y - YsrcMean
                YTY += torch.sum(yguess.t() * yCentered)
                YTX += torch.mm(yCentered.view(1, -1), x)

        b = YTX / YTY
        self.b = b


        # Step 3: mean match between source and target on the residual, after  transforming the covariates X - (X * beta_invariant) * b_invariant
        models = {}
        diffs = {}
        losses_all = {}
        self.total_weight = 0

        for m in source:
            models[m] = LinearModel(d).to(device)
            # custom initialization
            with torch.no_grad():
                models[m].lin1.bias.data = torch.zeros_like(models[m].lin1.bias)
            torch.nn.init.xavier_normal_(models[m].lin1.weight, gain=0.01)
            # Define loss function
            loss_fn = F.mse_loss
            opt = optim.Adam(models[m].parameters(), lr=self.lr)

            losses_all[m] = np.zeros(self.epochs)

            for epoch in range(self.epochs): # loop over the dataset multiple times
                running_loss = 0.0

                for i, data in enumerate(zip(dataloaders[m], dataloaders[target])):
                    opt.zero_grad()
                    loss = 0
                    x, y = data[0][0].to(device), data[0][1].to(device)
                    yguess = self.models1(x)
                    xmod = x - torch.mm(yguess, b)

                    xtar = data[1][0].to(device)
                    ytarguess = self.models1(xtar)
                    xtarmod = xtar - torch.mm(ytarguess, b)

                    loss += loss_fn(models[m](x).view(-1), y)
                    if self.wayMatch == 'mean':
                        discrepancy = torch.nn.MSELoss()
                        loss += self.lamMatch * discrepancy(models[m](xmod),
                                                            models[m](xtarmod))
                    elif self.wayMatch == 'mmd':
                        loss += self.lamMatch * mmd.mix_rbf_mmd2(models[m](xmod),
                                                                 models[m](xtarmod),
                                                                 self.sigma_list)
                    else:
                        raise('error discrepancy')
                    loss += self.lamL2 * torch.sum(models[m].lin1.weight ** 2) + \
                        self.lamL1 * torch.sum(torch.abs(models[m].lin1.weight))

                    loss.backward()
                    opt.step()

                    running_loss += loss.item()

                losses_all[m][epoch] = running_loss

            # need to compute diff after training


            diffs[m] = 0.
            with torch.no_grad():
                for i, data in enumerate(zip(dataloaders[m], dataloaders[target])):
                    x, y = data[0][0].to(device), data[0][1].to(device)
                    yguess = self.models1(x)
                    xmod = x - torch.mm(yguess, b)

                    xtar = data[1][0].to(device)
                    ytarguess = self.models1(xtar)
                    xtarmod = xtar - torch.mm(ytarguess, b)

                    if self.wayMatch == 'mean':
                        discrepancy = torch.nn.MSELoss()
                        diffs[m] += discrepancy(models[m](xmod), models[m](xtarmod)) / \
                                  self.epochs / (len(dataloaders[m].dataset)/dataloaders[m].batch_size)
                    elif self.wayMatch == 'mmd':
                        diffs[m] += mmd.mix_rbf_mmd2(models[m](xmod), models[m](xtarmod), self.sigma_list) / \
                                  self.epochs / (len(dataloaders[m].dataset)/dataloaders[m].batch_size)
                    else:
                        raise('error discrepancy')


            self.total_weight += torch.exp(-100.*diffs[m])

        # take the min diff loss to be current best losses and model
        minDiff = diffs[source[0]]
        minDiffIndx = source[0]
        self.losses = losses_all[source[0]]
        for m in source:
            if diffs[m] < minDiff:
                minDiff = diffs[m]
                minDiffIndx = m
                self.losses = losses_all[m]
                self.model = models[m]
        self.minDiffIndx = minDiffIndx

        self.models = models
        self.diffs = diffs

        return self

    def predict(self, X):
        ypredX1 = 0
        for m in self.source:
            ypredX1 += torch.exp(-100.*self.diffs[m]) * self.models[m](X)
        ypredX1 /= self.total_weight
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + self.wayMatch + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)
