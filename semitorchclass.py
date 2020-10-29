"""
``semitorchclass`` provides classes implementing various domain adaptation methods using torch and gradient method.
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

class ZeroBeta(BaseEstimator):
    """Estimator that sets beta to zero"""

    def fit(self, data, source, target):
        super().fit(data, source, target)

        d = data[target][0].shape[1]
        model = LinearModel(d).to(device)
        with torch.no_grad():
            model.lin1.weight.data = torch.zeros_like(model.lin1.weight)
            model.lin1.bias.data = torch.zeros_like(model.lin1.bias)

        self.model = model

        xtar, _ = data[target]
        self.ypred = self.model(xtar)

        return self

    def predict(self, X):
        ypredX = self.model(X)
        return ypredX

class Tar(BaseEstimator):
    """Oracle Linear regression (with l1 or l2 penalty) trained on the target domain"""
    def __init__(self, lamL2=0.0, lamL1=0.0, lr=1e-4, epochs=10):
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs

    def fit(self, data, source, target):
        super().fit(data, source, target)

        d = data[target][0].shape[1]
        model = LinearModel(d).to(device)
        with torch.no_grad():
            model.lin1.bias.data = torch.zeros_like(model.lin1.bias)
        # torch.nn.init.kaiming_normal_(model.lin1.weight, mode='fan_in')
        torch.nn.init.xavier_normal_(model.lin1.weight, gain=0.01)
        # Define loss function
        loss_fn = F.mse_loss
        opt = optim.Adam(model.parameters(), lr=self.lr)
        # opt = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        self.losses = np.zeros(self.epochs)
        xtar, ytar = data[target]
        # oracle estimator uses target labels
        for epoch in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(model(xtar), ytar.view(-1, 1)) + \
                self.lamL2 * torch.sum(model.lin1.weight ** 2) + \
                self.lamL1 * torch.sum(torch.abs(model.lin1.weight))
            # Perform gradient descent
            loss.backward()
            opt.step()
            self.losses[epoch] = loss.item()

        self.model = model

        self.ypred = self.model(xtar)

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

    def fit(self, data, source, target):
        super().fit(data, source, target)

        d = data[target][0].shape[1]
        model = LinearModel(d).to(device)
        # custom initialization
        with torch.no_grad():
            model.lin1.bias.data = torch.zeros_like(model.lin1.bias)
        torch.nn.init.xavier_normal_(model.lin1.weight, gain=0.01)
        # Define loss function
        loss_fn = F.mse_loss
        opt = optim.Adam(model.parameters(), lr=self.lr)
        # opt = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        self.losses = np.zeros(self.epochs)

        for epoch in range(self.epochs):
            loss = 0
            opt.zero_grad()
            for m in source:
                x, y = data[m]
                loss += loss_fn(model(x), y.view(-1, 1))/len(source)

            loss += self.lamL2 * torch.sum(model.lin1.weight ** 2)
            loss += self.lamL1 * torch.sum(torch.abs(model.lin1.weight))
            # Perform gradient descent
            loss.backward()
            opt.step()
            self.losses[epoch] = loss.item()
        self.model = model

        xtar, _ = data[target]
        self.ypred = self.model(xtar)

        return self

    def predict(self, X):
        ypredX = self.model(X)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)

def wayMatchSelector(wayMatch='mean'):
    if wayMatch == 'mean':
        return [lambda x: torch.mean(x, dim=0)]
    elif wayMatch == 'std':
        return [lambda x: torch.std(x, dim=0)]
    elif wayMatch == '25p':
        return [lambda x: torch.kthvalue(x, (1 + round(.25 * (x.shape[0] - 1))), dim=0)]
    elif wayMatch == '75p':
        return [lambda x: torch.kthvalue(x, (1 + round(.75 * (x.shape[0] - 1))), dim=0)]
    elif wayMatch == 'mean+std':
        return [lambda x: torch.mean(x, dim=0), lambda x: torch.std(x, dim=0)]
    elif wayMatch == 'mean+std+25p':
        return [lambda x: torch.mean(x, dim=0), lambda x: torch.std(x, dim=0), lambda x: torch.kthvalue(x, (1 + round(.25 * (x.shape[0] - 1))), dim=0)[0]]
    elif wayMatch == 'mean+std+25p+75p':
        return [lambda x: torch.mean(x, dim=0), lambda x: torch.std(x, dim=0), lambda x: torch.kthvalue(x, (1 + round(.25 * (x.shape[0] - 1))), dim=0)[0],
            lambda x: torch.kthvalue(x, (1 + round(.75 * (x.shape[0] - 1))), dim=0)[0]]
    else:
        print("Error: wayMatch not specified correctly, using mean")
        return [lambda x: torch.mean(x, 0)]



class DIP(BaseEstimator):
    """Pick one source, match mean of X * beta between source and target"""
    def __init__(self, lamMatch=10., lamL2=0., lamL1=0., sourceInd = 0, lr=1e-4, epochs=10, wayMatch='mean'):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.sourceInd = sourceInd
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatchSelector(wayMatch)

    def fit(self, data, source, target):
        super().fit(data, source, target)

        d = data[target][0].shape[1]
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
            x, y = data[source[self.sourceInd]]
            xtar, ytar = data[target]
            opt.zero_grad()
            loss = loss_fn(model(x), y.view(-1, 1)) + \
                self.lamL2 * torch.sum(model.lin1.weight ** 2) + \
                self.lamL1 * torch.sum(torch.abs(model.lin1.weight))
            for wayMatchLocal in self.wayMatch:
                loss += self.lamMatch * loss_fn(wayMatchLocal(model(x)), wayMatchLocal(model(xtar)))

            # Perform gradient descent
            loss.backward()
            opt.step()

            self.losses[epoch] = loss.item()
        self.model = model

        xtar, _ = data[target]
        self.ypred = self.model(xtar)

        return self

    def predict(self, X):
        ypredX = self.model(X)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


class DIPOracle(BaseEstimator):
    """Pick one source, match mean of X * beta between source and target, use target labels to fit (oracle)"""
    def __init__(self, lamMatch=10., lamL2=0., lamL1=0., sourceInd = 0, lr=1e-4, epochs=10, wayMatch='mean'):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.sourceInd = sourceInd
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatchSelector(wayMatch)

    def fit(self, data, source, target):
        super().fit(data, source, target)

        d = data[target][0].shape[1]
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
            x, y = data[source[self.sourceInd]]
            xtar, ytar = data[target]
            opt.zero_grad()
            loss = loss_fn(model(xtar), ytar.view(-1, 1)) + \
                self.lamL2 * torch.sum(model.lin1.weight ** 2) + \
                self.lamL1 * torch.sum(torch.abs(model.lin1.weight))

            for wayMatchLocal in self.wayMatch:
                loss += self.lamMatch * loss_fn(wayMatchLocal(model(x)), wayMatchLocal(model(xtar)))

            # Perform gradient descent
            loss.backward()
            opt.step()

            self.losses[epoch] = loss.item()
        self.model = model

        xtar, _ = data[target]
        self.ypred = self.model(xtar)

        return self

    def predict(self, X):
        ypredX = self.model(X)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


class DIPweigh(BaseEstimator):
    '''loop throught all source envs, match the mean of X * beta between source env i and target, weigh the final prediction based loss of env i'''
    def __init__(self, lamMatch=10., lamL2=0., lamL1=0., lr=1e-4,
        epochs=10, wayMatch='mean'):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatchSelector(wayMatch)

    def fit(self, data, source, target):
        super().fit(data, source, target)

        d = data[target][0].shape[1]
        models = {}
        diffs = {}
        ypreds = {}
        losses_all = {}
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
                x, y = data[m]
                xtar, ytar = data[target]
                opt.zero_grad()

                loss = loss_fn(model(x), y.view(-1, 1)) + \
                    self.lamL2* torch.sum(model.lin1.weight ** 2) + \
                    self.lamL1 * torch.sum(torch.abs(model.lin1.weight))

                for wayMatchLocal in self.wayMatch:
                    loss += self.lamMatch * loss_fn(wayMatchLocal(model(x)), wayMatchLocal(model(xtar)))
                # Perform gradient descent
                loss.backward()
                opt.step()

                losses_all[m][epoch] = loss.item()

            diffs[m] =0.
            for wayMatchLocal in self.wayMatch:
                diffs[m] += loss_fn(wayMatchLocal(model(x)), wayMatchLocal(model(xtar)))
            ypreds[m] = models[m](xtar)

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
        self.total_weight = 0
        self.ypred = 0
        for m in self.source:
            self.ypred += torch.exp(-100.*diffs[m]) * ypreds[m]
            self.total_weight += torch.exp(-100.*diffs[m])
        self.ypred /= self.total_weight
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
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


class CIP(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, no target env is needed"""
    def __init__(self, lamCIP=10., lamL2=0., lamL1=0., lr=1e-4, epochs=10, wayMatch='mean'):
        self.lamCIP = lamCIP
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatchSelector(wayMatch)

    def fit(self, data, source, target):
        super().fit(data, source, target)

        d = data[target][0].shape[1]
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
            loss = 0
            avgmodelxList = [0.] * len(self.wayMatch)
            opt.zero_grad()
            for m in source:
                x, y = data[m]
                # do the conditional on y
                xmod = x - torch.mm(y.view(-1, 1), torch.mm(y.view(1, -1), x))/torch.sum(y**2)
                for i, wayMatchLocal in enumerate(self.wayMatch):
                    avgmodelxList[i] += wayMatchLocal(model(xmod))/len(source)
            for m in source:
                x, y = data[m]
                xmod = x - torch.mm(y.view(-1, 1), torch.mm(y.view(1, -1), x))/torch.sum(y**2)
                loss += loss_fn(model(x), y.view(-1, 1))/len(source)
                for i, wayMatchLocal in enumerate(self.wayMatch):
                    loss += self.lamCIP * loss_fn(avgmodelxList[i], wayMatchLocal(model(xmod)))/len(source)
            loss += self.lamL2 * torch.sum(model.lin1.weight ** 2)
            loss += self.lamL1 * torch.sum(torch.abs(model.lin1.weight))
            # Perform gradient descent
            loss.backward()
            opt.step()

            self.losses[epoch] = loss.item()
        self.model = model

        xtar, _ = data[target]
        self.ypred = self.model(xtar)

        return self

    def predict(self, X):
        ypredX = self.model(X)
        return ypredX

    def __str__(self):
        return self.__class__.__name__ + "_CIP{:.1f}".format(self.lamCIP) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)


class CIRMweigh(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, use Yhat as proxy of Y to remove the Y parts in X.
    Match on the residual between one source env and target env"""
    def __init__(self, lamMatch=10., lamL2=0., lamL1=0., lr=1e-4, epochs=10, wayMatch='mean'):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.lamL1 = lamL1
        self.lr = lr
        self.epochs = epochs
        self.wayMatch = wayMatchSelector(wayMatch)

    def fit(self, data, source, target):
        super().fit(data, source, target)

        d = data[target][0].shape[1]
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
            loss = 0
            avgmodelxList = [0.] * len(self.wayMatch)
            opt.zero_grad()
            for m in source:
                x, y = data[m]
                # do the conditional on y
                xmod = x - torch.mm(y.view(-1, 1), torch.mm(y.view(1, -1), x))/torch.sum(y**2)
                for i, wayMatchLocal in enumerate(self.wayMatch):
                    avgmodelxList[i] += wayMatchLocal(models1(xmod))/len(source)
            for m in source:
                x, y = data[m]
                xmod = x - torch.mm(y.view(-1, 1), torch.mm(y.view(1, -1), x))/torch.sum(y**2)
                loss += loss_fn(models1(x), y.view(-1, 1))/len(source)
                for i, wayMatchLocal in enumerate(self.wayMatch):
                    loss += self.lamMatch * loss_fn(avgmodelxList[i], wayMatchLocal(models1(xmod)))/len(source)
            loss += self.lamL2 * torch.sum(models1.lin1.weight ** 2)
            loss += self.lamL1 * torch.sum(torch.abs(models1.lin1.weight))
            # Perform gradient descent
            loss.backward()
            opt.step()
            losses1[epoch] = loss.item()

        self.models1 = models1

        # fix grads now
        for param in models1.lin1.parameters():
            param.requires_grad = False

        # Step 2: remove the invariant part on all source envs, so that everything is independent of Y
        # get that coefficient b
        YsrcMean = 0
        ntotal = 0
        for m in source:
            YsrcMean += torch.sum(data[m][1])
            ntotal += data[m][1].shape[0]
        YsrcMean /= ntotal

        YTX = 0
        YTY = 0
        for m in source:
            x, y = data[m]
            yguess = self.models1(x)
            yCentered = y - YsrcMean
            YTY += torch.sum(yguess.t() * yCentered)
            YTX += torch.mm(yCentered.view(1, -1), x)

        b = YTX / YTY
        self.b = b


        # Step 3: mean match between source and target on the residual, after  transforming the covariates X - (X * beta_invariant) * b_invariant
        models = {}
        diffs = {}
        ypreds = {}
        losses_all = {}
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
            x, y = data[m]
            xmod = x - torch.mm(self.models1(x), b)
            xtar, ytar = data[target]
            xtarmod = xtar - torch.mm(self.models1(xtar), b)

            for epoch in range(self.epochs):
                loss = loss_fn(models[m](x), y.view(-1, 1)) + \
                    self.lamL2 * torch.sum(models[m].lin1.weight ** 2) + \
                    self.lamL1 * torch.sum(torch.abs(models[m].lin1.weight))

                for wayMatchLocal in self.wayMatch:
                    loss += self.lamMatch * loss_fn(wayMatchLocal(models[m](xmod)), wayMatchLocal(models[m](xtarmod)))
                # Perform gradient descent
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses_all[m][epoch] = loss.item()

            diffs[m] = 0.
            for wayMatchLocal in self.wayMatch:
                diffs[m] += loss_fn(wayMatchLocal(models[m](xmod)), wayMatchLocal(models[m](xtarmod)))
            ypreds[m] = models[m](xtar)

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
        self.total_weight = 0
        self.ypred = 0
        for m in self.source:
            self.ypred += torch.exp(-100.*diffs[m]) * ypreds[m]
            self.total_weight += torch.exp(-100.*diffs[m])
        self.ypred /= self.total_weight
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
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_L2={:.1f}".format(self.lamL2) + "_L1={:.1f}".format(self.lamL1)



