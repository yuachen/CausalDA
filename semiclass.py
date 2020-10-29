"""
``semiclass`` provides classes implementing various domain adaptation methods.
All domain adaptation methods have to be subclass of BaseEstimator.
This implementation aims for clarity rather than efficiency (it is not fast enough) and scalability (it can't really deal with large dimension or large sample case).
For example, numpy built-in linear system solver is often used.
"""

from abc import abstractmethod
import numpy as np
import operator

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

        xtar, _ = data[target]
        # add a column of ones for intercept
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        self.beta = np.zeros(xtar1.shape[1])
        # set the predicted responses
        self.ypred = xtar1.dot(self.beta)

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

class Tar(BaseEstimator):
    """Oracle Ridge (or OLS) trained on the target domain"""
    def __init__(self, lamL2=0.0):
        self.lamL2 = lamL2

    def fit(self, data, source, target):
        super().fit(data, source, target)

        xtar, ytar = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        ntar = xtar.shape[0]
        A = np.eye(xtar1.shape[1])
        A[-1, -1] = 0
        beta = np.linalg.solve(xtar1.T.dot(xtar1)/ntar + self.lamL2*A, xtar1.T.dot(ytar)/ntar)
        self.beta = beta
        ypred = xtar1.dot(beta)
        self.ypred = ypred

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_Ridge{:.1f}".format(self.lamL2)

class Src(BaseEstimator):
    """Use one source env and then run Ridge (or OLS)"""
    def __init__(self, lamL2=0.0, sourceInd = 0):
        self.lamL2 = lamL2
        self.sourceInd = sourceInd

    def fit(self, data, source, target):
        super().fit(data, source, target)

        boolA = False
        x, y = data[source[self.sourceInd]]
        n = x.shape[0]
        x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        XX = x1.T.dot(x1)
        XY = x1.T.dot(y)

        if not boolA:
            A = np.eye(x1.shape[1])
            A[-1, -1] = 0
            boolA = True

        beta = np.linalg.solve(XX/n + self.lamL2*A, XY/n)
        self.beta = beta

        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        ypred = xtar1.dot(beta)
        self.ypred = ypred

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_Ridge{:.1f}".format(self.lamL2)


class SrcPool(BaseEstimator):
    """Pool all source data together and then run Ridge (or OLS)"""
    def __init__(self, lamL2=0.0):
        self.lamL2 = lamL2

    def fit(self, data, source, target):
        super().fit(data, source, target)

        XY = 0.
        XX = 0.
        ntotal = 0
        boolA = False
        for m in source:
            x, y = data[m]
            ntotal += x.shape[0]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            XX += x1.T.dot(x1)
            XY += x1.T.dot(y)

            if not boolA:
                A = np.eye(x1.shape[1])
                A[-1, -1] = 0
                boolA = True

        beta = np.linalg.solve(XX/ntotal + self.lamL2*A, XY/ntotal)
        self.beta = beta

        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        ypred = xtar1.dot(beta)
        self.ypred = ypred

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_Ridge{:.1f}".format(self.lamL2)

class DirectImpute(BaseEstimator):
    """Direct imputation of target XY using the fact that the intervention is uncorrelated with Y"""
    def __init__(self, lamL2=0.0, center=True):
        self.center = center
        self.lamL2 = lamL2

    def fit(self, data, source, target):
        super().fit(data, source, target)

        fakeXY = 0.
        Msource = len(source)
        for m in source:
            x, y = data[m]
            nm = x.shape[0]
            if self.center:
                y = y - np.mean(y)
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            fakeXY += x1.T.dot(y)/nm/Msource

        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        ntar = xtar.shape[0]
        A = np.eye(x1.shape[1])
        A[-1, -1] = 0
        beta = np.linalg.solve(xtar1.T.dot(xtar1)/ntar+self.lamL2*A, fakeXY)
        self.beta = beta

        ypred = xtar1.dot(beta)
        self.ypred = ypred

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_Ridge{:.1f}".format(self.lamL2)

class DIP(BaseEstimator):
    """Pick one source, DIP match mean of X * beta between source and target"""
    def __init__(self, lamMatch=10., lamL2=0., sourceInd = 0):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.sourceInd = sourceInd

    def fit(self, data, source, target):
        super().fit(data, source, target)

        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        x, y = data[source[self.sourceInd]]
        x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        n1 = x.shape[0]
        diffx1 = np.mean(xtar1, axis=0) - np.mean(x1, axis=0)
        XTX = x1.T.dot(x1)/n1 + self.lamMatch * np.outer(diffx1, diffx1)
        XTY = x1.T.dot(y)/n1

        A = np.eye(x1.shape[1])
        A[-1, -1] = 0

        beta = np.linalg.solve(XTX+self.lamL2*A, XTY)
        self.beta = beta

        ypred = xtar1.dot(beta)
        self.ypred = ypred

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_Ridge{:.1f}".format(self.lamL2)

class DIPmix(BaseEstimator):
    """Pick one source, DIP match mean of X * beta between source and target
    the version that deals with mixed-causal-anticausal case
    we first remove the causal part, do DIP and then add the causal part back
    This is an oracle estimator"""
    def __init__(self, causal_index=[0], lamMatch=10., lamL2=0., sourceInd = 0):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.sourceInd = sourceInd
        self.causal_index = causal_index

    def fit(self, data, source, target):
        super().fit(data, source, target)

        d = data[source[0]][0].shape[1]
        self.noncausal_index = list(set(np.arange(d)) - set(self.causal_index))

        def get_causal_beta(indexk):
            # for one covariate coordinate x or for y
            # indexk is the index of the covariate coordinate x
            XY = 0.
            XX = 0.
            ntotal = 0
            boolA = False
            betacausal1_restrict = 0
            for m in source:
                x, y = data[m]
                if indexk != -1:
                    y = x[:, indexk]
                # only use the causal part of x
                x = x[:, self.causal_index]
                x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
                ntotal += x.shape[0]
                XX += x1.T.dot(x1)
                XY += x1.T.dot(y)

                if not boolA:
                    A = np.eye(x1.shape[1])
                    A[-1, -1] = 0
                    boolA = True

                betacausal1_restrict += np.linalg.solve(XX/ntotal + self.lamL2*A, XY/ntotal)/len(source)
            return(betacausal1_restrict)

        beta_corrections = {}
        for indexk in self.noncausal_index:
            beta_corrections[indexk] = get_causal_beta(indexk)
        beta_corrections[-1] = get_causal_beta(-1)

        # create new dataset based by removing causal part
        betacausal1 = np.zeros(d)
        betacausal1[self.causal_index] = beta_corrections[-1][:-1]
        self.betacausal1 = betacausal1
        # for cirm, y - x * betacuasal1 will be used as a replacement for y
        # Now modify the dataset
        dataNew = {}
        for m in np.concatenate((source, [target])):
            x, y = data[m]
            xNew = np.zeros_like(x[:, self.noncausal_index])
            for k, indexk in enumerate(self.noncausal_index):
                xNew[:, k] = x[:, indexk] - x[:, self.causal_index].dot(beta_corrections[indexk][:-1])
            yNew = y - x.dot(self.betacausal1)
            dataNew[m] = xNew, yNew

        # do DIP on the new dataset
        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        xtarNew, _ = dataNew[target]
        xtarNew1 = np.concatenate((xtarNew, np.ones((xtarNew.shape[0], 1))), axis=1)
        x, y = data[source[self.sourceInd]]
        xNew, yNew = dataNew[source[self.sourceInd]]
        x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        xNew1 = np.concatenate((xNew, np.ones((xNew.shape[0], 1))), axis=1)
        n1 = xNew.shape[0]
        diffx1New = np.mean(xtarNew1, axis=0) - np.mean(xNew1, axis=0)
        diffx1 = np.zeros(d+1)
        diffx1[self.noncausal_index] = diffx1New[:-1]
        XTX = xNew1.T.dot(xNew1)/n1 + self.lamMatch * np.outer(diffx1New, diffx1New)
        XTY = xNew1.T.dot(yNew)/n1

        A = np.eye(xNew1.shape[1])
        A[-1, -1] = 0
        betaNew = np.linalg.solve(XTX+self.lamL2*A, XTY)
        self.beta = np.zeros(d+1)
        self.beta[self.noncausal_index] = betaNew[:-1]
        self.beta[-1] = betaNew[-1]
        self.beta[self.causal_index] = beta_corrections[-1][:-1]
        for k, indexk in enumerate(self.noncausal_index):
            self.beta[self.causal_index] -= betaNew[k]*beta_corrections[indexk][:-1]

        ypred = xtar1.dot(self.beta)
        self.ypred = ypred

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_Ridge{:.1f}".format(self.lamL2)

class DIPOracle(BaseEstimator):
    """Pick one source, DIP match mean of X * beta between source and target, use target labels to fit (oracle)"""
    def __init__(self, lamMatch=10., lamL2=0., sourceInd = 0):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.sourceInd = sourceInd

    def fit(self, data, source, target):
        super().fit(data, source, target)

        xtar, ytar = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        ntar = xtar.shape[0]
        x, y = data[source[self.sourceInd]]
        x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        n1 = x.shape[0]
        diffx1 = np.mean(xtar1, axis=0) - np.mean(x1, axis=0)
        XTX = xtar1.T.dot(xtar1)/ntar + self.lamMatch * np.outer(diffx1, diffx1)
        XTY = xtar1.T.dot(ytar)/ntar

        A = np.eye(x1.shape[1])
        A[-1, -1] = 0

        beta = np.linalg.solve(XTX+self.lamL2*A, XTY)
        self.beta = beta

        ypred = xtar1.dot(beta)
        self.ypred = ypred

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_Ridge{:.1f}".format(self.lamL2)

class DIPweigh(BaseEstimator):
    '''loop throught all source envs, match the mean of X * beta between source env i and target, weigh the final prediction based loss of env i'''

    def __init__(self, lamMatch=10.0, lamL2=0.0, weightrho=1000.):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.weightrho = weightrho

    def fit(self, data, source, target):
        super().fit(data, source, target)

        # mth position contains beta from mth source env
        self.betas = {}
        # mth position contains predicted response from mth source env
        ypreds = {}
        # source env selection criteria, src loss
        self.crits = {}
        # normalized version of the selection criteria, to avoid overflow
        self.crits_norm = {}
        self.ypred = 0
        self.total_weight = 0

        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            nm = x.shape[0]
            diffx1 = np.mean(xtar1, axis=0) - np.mean(x1, axis=0)
            XTX = x1.T.dot(x1)/nm + self.lamMatch * np.outer(diffx1, diffx1)
            XTY = x1.T.dot(y)/nm

            A = np.eye(x1.shape[1])
            A[-1, -1] = 0

            self.betas[m] = np.linalg.solve(XTX + self.lamL2*A, XTY)
            ypreds[m] = xtar1.dot(self.betas[m])
            # souce env selection criteria is source loss
            self.crits[m] = np.sum((x1.dot(self.betas[m])-y)**2)/nm  + self.lamMatch * np.inner(diffx1, self.betas[m])**2

        minDiffIndx = min(self.crits.items(), key=operator.itemgetter(1))[0]
        # kept for version compability
        self.minDiffIndx = minDiffIndx
        self.min_critindex = minDiffIndx
        # use normalized weights to avoid numerical overflow
        for m in source:
            self.crits_norm[m] = self.crits[m] - self.crits[self.min_critindex]
            self.ypred += np.exp(-self.weightrho * self.crits_norm[m]) * ypreds[m]
            self.total_weight += np.exp(-self.weightrho * self.crits_norm[m])

        self.ypred /= self.total_weight

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = 0
        for k in range(len(self.source)):
            ypredX1 += np.exp(-self.weightrho * self.crits_norm[self.source[k]]) * X1.dot(self.betas[self.source[k]])
        ypredX1 /= self.total_weight

        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_Ridge{:.1f}".format(self.lamL2)

class CIPalt(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, no target env is needed"""
    def __init__(self, lamCIP=10.0, lamL2=0.0):
        self.lamCIP = lamCIP
        self.lamL2 = lamL2

    def fit(self, data, source, target):
        super().fit(data, source, target)

        XTX = 0
        XTY = 0
        boolA = False
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            n1 = x1.shape[0]
            XTX += x1.T.dot(x1) / n1
            XTY += x1.T.dot(y) / n1
            conditionx1 = np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1)
            for j in source:
                if j != m:
                    xj, yj = data[j]
                    xj1 = np.concatenate((xj, np.ones((xj.shape[0], 1))), axis=1)
                    conditionxj1 = np.mean(xj1, axis=0) - np.mean(yj) * 1./np.sum(yj**2) * yj.dot(xj1)
                    diffxj1 = conditionx1 - conditionxj1
                    XTX += self.lamCIP / len(source) * 2. * np.outer(diffxj1, diffxj1)

            if not boolA:
                A = np.eye(x1.shape[1])
                A[-1, -1] = 0
                boolA = True

        beta = np.linalg.solve(XTX + self.lamL2 * A, XTY)
        self.beta = beta

        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)

        ypred = xtar1.dot(beta)
        self.ypred = ypred

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamCIP) + "_Ridge{:.1f}".format(self.lamL2)

class CIP(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, no target env is needed"""
    def __init__(self, lamCIP=10.0, lamL2=0.0):
        self.lamCIP = lamCIP
        self.lamL2 = lamL2

    def fit(self, data, source, target):
        super().fit(data, source, target)

        XTX = 0
        XTY = 0
        boolA = False
        avconditionx1 = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            avconditionx1 += (np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1))/len(source)
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            n1 = x1.shape[0]
            XTX += x1.T.dot(x1) / n1 / len(source)
            XTY += x1.T.dot(y) / n1 / len(source)
            conditionx1 = np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1)
            diffx1 = conditionx1 - avconditionx1
            XTX += self.lamCIP / len(source) * np.outer(diffx1, diffx1)

            if not boolA:
                A = np.eye(x1.shape[1])
                A[-1, -1] = 0
                boolA = True

        self.beta = np.linalg.solve(XTX + self.lamL2 * A, XTY)

        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)

        self.ypred = xtar1.dot(self.beta)

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_CIP{:.1f}".format(self.lamCIP) + "_Ridge{:.1f}".format(self.lamL2)


class RII(BaseEstimator):
    """Residiual invariant and independent estimator,
       Match the residual Y - X * beta across source envs, no target env is needed"""
    def __init__(self, lamRII=10.0, lamL2=0.0):
        self.lamRII = lamRII
        self.lamL2 = lamL2

    def fit(self, data, source, target):
        super().fit(data, source, target)

        XTX = 0
        XTY = 0
        boolA = False
        avgx1mean = 0
        avgymean = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            avgx1mean += np.mean(x1, axis=0)/len(source)
            avgymean += np.mean(y)/len(source)



        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            n1 = x1.shape[0]
            x1mean = np.mean(x1, axis=0)
            ymean = np.mean(y)
            xtymean = x1.T.dot(y - ymean) / n1
            ytymean = np.mean(y * (y-ymean))
            XTX += x1.T.dot(x1) / n1
            XTY += x1.T.dot(y) / n1
            diffx1 = x1mean - avgx1mean
            # for the residual invariant penalty
            XTX += self.lamRII * np.outer(diffx1, diffx1)
            XTY += self.lamRII * diffx1 * (ymean - avgymean)
            # for the residual independent penalty
            XTX += self.lamRII * np.outer(xtymean, xtymean)
            XTY += self.lamRII * xtymean * ytymean


            if not boolA:
                A = np.eye(x1.shape[1])
                A[-1, -1] = 0
                boolA = True

        beta = np.linalg.solve(XTX + self.lamL2 * A, XTY)
        self.beta = beta

        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)

        ypred = xtar1.dot(beta)
        self.ypred = ypred

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_RII{:.1f}".format(self.lamRII) + "_Ridge{:.1f}".format(self.lamL2)


class CondMatchSrcTarWeigh(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, use Yhat as proxy of Y to do conditional match between source and target.
    This method is not guaranteed to work"""
    def __init__(self, lamMatch=10.0, lamL2=0.0):
        self.lamMatch = lamMatch
        self.lamL2 = lamL2

    def fit(self, data, source, target):
        super().fit(data, source, target)

        # use source envs to match the conditional mean
        # find beta_invariant
        XTX = 0
        XTY = 0
        boolA = False
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            n1 = x1.shape[0]
            XTX += x1.T.dot(x1) / n1
            XTY += x1.T.dot(y) / n1
            conditionx1 = np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1)
            for j in source:
                if j != m:
                    xj, yj = data[j]
                    xj1 = np.concatenate((xj, np.ones((xj.shape[0], 1))), axis=1)
                    conditionxj1 = np.mean(xj1, axis=0) - np.mean(yj) * 1./np.sum(yj**2) * yj.dot(xj1)
                    diffxj1 = conditionx1 - conditionxj1
                    XTX += self.lamCIP * np.outer(diffxj1, diffxj1)

            if not boolA:
                A = np.eye(x1.shape[1])
                A[-1, -1] = 0
                boolA = True

        beta_invariant = np.linalg.solve(XTX + self.lamL2 * A, XTY)
        self.beta_invariant = beta_invariant

        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)

        # use Yhat as proxy of Y in the target env
        yguesstar = xtar1.dot(beta_invariant)
        conditionxtar1 = np.mean(xtar1, axis=0) \
            - np.mean(yguesstar) * 1./np.sum(yguesstar**2) * yguesstar.dot(xtar1)

        # now do conditonal match between each source env and target env
        betas = {}
        ypreds = {}
        diffs = {}
        self.ypred = 0
        self.total_weight = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            nm = x.shape[0]
            yguess = x1.dot(beta_invariant)

            conditionx1 = np.mean(x1, axis=0) - np.mean(yguess) * 1./np.sum(yguess**2) * yguess.dot(x1)
            diffx1 = conditionx1 - conditionxtar1

            XTXt = x1.T.dot(x1)/nm + self.lamMatch * np.outer(diffx1, diffx1)
            XTYt = x1.T.dot(y)/nm

            betas[m] = np.linalg.solve(XTXt + self.lamL2 * A, XTYt)
            ypreds[m] = xtar1.dot(betas[m])
            diffs[m] = np.inner(diffx1, betas[m])**2
#             diffs[m] = np.sum((x1.dot(betas[m])-y)**2)/nm+ self.lamMatch * np.inner(diffx1, betas[m])**2
            self.ypred += np.exp(-10000 * diffs[m]) * ypreds[m]
            self.total_weight += np.exp(-10000 * diffs[m])
        self.ypred /= self.total_weight
#         m_argmin = min(diffs.items(), key=operator.itemgetter(1))[0]
#         self.ypred = ypreds[m_argmin]

        self.betas = betas
        self.ypreds = ypreds
        self.diffs = diffs

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = 0
        for k in range(len(self.source)):
            ypredX1 += np.exp(-10000 * self.diffs[self.source[k]])* X1.dot(self.betas[self.source[k]])
        ypredX1 /= self.total_weight
#         m_argmin = min(self.diffs.items(), key=operator.itemgetter(1))[0]
#         ypredX1 = X1.dot(self.betas[m_argmin])
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_Ridge{:.1f}".format(self.lamL2)

class CIRM(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, use Yhat as proxy of Y to remove the Y parts in X.
    Match on the residual between one source env and target env"""
    def __init__(self, lamCIP=10.0, lamMatch=10.0, lamL2=0.0, sourceInd = 0):
        self.lamCIP = lamCIP
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.sourceInd = sourceInd

    def fit(self, data, source, target):
        super().fit(data, source, target)

        # Step 1: use source envs to match the conditional mean
        # find beta_invariant
        XTX = 0
        XTY = 0
        boolA = False
        avconditionx1 = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            avconditionx1 += (np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1))/len(source)
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            n1 = x1.shape[0]
            XTX += x1.T.dot(x1) / n1
            XTY += x1.T.dot(y) / n1
            conditionx1 = np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1)
            diffx1 = conditionx1 - avconditionx1
            XTX += self.lamCIP * np.outer(diffx1, diffx1)

            if not boolA:
                A = np.eye(x1.shape[1])
                A[-1, -1] = 0
                boolA = True

        beta_invariant = np.linalg.solve(XTX + self.lamL2*A, XTY)
        self.beta_invariant = beta_invariant

        # Step 2: remove the invariant part on all source envs, so that everything is independent of Y
        # get that coefficient b
        YsrcMean = 0
        ntotal = 0
        for m in source:
            YsrcMean += np.sum(data[m][1])
            ntotal += data[m][1].shape[0]
        YsrcMean /= ntotal

        XTY = 0
        YTY = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            yguess = x1.dot(beta_invariant)
            # yguess = x.dot(beta_invariant[:-1])
            yCentered = y - YsrcMean
            YTY += np.sum(yguess * yCentered)
            XTY += x.T.dot(yCentered)
        b_invariant = np.zeros_like(beta_invariant)
        b_invariant[:-1] = XTY / YTY
        self.b_invariant = b_invariant

        # Step 3: mean match between source and target on the residual, after  transforming the covariates X - (X * beta_invariant) * b_invariant
        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        conditionxtar1 = np.mean(xtar1, axis=0) - np.mean(xtar1.dot(beta_invariant)) * b_invariant
        conditionxtar1[-1] = 0

        betas = {}
        ypreds = {}
        ypred = 0

        x, y = data[source[self.sourcInd]]
        x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        nm = x.shape[0]
        conditionx1 = np.mean(x1, axis=0) - np.mean(x1.dot(beta_invariant)) * b_invariant
        conditionx1[-1] = 0
        diffx1 = conditionx1 - conditionxtar1

        XTXt = x1.T.dot(x1)/nm + self.lamMatch * np.outer(diffx1, diffx1)
        XTYt = x1.T.dot(y)/nm

        self.beta = np.linalg.solve(XTXt + self.lamL2*A, XTYt)
        self.ypred = xtar1.dot(self.beta)

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_CIP{:.1f}".format(self.lamCIP) + "_Match{:.1f}".format(self.lamMatch) + "_Ridge{:.1f}".format(self.lamL2)


class CIRMi(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, use Yhat as proxy of Y to remove the Y parts in X.
    Match on the residual between one source env and target env
    with additional residual independent constraint"""
    def __init__(self, lamCIP=10.0, lamMatch=10.0, lamL2=0.0, sourceInd = 0):
        self.lamCIP = lamCIP
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.sourceInd = sourceInd

    def fit(self, data, source, target):
        super().fit(data, source, target)

        # Step 1: use source envs to match the conditional mean
        # find beta_invariant
        XTX = 0
        XTY = 0
        boolA = False
        avconditionx1 = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            avconditionx1 += (np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1))/len(source)
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            n1 = x1.shape[0]
            XTX += x1.T.dot(x1) / n1
            XTY += x1.T.dot(y) / n1
            conditionx1 = np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1)
            diffx1 = conditionx1 - avconditionx1
            XTX += self.lamCIP * np.outer(diffx1, diffx1)

            if not boolA:
                A = np.eye(x1.shape[1])
                A[-1, -1] = 0
                boolA = True

        beta_invariant = np.linalg.solve(XTX + self.lamL2*A, XTY)
        self.beta_invariant = beta_invariant

        # Step 2: remove the invariant part on all source envs, so that everything is independent of Y
        # get that coefficient b
        YsrcMean = 0
        ntotal = 0
        for m in source:
            YsrcMean += np.sum(data[m][1])
            ntotal += data[m][1].shape[0]
        YsrcMean /= ntotal

        XTY = 0
        YTY = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            yguess = x1.dot(beta_invariant)
            # yguess = x.dot(beta_invariant[:-1])
            yCentered = y - YsrcMean
            YTY += np.sum(yguess * yCentered)
            XTY += x.T.dot(yCentered)
        b_invariant = np.zeros_like(beta_invariant)
        b_invariant[:-1] = XTY / YTY
        self.b_invariant = b_invariant

        # Step 3: mean match between source and target on the residual, after  transforming the covariates X - (X * beta_invariant) * b_invariant
        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        conditionxtar1 = np.mean(xtar1, axis=0) - np.mean(xtar1.dot(beta_invariant)) * b_invariant
        conditionxtar1[-1] = 0

        betas = {}
        ypreds = {}
        diffs = {}
        ypred = 0

        x, y = data[source[self.sourcInd]]
        x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        nm = x.shape[0]
        conditionx1 = np.mean(x1, axis=0) - np.mean(x1.dot(beta_invariant)) * b_invariant
        conditionx1[-1] = 0
        diffx1 = conditionx1 - conditionxtar1

        XTXt = x1.T.dot(x1)/nm + self.lamMatch * np.outer(diffx1, diffx1)
        XTYt = x1.T.dot(y)/nm

        ymean = np.mean(y)
        xtymean = x1.T.dot(y - ymean) / nm
        ytymean = np.mean(y * (y-ymean))
        # for the residual independent penalty
        XTXt += self.lamMatch * np.outer(xtymean, xtymean)
        XTYt += self.lamMatch * xtymean * ytymean

        self.beta = np.linalg.solve(XTXt + self.lamL2*A, XTYt)
        self.ypred = xtar1.dot(self.beta)

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_CIP{:.1f}".format(self.lamCIP) + "_Match{:.1f}".format(self.lamMatch) + "_Ridge{:.1f}".format(self.lamL2)


class CIRMweigh(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, use Yhat as proxy of Y to remove the Y parts in X.
    Match on the residual between one source env and target env"""
    def __init__(self, lamCIP=10.0, lamMatch=10.0, lamL2=0.0, weightrho=1000.):
        self.lamCIP = lamCIP
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.weightrho = weightrho

    def fit(self, data, source, target):
        super().fit(data, source, target)

        # Step 1: use source envs to match the conditional mean
        # find beta_invariant
        XTX = 0
        XTY = 0
        boolA = False
        avconditionx1 = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            avconditionx1 += (np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1))/len(source)
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            n1 = x1.shape[0]
            XTX += x1.T.dot(x1) / n1
            XTY += x1.T.dot(y) / n1
            conditionx1 = np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1)
            diffx1 = conditionx1 - avconditionx1
            XTX += self.lamCIP * np.outer(diffx1, diffx1)

            if not boolA:
                A = np.eye(x1.shape[1])
                A[-1, -1] = 0
                boolA = True

        beta_invariant = np.linalg.solve(XTX + self.lamL2*A, XTY)
        self.beta_invariant = beta_invariant

        # Step 2: remove the invariant part on all source envs, so that everything is independent of Y
        # get that coefficient b
        YsrcMean = 0
        ntotal = 0
        for m in source:
            YsrcMean += np.sum(data[m][1])
            ntotal += data[m][1].shape[0]
        YsrcMean /= ntotal

        XTY = 0
        YTY = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            yguess = x1.dot(beta_invariant)
            # yguess = x.dot(beta_invariant[:-1])
            yCentered = y - YsrcMean
            YTY += np.sum(yguess * yCentered)
            XTY += x.T.dot(yCentered)
        b_invariant = np.zeros_like(beta_invariant)
        b_invariant[:-1] = XTY / YTY
        self.b_invariant = b_invariant

        # Step 3: mean match between source and target on the residual, after  transforming the covariates X - (X * beta_invariant) * b_invariant
        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        conditionxtar1 = np.mean(xtar1, axis=0) - np.mean(xtar1.dot(beta_invariant)) * b_invariant
        conditionxtar1[-1] = 0

        self.betas = {}
        ypreds = {}
        self.crits_norm = {}
        self.crits = {}
        self.ypred = 0
        self.total_weight = 0

        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            nm = x.shape[0]
            conditionx1 = np.mean(x1, axis=0) - np.mean(x1.dot(beta_invariant)) * b_invariant
            conditionx1[-1] = 0
            diffx1 = conditionx1 - conditionxtar1

            XTXt = x1.T.dot(x1)/nm + self.lamMatch * np.outer(diffx1, diffx1)
            XTYt = x1.T.dot(y)/nm

            self.betas[m] = np.linalg.solve(XTXt + self.lamL2*A, XTYt)
            ypreds[m] = xtar1.dot(self.betas[m])

            self.crits[m] = np.sum((x1.dot(self.betas[m])-y)**2)/nm+ self.lamMatch * np.inner(diffx1, self.betas[m])**2

        minDiffIndx = min(self.crits.items(), key=operator.itemgetter(1))[0]
        self.minDiffIndx = minDiffIndx
        self.min_critindex = minDiffIndx
        # use normalized weights to avoid numerical overflow
        for m in source:
            self.crits_norm[m] = self.crits[m] - self.crits[self.min_critindex]
            self.ypred += np.exp(-self.weightrho * self.crits_norm[m]) * ypreds[m]
            self.total_weight += np.exp(-self.weightrho * self.crits_norm[m])

        self.ypred /= self.total_weight

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = 0
        for k in range(len(self.source)):
            ypredX1 += np.exp(-self.weightrho * self.crits_norm[self.source[k]]) * X1.dot(self.betas[self.source[k]])
        ypredX1 /= self.total_weight
#         m_argmin = min(self.diffs.items(), key=operator.itemgetter(1))[0]
#         ypredX1 = X1.dot(self.betas[m_argmin])
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_CIP{:.1f}".format(self.lamCIP) + "_Match{:.1f}".format(self.lamMatch) + "_Ridge{:.1f}".format(self.lamL2)


class CIRMmixweigh(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, use Yhat as proxy of Y to remove the Y parts in X.
    Match on the residual between one source env and target env
    the version that deals with mixed-causal-anticausal case
    we first remove the causal part, do CIRM and then add the causal part back"""
    def __init__(self, causal_index=[0], lamCIP=10.0, lamMatch=10.0, lamL2=0.0, weightrho = 1000.):
        self.lamCIP = lamCIP
        self.lamMatch = lamMatch
        self.lamL2 = lamL2
        self.causal_index = causal_index
        self.weightrho = weightrho

    def fit(self, data, source, target):
        super().fit(data, source, target)

        d = data[source[0]][0].shape[1]
        self.noncausal_index = list(set(np.arange(d)) - set(self.causal_index))

        def get_causal_beta(indexk):
            XY = 0.
            XX = 0.
            ntotal = 0
            boolA = False
            betacausal1_restrict = 0
            for m in source:
                x, y = data[m]
                if indexk != -1:
                    y = x[:, indexk]
                # only use the causal part of x
                x = x[:, self.causal_index]
                x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
                ntotal += x.shape[0]
                XX += x1.T.dot(x1)
                XY += x1.T.dot(y)

                if not boolA:
                    A = np.eye(x1.shape[1])
                    A[-1, -1] = 0
                    boolA = True

                betacausal1_restrict += np.linalg.solve(XX/ntotal + self.lamL2*A, XY/ntotal)/len(source)
            return(betacausal1_restrict)


        beta_corrections = {}
        for indexk in self.noncausal_index:
            beta_corrections[indexk] = get_causal_beta(indexk)
        beta_corrections[-1] = get_causal_beta(-1)

        # Step 0: run SrcPool on the causal_index

        betacausal1 = np.zeros(d)
        betacausal1[self.causal_index] = beta_corrections[-1][:-1]
        self.betacausal1 = betacausal1
        # for cirm, y - x * betacuasal1 will be used as a replacement for y
        # Now modify the dataset
        dataNew = {}
        for m in np.concatenate((source, [target])):
            x, y = data[m]
            xNew = np.zeros_like(x[:, self.noncausal_index])
            for k, indexk in enumerate(self.noncausal_index):
                xNew[:, k] = x[:, indexk] - x[:, self.causal_index].dot(beta_corrections[indexk][:-1])
            yNew = y - x.dot(self.betacausal1)
            dataNew[m] = xNew, yNew


        # Step 1: use source envs to match the conditional mean
        # find beta_invariant
        XTX = 0
        XTY = 0
        boolA = False
        avconditionx1 = 0
        for m in source:
            x, y = dataNew[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            avconditionx1 += (np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1))/len(source)
        for m in source:
            x, y = dataNew[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            n1 = x1.shape[0]
            XTX += x1.T.dot(x1) / n1
            XTY += x1.T.dot(y) / n1
            conditionx1 = np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1)
            diffx1 = conditionx1 - avconditionx1
            XTX += self.lamCIP * np.outer(diffx1, diffx1)

            if not boolA:
                A = np.eye(x1.shape[1])
                A[-1, -1] = 0
                boolA = True

        beta_invariant = np.linalg.solve(XTX + self.lamL2*A, XTY)
        self.beta_invariant = beta_invariant

        # Step 2: remove the invariant part on all source envs, so that everything is independent of Y
        # get that coefficient b
        YsrcMean = 0
        ntotal = 0
        for m in source:
            YsrcMean += np.sum(dataNew[m][1])
            ntotal += dataNew[m][1].shape[0]
        YsrcMean /= ntotal

        XTY = 0
        YTY = 0
        for m in source:
            x, y = dataNew[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            yguess = x1.dot(beta_invariant)
            # yguess = x.dot(beta_invariant[:-1])
            yCentered = y - YsrcMean
            YTY += np.sum(yguess * yCentered)
            XTY += x.T.dot(yCentered)
        b_invariant = np.zeros_like(beta_invariant)
        b_invariant[:-1] = XTY / YTY
        self.b_invariant = b_invariant

        # Step 3: mean match between source and target on the residual, after  transforming the covariates X - (X * beta_invariant) * b_invariant
        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        xtarNew, _ = dataNew[target]
        xtarNew1 = np.concatenate((xtarNew, np.ones((xtarNew.shape[0], 1))), axis=1)
        conditionxtar1 = np.mean(xtarNew1, axis=0) - np.mean(xtarNew1.dot(beta_invariant)) * b_invariant
        conditionxtar1[-1] = 0

        self.betas = {}
        ypreds = {}
        self.crits = {}
        self.crits_norm = {}
        self.ypred = 0
        self.total_weight = 0
        for m in source:
            x, y = dataNew[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            nm = x.shape[0]
            conditionx1 = np.mean(x1, axis=0) - np.mean(x1.dot(beta_invariant)) * b_invariant
            conditionx1[-1] = 0
            diffx1 = conditionx1 - conditionxtar1

            XTXt = x1.T.dot(x1)/nm + self.lamMatch * np.outer(diffx1, diffx1)
            XTYt = x1.T.dot(y)/nm

            betaNew = np.linalg.solve(XTXt + self.lamL2*A, XTYt)
            self.betas[m] = np.zeros(d+1)
            self.betas[m][self.noncausal_index] = betaNew[:-1]
            self.betas[m][-1] = betaNew[-1]
            self.betas[m][self.causal_index] = beta_corrections[-1][:-1]
            for k, indexk in enumerate(self.noncausal_index):
                self.betas[m][self.causal_index] -= betaNew[k]*beta_corrections[indexk][:-1]
            ypreds[m] = xtar1.dot(self.betas[m])

            self.crits[m] = np.sum((x1.dot(betaNew)-y)**2)/nm + self.lamMatch * np.inner(diffx1, betaNew)**2
        minDiffIndx = min(self.crits.items(), key=operator.itemgetter(1))[0]
        self.minDiffIndx = minDiffIndx
        self.min_critindex = minDiffIndx
        # use normalized weights to avoid numerical overflow
        for m in source:
            self.crits_norm[m] = self.crits[m] - self.crits[minDiffIndx]
            self.ypred += np.exp(-self.weightrho * self.crits_norm[m]) * ypreds[m]
            self.total_weight += np.exp(-self.weightrho * self.crits_norm[m])
        self.ypred /= self.total_weight

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = 0
        for k in range(len(self.source)):
            ypredX1 += np.exp(-self.weightrho * self.crits_norm[self.source[k]]) * X1.dot(self.betas[self.source[k]])
        ypredX1 /= self.total_weight
#         m_argmin = min(self.diffs.items(), key=operator.itemgetter(1))[0]
#         ypredX1 = X1.dot(self.betas[m_argmin])
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_CIP{:.1f}".format(self.lamCIP) + "_Match{:.1f}".format(self.lamMatch) + "_Ridge{:.1f}".format(self.lamL2)


class CIRMiweigh(BaseEstimator):
    """Match the conditional (on Y) mean of X * beta across source envs, use Yhat as proxy of Y to remove the Y parts in X.
    Match on the residual between one source env and target env
        with additional residual independent """
    def __init__(self, lamCIP=10.0, lamMatch=10.0, lamL2=0.0):
        self.lamCIP = lamCIP
        self.lamMatch = lamMatch
        self.lamL2 = lamL2

    def fit(self, data, source, target):
        super().fit(data, source, target)

        # Step 1: use source envs to match the conditional mean
        # find beta_invariant
        XTX = 0
        XTY = 0
        boolA = False
        avconditionx1 = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            avconditionx1 += (np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1))/len(source)
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            n1 = x1.shape[0]
            XTX += x1.T.dot(x1) / n1
            XTY += x1.T.dot(y) / n1
            conditionx1 = np.mean(x1, axis=0) - np.mean(y) * 1./np.sum(y**2) * y.dot(x1)
            diffx1 = conditionx1 - avconditionx1
            XTX += self.lamCIP * np.outer(diffx1, diffx1)

            if not boolA:
                A = np.eye(x1.shape[1])
                A[-1, -1] = 0
                boolA = True

        beta_invariant = np.linalg.solve(XTX + self.lamL2*A, XTY)
        self.beta_invariant = beta_invariant

        # Step 2: remove the invariant part on all source envs, so that everything is independent of Y
        # get that coefficient b
        YsrcMean = 0
        ntotal = 0
        for m in source:
            YsrcMean += np.sum(data[m][1])
            ntotal += data[m][1].shape[0]
        YsrcMean /= ntotal

        XTY = 0
        YTY = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            yguess = x1.dot(beta_invariant)
            # yguess = x.dot(beta_invariant[:-1])
            yCentered = y - YsrcMean
            YTY += np.sum(yguess * yCentered)
            XTY += x.T.dot(yCentered)
        b_invariant = np.zeros_like(beta_invariant)
        b_invariant[:-1] = XTY / YTY
        self.b_invariant = b_invariant

        # Step 3: mean match between source and target on the residual, after  transforming the covariates X - (X * beta_invariant) * b_invariant
        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        conditionxtar1 = np.mean(xtar1, axis=0) - np.mean(xtar1.dot(beta_invariant)) * b_invariant
        conditionxtar1[-1] = 0

        betas = {}
        ypreds = {}
        diffs = {}
        ypred = 0
        total_weight = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            nm = x.shape[0]
            conditionx1 = np.mean(x1, axis=0) - np.mean(x1.dot(beta_invariant)) * b_invariant
            conditionx1[-1] = 0
            diffx1 = conditionx1 - conditionxtar1

            XTXt = x1.T.dot(x1)/nm + self.lamMatch * np.outer(diffx1, diffx1)
            XTYt = x1.T.dot(y)/nm

            ymean = np.mean(y)
            xtymean = x1.T.dot(y - ymean) / nm
            ytymean = np.mean(y * (y-ymean))
            # for the residual independent penalty
            XTXt += self.lamMatch * np.outer(xtymean, xtymean)
            XTYt += self.lamMatch * xtymean * ytymean

            betas[m] = np.linalg.solve(XTXt + self.lamL2*A, XTYt)
            ypreds[m] = xtar1.dot(betas[m])
            diffs[m] = np.inner(diffx1, betas[m])**2
#             diffs[m] = np.sum((x1.dot(betas[m])-y)**2)/nm+ self.lamMatch * np.inner(diffx1, betas[m])**2
            ypred += np.exp(-10000 * diffs[m]) * ypreds[m]
            total_weight += np.exp(-10000 * diffs[m])
        ypred /= total_weight
        self.ypred = ypred
        self.total_weight = total_weight
#         m_argmin = min(diffs.items(), key=operator.itemgetter(1))[0]
#         self.ypred = ypreds[m_argmin]


        self.betas = betas
        self.ypreds = ypreds
        self.diffs = diffs

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = 0
        for k in range(len(self.source)):
            ypredX1 += np.exp(-10000 * self.diffs[self.source[k]]) * X1.dot(self.betas[self.source[k]])
        ypredX1 /= self.total_weight
#         m_argmin = min(self.diffs.items(), key=operator.itemgetter(1))[0]
#         ypredX1 = X1.dot(self.betas[m_argmin])
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_CIP{:.1f}".format(self.lamCIP) + "_Match{:.1f}".format(self.lamMatch) + "_Ridge{:.1f}".format(self.lamL2)


class RIIRMweigh(BaseEstimator):
    """Residiual invariant and independent, residual match estimator,
       Match the residual Y - X * beta across source envs,
    use Yhat as proxy of Y to remove the Y parts in X.
    Match on the residual between one source env and target env"""
    def __init__(self, lamRII=10.0, lamMatch=10.0, lamL2=0.0):
        self.lamRII = lamRII
        self.lamMatch = lamMatch
        self.lamL2 = lamL2

    def fit(self, data, source, target):
        super().fit(data, source, target)

        # Step 1: use source envs to match the conditional mean
        # find beta_invariant
        XTX = 0
        XTY = 0
        boolA = False
        avgx1mean = 0
        avgymean = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            avgx1mean += np.mean(x1, axis=0)/len(source)
            avgymean += np.mean(y)/len(source)

        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            n1 = x1.shape[0]
            x1mean = np.mean(x1, axis=0)
            ymean = np.mean(y)
            xtymean = x1.T.dot(y - ymean) / n1
            ytymean = np.mean(y * (y-ymean))
            XTX += x1.T.dot(x1) / n1
            XTY += x1.T.dot(y) / n1
            diffx1 = x1mean - avgx1mean
            # for the residual invariant penalty
            XTX += self.lamRII * np.outer(diffx1, diffx1)
            XTY += self.lamRII * diffx1 * (ymean - avgymean)
            # for the residual independent penalty
            XTX += self.lamRII * np.outer(xtymean, xtymean)
            XTY += self.lamRII * xtymean * ytymean


            if not boolA:
                A = np.eye(x1.shape[1])
                A[-1, -1] = 0
                boolA = True

        beta_invariant = np.linalg.solve(XTX + self.lamL2*A, XTY)
        self.beta_invariant = beta_invariant

        # Step 2: remove the invariant part on all source envs, so that everything is independent of Y
        # get that coefficient b
        YsrcMean = 0
        ntotal = 0
        for m in source:
            YsrcMean += np.sum(data[m][1])
            ntotal += data[m][1].shape[0]
        YsrcMean /= ntotal

        XTY = 0
        YTY = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            yguess = x1.dot(beta_invariant)
            # yguess = x.dot(beta_invariant[:-1])
            yCentered = y - YsrcMean
            YTY += np.sum(yguess * yCentered)
            XTY += x.T.dot(yCentered)
        b_invariant = np.zeros_like(beta_invariant)
        b_invariant[:-1] = XTY / YTY
        self.b_invariant = b_invariant

        # Step 3: mean match between source and target on the residual, after  transforming the covariates X - (X * beta_invariant) * b_invariant
        xtar, _ = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        conditionxtar1 = np.mean(xtar1, axis=0) - np.mean(xtar1.dot(beta_invariant)) * b_invariant
        conditionxtar1[-1] = 0

        betas = {}
        ypreds = {}
        diffs = {}
        ypred = 0
        total_weight = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            nm = x.shape[0]
            conditionx1 = np.mean(x1, axis=0) - np.mean(x1.dot(beta_invariant)) * b_invariant
            conditionx1[-1] = 0
            diffx1 = conditionx1 - conditionxtar1

            XTXt = x1.T.dot(x1)/nm + self.lamMatch * np.outer(diffx1, diffx1)
            XTYt = x1.T.dot(y)/nm

            betas[m] = np.linalg.solve(XTXt + self.lamL2*A, XTYt)
            ypreds[m] = xtar1.dot(betas[m])
            diffs[m] = np.inner(diffx1, betas[m])**2
#             diffs[m] = np.sum((x1.dot(betas[m])-y)**2)/nm+ self.lamMatch * np.inner(diffx1, betas[m])**2
            ypred += np.exp(-10000 * diffs[m]) * ypreds[m]
            total_weight += np.exp(-10000 * diffs[m])
        ypred /= total_weight
        self.ypred = ypred
        self.total_weight = total_weight
#         m_argmin = min(diffs.items(), key=operator.itemgetter(1))[0]
#         self.ypred = ypreds[m_argmin]


        self.betas = betas
        self.ypreds = ypreds
        self.diffs = diffs

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = 0
        for k in range(len(self.source)):
            ypredX1 += np.exp(-10000 * self.diffs[self.source[k]]) * X1.dot(self.betas[self.source[k]])
        ypredX1 /= self.total_weight
#         m_argmin = min(self.diffs.items(), key=operator.itemgetter(1))[0]
#         ypredX1 = X1.dot(self.betas[m_argmin])
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_RII{:.1f}".format(self.lamRII) + "_Match{:.1f}".format(self.lamMatch) + "_Ridge{:.1f}".format(self.lamL2)



class Anchor(BaseEstimator):
    """Anchor regression"""
    def __init__(self, lamMatch=10., lamL2=0.):
        self.lamMatch=lamMatch
        self.lamL2=lamL2

    def fit(self, data, source, target):
        super().fit(data, source, target)

        xmean0 = 0
        ymean0 = 0
        for m in source:
            x, y = data[m]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            xmean0 += np.mean(x1, axis=0)
            ymean0 += np.mean(y)
        xmean0 /= len(source)
        ymean0 /= len(source)

        XTX = 0
        XTY = 0
        boolA = False
        for m in source:
            x, y = data[m]
            nm = x.shape[0]
            x1 = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            XTX += x1.T.dot(x1)/nm
            XTY += x1.T.dot(y)/nm

            diffxm = np.mean(x1, axis=0) - xmean0
            XTX += self.lamMatch * np.outer(diffxm, diffxm)
            diffym = np.mean(y) - ymean0
            XTY += self.lamMatch * diffym * diffxm

            if not boolA:
                A = np.eye(x1.shape[1])
                A[-1, -1] = 0
                boolA = True

        self.beta = np.linalg.solve(XTX + self.lamL2*A, XTY)
        xtar, ytar = data[target]
        xtar1 = np.concatenate((xtar, np.ones((xtar.shape[0], 1))), axis=1)
        self.ypred = xtar1.dot(self.beta)

        return self

    def predict(self, X):
        X1 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        ypredX1 = X1.dot(self.beta)
        return ypredX1

    def __str__(self):
        return self.__class__.__name__ + "_Match{:.1f}".format(self.lamMatch) + "_Ridge{:.1f}".format(self.lamL2)


