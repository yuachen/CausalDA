import numpy as np
from abc import abstractmethod

class RandomGen():
    """Base class for random number generation"""
    @abstractmethod
    def generate(self, m, n, d):
        pass

class Gaussian(RandomGen):
    def __init__(self, M, meanAs, varAs):
        self.M = M
        self.meanAs = meanAs
        self.varAs = varAs

    def generate(self, m, n, d):
        vec = np.random.randn(n, d).dot(np.diag(np.sqrt(self.varAs[m, ])))
        vec += self.meanAs[m, :].reshape(1, -1)
        return vec

class Mix2Gaussian(RandomGen):
    def __init__(self, M, meanAsList, varAs):
        self.M = M
        self.meanAsList = meanAsList
        self.varAs = varAs

    def generate(self, m, n, d):
        vec = np.random.randn(n, d).dot(np.diag(np.sqrt(self.varAs[m, ])))
        mixture_idx = np.random.choice(2, size=n, replace=True, p=[0.5, 0.5])
        for i in range(n):
            vec[i, :] += self.meanAsList[mixture_idx[i]][m, :]
        return vec

class MixkGaussian(RandomGen):
    def __init__(self, M, meanAsList, varAs):
        self.M = M
        self.meanAsList = meanAsList
        self.varAs = varAs
        self.k = len(self.meanAsList)

    def generate(self, m, n, d):
        vec = np.random.randn(n, d).dot(np.diag(np.sqrt(self.varAs[m, ])))
        mixture_idx = np.random.choice(self.k, size=n, replace=True, p=np.ones(self.k)/self.k)
        for i in range(n):
            vec[i, :] += self.meanAsList[mixture_idx[i]][m, :]
        return vec
