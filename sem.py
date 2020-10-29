import numpy as np

import matplotlib.pyplot as plt

from networkx import nx

# basic structural equation
class SEM():
    def __init__(self, B, noisef, interAf, invariantList=[], message=None):
        self.B = B
        self.dp1 = B.shape[0]
        self.M = interAf.M
        self.interAf = interAf
        self.noisef = noisef

        # just for plot purpose
        self.invariantList = invariantList
        self.message = message

        # raise an error if it is not invertible
        self.IBinv = np.linalg.inv(np.eye(self.dp1) - self.B)

    def generateSamples(self, n, m=0):
        # generate n samples from mth environment
        noise = self.noisef.generate(0, n, self.dp1)
        interA = self.interAf.generate(m, n, self.dp1)

        data = (noise + interA).dot(self.IBinv.T)

        # return x and y separately
        return data[:, :-1], data[:, -1]

    def generateAllSamples(self, n):
        res = {}
        for m in range(self.M):
            res[m] = self.generateSamples(n, m)

        return res

    def draw(self, layout='circular', figsize=(12, 8)):
        plt.figure(figsize=figsize)
        G = nx.DiGraph()
        G.add_nodes_from(np.arange(self.dp1))
        for i in range(self.dp1):
            for j in range(self.dp1):
                if not np.isclose(self.B[i, j], 0):
                    G.add_edge(j, i, weight=self.B[i, j])

        # labels
        labels={}
        for i in range(self.dp1-1):
            labels[i] = "X"+str(i)
        labels[self.dp1-1] = "Y"

        # position
        if layout=="spring":
            pos = nx.spring_layout(G)
        elif layout=="kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.circular_layout(G)

        nx.draw_networkx_nodes(G,pos,nodelist = list(np.arange(self.dp1-1)), node_color='b', node_size=1000, alpha=0.5)
        nx.draw_networkx_nodes(G,pos,nodelist = [self.dp1-1], node_color='r',
            node_size=1000, alpha=0.8)
        nx.draw_networkx_nodes(G,pos,nodelist = list(self.invariantList), node_color='y', node_size=1000, alpha=0.8)
        nx.draw_networkx_edges(G,pos,width=3.0,alpha=0.5, arrowsize=40)

        arc_weight=nx.get_edge_attributes(G,'weight')
        arc_weight_format = {i:'{:.2f}'.format(arc_weight[i]) for i in arc_weight}


        nx.draw_networkx_edge_labels(G, pos,edge_color= 'k', label_pos=0.7, edge_labels=arc_weight_format)

        nx.draw_networkx_labels(G,pos,labels,font_size=16)
        plt.draw()
        plt.show()


