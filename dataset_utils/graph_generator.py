#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 09/07/2017 20:32
# @Author  : LucienWang
# @File    : graph_generator.py

import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
# from datetime import datetime
from utils.tools import plot_sampled_graph, current_microsecond
import logging.config

def generate_barabasi(n_nodes, m=1, plot_graph=False):
    graph = nx.barabasi_albert_graph(n_nodes, m)
    if plot_graph:
        plot_sampled_graph(graph)
    return graph


def generate_complete_graph(n_nodes, m=1):
    return nx.complete_graph(n_nodes)


def generate_worst_graph(n_nodes, m=1):
    graph = nx.Graph()
    for i in range(1, int(n_nodes / 2)):
        graph.add_edge(0, i)
    for i in range(int(n_nodes / 2), n_nodes, 2):
        graph.add_edge(i, i + 1)
    return graph

def generate_erdos_renyi(n_nodes, p=.1):
    return nx.complete_graph(n_nodes, p)

class GraphDataSet:
    def __init__(self, G):
        self.G = G

    def dataset(self, n_visible):
        X = []
        Y = []
        for i, j in self.G.edges():
            # dot_features = np.asarray(np.matrix(self.G.node[i]['feature'][:n_visible]).T.dot(np.matrix(self.G.node[j]['feature'][:n_visible]))).flatten()
            # X.append(dot_features)
            X.append(np.hstack((self.G.node[i]['feature'][:n_visible] - self.G.node[j]['feature'][:n_visible])))
            Y.append(self.G.edges[i, j]['label'])
        X = np.asarray(X, dtype='float32')
        Y = np.asarray(Y, dtype='float32')
        return (X, Y)

    def adj_matrix(self):
        adj = np.zeros((len(self.G.nodes), len(self.G.nodes)))
        for i, j in self.G.edges():
            adj[i][j] = 1
            adj[j][i] = 1
        return adj

    def re_generate(self, node_id):
        self.generate_feature(node_id)
        for j in self.G.adj[node_id]:
            self.generate_target(node_id, j)

    def generate_target(self, i, j):
        pass

    def generate_feature(self, node_index):
        self.G.node[node_index]['feature'] = self.random.uniform(-1, 1, size=self.n_features)

    def edges(self):
        return self.G.edges()

    def nodes(self):
        return self.G.nodes()

    def frational_matching(self):
        return nx.max_weight_matching(self.G)


class GraphLinearDataSet(GraphDataSet):
    def __init__(self, G, n_features, random=None, w=None, b=None):
        '''
        Generate networked data.
        :param G: graph
        :param n_hidden: number of invisible features
        :param random: numpy RandomState
        :param w: linear weights
        :param b: intercept
        '''
        GraphDataSet.__init__(self, G)
        self.n_features = n_features
        if random is None:
            self.random = np.random.RandomState(current_microsecond())
        else:
            self.random = random
        # feature
        self.X = []
        # label
        self.Y = []
        # graph
        self.G = G
        # generate feature
        for i in self.G.nodes:
            self.generate_feature(i)
            # generate label
            # the weight of bayesian function
        if w is None:
            self.w = self.random.uniform(-1, 1, size=self.n_features)
        else:
            self.w = w
        if b is None:
            self.b = self.random.uniform(-1, 1)
        else:
            self.b = b
        for i,j in self.G.edges:
            self.generate_target(i, j)

    def generate_target(self, i, j):
        sep = np.dot((self.G.node[i]['feature'] - self.G.node[j]['feature']), self.w)
        self.G.edges[i, j]['label'] = 1 if sep >= 0 else 0

class GraphDistanceRegDataSet(GraphDataSet):
    def __init__(self, G, n_features, random=None, w=None, b=None):
        '''
        Generate networked data.
        :param G: graph
        :param n_hidden: number of invisible features
        :param random: numpy RandomState
        :param w: linear weights
        :param b: intercept
        '''
        GraphDataSet.__init__(self, G)
        self.n_features = n_features
        if random is None:
            self.random = np.random.RandomState(current_microsecond())
        else:
            self.random = random
        # feature
        self.X = []
        # label
        self.Y = []
        # graph
        self.G = G
        # generate feature
        for i in self.G.nodes:
            self.generate_feature(i)
            # generate label
            # the weight of bayesian function
        if w is None:
            self.w = self.random.uniform(-1, 1, size=(self.n_features, self.n_features))
        else:
            self.w = w
        if b is None:
            self.b = self.random.uniform(-1, 1)
        else:
            self.b = b
#        distances = []
        for i,j in self.G.edges:
            self.generate_target(i, j)
#            distances.append(self.G.edges[i,j]['dist'])
#        print(np.median(distances))

    def generate_target(self, i, j):
        dist = self.G.node[i]['feature'] - self.G.node[j]['feature']
        pseudo_dist = np.dot(dist, self.w)
        pseudo_dist = np.dot(pseudo_dist, pseudo_dist)
        self.G.edges[i, j]['label'] = pseudo_dist

class GraphDistanceDataSet(GraphDataSet):
    def __init__(self, G, n_features, random=None, w=None, b=None):
        '''
        Generate networked data.
        :param G: graph
        :param n_hidden: number of invisible features
        :param random: numpy RandomState
        :param w: linear weights
        :param b: intercept
        '''
        GraphDataSet.__init__(self, G)
        self.n_features = n_features
        if random is None:
            self.random = np.random.RandomState(current_microsecond())
        else:
            self.random = random
        # feature
        self.X = []
        # label
        self.Y = []
        # graph
        self.G = G
        # generate feature
        for i in self.G.nodes:
            self.generate_feature(i)
            # generate label
            # the weight of bayesian function
        if w is None:
            self.w = self.random.uniform(-1, 1, size=(self.n_features, self.n_features))
        else:
            self.w = w
        if b is None:
            self.b = self.random.uniform(-1, 1)
        else:
            self.b = b
        distances = []
        for i,j in self.G.edges:
            self.generate_target(i, j)
            distances.append(self.G.edges[i,j]['dist'])
#        print(np.median(distances))

    def generate_target(self, i, j):
        dist = self.G.node[i]['feature'] - self.G.node[j]['feature']
        pseudo_dist = np.dot(self.w, dist)
        pseudo_dist = np.dot(pseudo_dist, pseudo_dist)
        self.G.edges[i, j]['dist'] = pseudo_dist
        self.G.edges[i, j]['label'] = 1 if pseudo_dist <= 20 else 0

class GraphLinearRegDataSet(GraphDataSet):
    def __init__(self, G, n_features, random=None, w=None, b=None):
        '''
        Generate networked data.
        :param G: graph
        :param n_hidden: number of invisible features
        :param random: numpy RandomState
        :param w: linear weights
        :param b: intercept
        '''
        GraphDataSet.__init__(self, G)
        self.n_features = n_features
        if random is None:
            self.random = np.random.RandomState(current_microsecond())
        else:
            self.random = random
        # feature
        self.X = []
        # label
        self.Y = []
        # graph
        self.G = G
        # generate feature
        for i in self.G.nodes:
            self.generate_feature(i)
        if w is None:
            self.w = self.random.uniform(-1, 1, size=2 * self.n_features)
        else:
            self.w = w
        if b is None:
            self.b = self.random.uniform(-1, 1)
        else:
            self.b = b
        for i, j in self.G.edges():
            self.generate_target(i, j)

    def generate_target(self, i, j):
        # sep = np.dot(np.hstack((self.G.node[i]['feature'], self.G.node[j]['feature'])), self.w)
        # self.G.edges[i, j]['label'] = sep
        dist = np.abs(self.G.node[i]['feature']-self.G.node[j]['feature'])
        sep = np.dot(np.hstack((dist, 1)), np.hstack((self.w, self.b)))
        self.G.edges[i, j]['label'] = np.abs(sep)
#
#
# class GraphDistanceDataSet(GraphDataSet):
#     def __init__(self, G, n_features, random=None, threshold=None):
#         '''
#         Generate networked data.
#         :param G: graph
#         :param n_visible: number of visible features
#         :param random: numpy RandomState
#         :param threshold: example whose length is not larger than threshold is labeled as 1.
#         '''
#         GraphDataSet.__init__(self, G)
#         # self.n_examples = n_examples
#         self.n_features = n_features
#         if random is None:
#             self.random = np.random.RandomState(current_microsecond())
#         else:
#             self.random = random
#         # feature
#         self.X = []
#         # label
#         self.Y = []
#         # generate feature
#         for i in range(self.G.number_of_nodes()):
#             self.G.node[i]['feature'] = self.random.uniform(-1, 1, size=self.n_features)
#         # generate label
#         # the weight of bayesian function
#         distance = []
#         for i, j in self.G.edges():
#             distance.append(np.sqrt(np.sum((self.G.node[i]['feature'] - self.G.node[j]['feature']) ** 2)))
#         if threshold is None:
#             self.threshold = np.median(distance)
#         else:
#             self.threshold = threshold
#         k = 0
#         for i, j in self.G.edges():
#             self.G.edges[i, j]['label'] = 1 if distance[k] <= self.threshold else 0
#             k += 1
#
#
# class GraphDistanceRegDataSet(GraphDataSet):
#     def __init__(self, G, n_features, random=None):
#         GraphDataSet.__init__(self, G)
#         self.n_features = n_features
#         if random is None:
#             self.random = np.random.RandomState(current_microsecond())
#         else:
#             self.random = random
#         self.X = []
#         self.Y = []
#         # generate feature
#         for i in range(self.G.number_of_nodes()):
#             self.G.node[i]['feature'] = self.random.uniform(-1, 1, size=self.n_features)
#         for i, j in self.G.edges():
#             distance = np.sqrt(np.sum((self.G.node[i]['feature'] - self.G.node[j]['feature']) ** 2))
#             self.G.edges[i, j]['label'] = distance


def test():
    graph_data = GraphLinearDataSet(1000, 50, 20)
    # X, Y = graph_data.dataset()
    # logger.info(np.str(X))
    # logger.error(np.str(graph_data.adj_matrix()))
    print(graph_data.frational_matching())


if __name__ == '__main__':
    test()

# use linear regression loss function to classify examples,
# so adding weight to each example means adding weight to its feature and label (actually the target value).
