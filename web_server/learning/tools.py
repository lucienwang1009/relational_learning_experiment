#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project : experiments_networked
# @File    : tools.py
# @Author  : lucien
# @Email   : lucienwang@qq.com
# @Date    : 20/02/2018
# ------------------------------------------------------
# Change Activity
# 20/02/2018  :

__author__ = 'lucien'

import logging.config
import logging
import time
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib as mat
import random
import numpy as np
from collections import Counter
import csv

current_microsecond = lambda: int(round(time.time()))


def plot_power_law(g):
    degree = [d[1] for d in list(nx.degree(g))]
    c = Counter(degree)
    x = [d[0] for d in c.items()]
    x = np.asarray(x)
    y = [d[1] for d in c.items()]
    y = np.asarray(y)
    return list(zip(np.log(x), np.log(y)))


def draw_sampled_graph(g, is_sample=False, node_num=100):
    graph = g
    if is_sample and node_num < len(graph.nodes):
        graph = nx.Graph()
        sampled_nodes = random.sample(list(g.nodes()), node_num-1)
        sampled_nodes.append(0)
        for n in sampled_nodes:
            for adj in g.adj[n]:
                if adj in sampled_nodes:
                    graph.add_edge(n, adj)
    pos = nx.kamada_kawai_layout(graph)
    cmap = plt.cm.Blues
    max_degree = max(dict(graph.degree).values())
    for n in graph.nodes:
        graph.nodes[n]['x'] = pos[n][0]
        graph.nodes[n]['y'] = pos[n][1]
        graph.nodes[n]['symbolSize'] = graph.degree(n)
        color = cmap(1 - graph.degree(n) / max_degree + 0.3)
        graph.nodes[n]['itemStyle'] = {'color': '%s' % mat.colors.rgb2hex(color[:3])}
    # nx.draw(graph, node_size=[5*graph.degree(v) for v in graph], node_color=[-math.sqrt(graph.degree(v)) for v in graph],
    #        cmap=plt.cm.Blues, with_lables=False, edge_color='#1862ab', width=0.3, pos=pos)
    return graph

class ExperimentsProesser:
    def __init__():
        self.process = -1
        self.results = []
