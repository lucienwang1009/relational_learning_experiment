from scipy.optimize import linprog
import numpy as np
import networkx as nx
from cvxopt import solvers, matrix

def fmn(nodes, edges):
    n_examples = len(edges)
    n_nodes = len(nodes)
    A = np.zeros((n_nodes, n_examples))
    k = 0
    edges_of_node = [[] for i in range(n_nodes)]
    for i, j in edges:
        edges_of_node[i].append(k)
        edges_of_node[j].append(k)
        k += 1
    # edges_of_node = edges_of_node.astype(np.int)
    # print(edges_of_node)
    for i in range(n_nodes):
        for e in edges_of_node[i]:
            A[i][e] = 1
    b = np.ones(n_nodes)
    c = -np.ones(n_examples)
    result = linprog(c, A, b)
    # print(edges)
    # logger.info('c: %s\n A: %s\n b: %s' % (np.str(c), np.str(A), np.str(b)))
    return result.x

def maximal_matching(nodes, edges):
    g = nx.from_edgelist(edges)
    mat = nx.maximal_matching(g)
    weight = np.zeros(len(g.edges))
    k = 0
    for e in g.edges:
        if e in mat:
            weight[k] = 1
        k += 1
    return weight

def worst_weighting(nodes, edges):
    n_examples = len(edges)
    weights = np.zeros(n_examples)
    for i in range(int(n_examples / 2)):
        weights[i] = 2 / n_examples
    for i in range(int(n_examples / 2), n_examples):
        weights[i] = 1
    return weights

def worst_weighting1(nodes, edges):
    n_examples = len(edges)
    weights = np.zeros(n_examples)
    weights[0] = 1
    for i in range(int(n_examples / 2), n_examples):
        weights[i] = 1
    return weights

def solve_min_a(nodes, edges):
    n_example = len(edges)
    n_nodes = len(nodes)
    c = np.zeros(n_example+1)
    c[-1] = 1
    A_ub = np.zeros((n_example+n_nodes, n_example+1))
    b_ub = np.zeros(n_example+n_nodes)
    A_eq = np.ones((1, n_example+1))
    A_eq[0, -1] = 0
    b_eq = 1
    for i in range(n_example):
        A_ub[i,i] = -1
    k = 0
    edges_of_node = [[] for i in range(n_nodes)]
    for i, j in edges:
        edges_of_node[i].append(k)
        edges_of_node[j].append(k)
        k += 1
    for i in range(n_nodes):
        for e in edges_of_node[i]:
            A_ub[i, e] = 1
        A_ub[i, -1] = -1
    result = linprog(c, A_ub, b_ub, A_eq, b_eq)
    return np.dot(result.x, c)

def opt_fix_a(nodes, edges, a):
    pass

def my_weight(nodes, edges):
    min_a = solve_min_a(nodes, edges)
    return min_a


if __name__ == '__main__':
    g = nx.barabasi_albert_graph(100, 1)
    print(my_weight(g.nodes, g.edges))

