import networkx as nx
import sys
sys.path.append('C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\Graph Theory') # here is the path with GVIS
from graph_visualization import visualize
from itertools import combinations
import math

# how to merge two graphs:
'''
# Create two graphs
G1 = nx.Graph()
G1.add_edges_from([ ('a',  'b'),  ('b',  'c')])

G2 = nx.Graph()
G2.add_edges_from([ ('c',  'd'),  ('d',  'e')])

# Merge the graphs
G_merged = nx.compose(G1,  G2)
'''
#how to merge multiple graphs
'''
# Create multiple graphs
G1 = nx.Graph()
G1.add_edges_from([ ('a',  'b'),  ('b',  'c')])

G2 = nx.Graph()
G2.add_edges_from([ ('c',  'd'),  ('d',  'e')])

G3 = nx.Graph()
G3.add_edges_from([ ('e',  'f'),  ('f',  'g')])

# Merge the graphs
G_merged = nx.compose_all([ G1,  G2,  G3])
'''

def inspect(G):
    nodes = list(G.nodes)
    edges = list(G.edges)
    
    Ginfo = {
        "Nodes": nodes,
        "Edges": edges
    }
    
    return Ginfo

def build(vertices,edges):
    """
    Create a graph using NetworkX from a list of vertices and edges.
    build([ u, v, w, ...],  [ (u,  v),  (w,  v),  ...])
    """
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    return G

def merge(*graphs):
    """
    Merge multiple NetworkX graphs into a single graph.
    """
    G = nx.Graph()
    
    for graph in graphs:
        G.add_nodes_from(graph.nodes())
        G.add_edges_from(graph.edges())
    
    return G

def path(cycle):

    C = list(cycle)
    G = nx.Graph()
    G.add_nodes_from(C)
    for i in range(len(C) - 1):
        G.add_edge(C[ i],C[ i + 1])
    return G

def cycle(cycle):

    C = list(cycle)
    G = nx.Graph()
    G.add_nodes_from(C)
    for i in range(len(C)):
        G.add_edge(C[ i],C[ (i + 1) % len(C)])
    return G

def star(hub,neighbors):

    leaves = list(neighbors)
    G = nx.Graph()
    G.add_node(hub)
    for node in leaves:
        G.add_node(node)
        G.add_edge(hub,node)
    return G

def trees(n):
    def is_isomorphic_to_any(graph,graph_list):
        for g in graph_list:
            if nx.is_isomorphic(graph,g):
                return True
        return False

    all_trees = [ ]
    nodes = list(range(n + 1))  # A tree with n edges has n+1 nodes

    for edges in combinations(combinations(nodes,  2),n):
        G = nx.Graph()
        G.add_edges_from(edges)
        if nx.is_tree(G) and not is_isomorphic_to_any(G,all_trees):
            all_trees.append(G)

    return all_trees
#notebook
t = 2
inc = 14*(t-1)
graph_path = 'C:/Users/baneg/OneDrive/Desktop/Git/Python/Research/Graph Theory/sigmas/sigmatex'
#? 8 (mod 14)

#! (61)

#*(61)-1

G = merge(path([ 0, 6, 1, 5,  2,  9,  7]),  path([ 3, 4]))

F_61_1 = [ G] #defines the decomposition 'object' a list of graph labelings
#visualize(14*t+7, F_61_1,  '(61)-1',  graph_path)


#*(61)-2

G  = merge(path([ 0, 6, 1, 5, 2, 9]), path([ 5, 3]),  path([ 7, 8]))

F_61_2 = [ G]
#visualize(14*t+7, F_61_2,  '(61)-2',  graph_path)

#*(61)-3

G  = merge(path([ 0, 6, 1, 5, 2, 9]), path([ 2, 4]),  path([ 7, 8]))

F_61_3 = [ G]
#visualize(14*t+7, F_61_3,  '(61)-3',  graph_path)

#*(61)-4

G = merge(star(1, [ 5, 6, 7, 4]), path([ 9, 2, 4]),  path([ 11, 10]))

F_61_4 = [ G]
#visualize(14*t+7, F_61_4,  '(61)-4',  graph_path)

#*(61)-5

G = merge(path([ 3, 8, 1, 4, 2]), path([ 5, 1, 7]),  path([ 9, 10]))

F_61_5 = [ G]
#visualize(14*t+7, F_61_5,  '(61)-5',  graph_path)

#*(61)-6

G = merge(star(8, [ 7, 4, 1]), star(6, [ 1, 0, 3]), path([ 9, 11]))

F_61_6 = [ G]
#visualize(14*t+7, F_61_6,  '(61)-6',  graph_path)

#*(61)-7

G = merge(star(1, [ 8, 5, 6, 7]), path([ 3, 6, 4]), path([ 9, 10]))

F_61_7 = [ G]
#visualize(14*t+7, F_61_7,  '(61)-7',  graph_path)

#*(61)-8

G = merge(star(1, [ 5, 6, 7, 8, 4]), path([ 3, 5, 1]), path([ 9, 10]))

F_61_8 = [ G]
#visualize(14*t+7, F_61_8,  '(61)-8',  graph_path)

#*(61)-9

G = merge(path([ 5, 11, 9, 12, 7]), path([ 9, 10, 6]), path([ 1, 8]))

F_61_9 = [ G]
#visualize(14*t+7, F_61_9,  '(61)-9',  graph_path)

#*(61)-10

G = merge(star(8, [ 1, 4, 5]),  path([ 3, 1, 6, 0]), path([ 9, 10]))

F_61_10 = [ G]
#visualize(14*t+7, F_61_10,  '(61)-10',  graph_path)

#*(61)-11

G = merge(star(1, [ 3 ,4, 5, 6, 7, 8]), path([ 10, 9]))

F_61_11 = [ G]
visualize(14*t+7, F_61_11,  '(61)-11',  graph_path)


#! (52)

#*(52)-1

G = merge(path([ 0, 6, 1, 5, 2, 9]), path([ 12, 10, 11]))

F_52_1 = [ G]
#visualize(14*t+7, F_52_1,  '(52)-1',  graph_path)

#*(52)-2

G = merge(path([ 3, 6, 1, 8, 4]), path([ 0, 6]), path([ 10, 9, 11]))

F_52_2 = [ G]
#visualize(14*t+7, F_52_2,  '(52)-2',  graph_path)

#*(52)-3

G = merge(path([ 5, 11, 9, 12, 7]), path([ 9, 10]), path([ 1, 8, 4]))

F_52_3 = [ G]
#visualize(14*t+7, F_52_3,  '(52)-3',  graph_path)

#*(52)-4

G = merge(path([ 3, 8, 1, 7]), path([ 4, 8, 1, 6]),  path([ 10, 9, 11]))

F_52_4 = [ G]
#visualize(14*t+7, F_52_4,  '(52)-4',  graph_path)

#*(52)-5

G = merge(star(1, [ 8, 5, 4, 7]), path([ 8, 3]), path([ 10, 9, 11]))

F_52_5 = [ G]
#visualize(14*t+7, F_52_5,  '(52)-5',  graph_path)

#*(52)-6

G = merge(star(1, [ 4, 5, 6, 7, 8]), path([ 10, 9, 11]))

F_52_6 = [ G]
#visualize(14*t+7, F_52_6,  '(52)-6',  graph_path)

#! (43)

#*(43)-1

G = merge(path([ 0, 6, 1, 5, 2]), path([ 3, 10, 8, 9]))

F_43_1 = [ G]
#visualize(14*t+7, F_43_1,  '(43)-1',  graph_path)

#*(43)-2

G = merge(path([ 5, 8, 1, 7]), path([ 1, 6]),  path([ 0, 4, 2, 3]))

F_43_2 = [ G]
#visualize(14*t+7, F_43_2,  '(43)-2',  graph_path)

#*(43)-3

G = merge(path([ 4, 8, 1, 7]), path([ 1, 6]), star(9, [ 10, 12, 11]))

F_43_3 = [ G]
#visualize(14*t+7, F_43_3,  '(43)-3',  graph_path)

#*(43)-4

G = merge(star(0, [ 6, 5, 4, 3]), path([ 2, 9, 7, 8]))

F_43_4 = [ G]
#visualize(14*t+7, F_43_4,  '(43)-4',  graph_path)

#*(43)-5

G = merge(star(9, [ 12, 11, 10]), path([ 4, 8, 1, 7, 2]))

F_43_5 = [ G]
#visualize(14*t+7, F_43_5,  '(43)-5',  graph_path)

#*(43)-6

G = merge(star(0, [ 6, 5, 4, 3]), star(9, [ 8, 7, 2]))

F_43_6 = [ G]
#visualize(14*t+7, F_43_6,  '(43)-6',  graph_path)

#! (511)

#*(511)-1

G = merge(path([ 0, 6, 1, 5, 2, 9]), path([ 8, 10]), path([ 3, 4]))

F_511_1 = [ G]
#visualize(14*t+7, F_511_1,  '(511)-1',  graph_path)


#*(511)-2

G = merge(path([ 4, 8, 1, 6, 3]), path([ 6, 0]), path([ 5, 7]), path([ 9, 10]))

F_511_2 = [ G]
#visualize(14*t+7, F_511_2,  '(511)-2',  graph_path)

#*(511)-3

G = merge(star(1, [ 4, 5, 8, 7]), path([ 8, 3]), path([ 0, 2]), path([ 9, 10]))

F_511_3 = [ G]
#visualize(14*t+7, F_511_3,  '(511)-3',  graph_path)

#*(511)-4

G = merge(path([ 5, 8, 1, 7]), path([ 4, 8, 1, 6]),  path([ 0, 2]), path([ 9, 10]))

F_511_4 = [ G]
#visualize(14*t+7, F_511_4,  '(511)-4',  graph_path)

#*(511)-5

G = merge(path([ 5, 11, 9, 12, 7]), path([ 9, 10]), path([ 1, 8]), path([ 0, 4]))

F_511_5 = [ G]
#visualize(14*t+7, F_511_5,  '(511)-5',  graph_path)

#*(511)-6

G = merge(star(1, [ 4, 5, 6, 7, 8]), path([ 2, 3]), path([ 9, 11]))

F_511_6 = [ G]
#visualize(14*t+7, F_511_6,  '(511)-6',  graph_path)

#! (421)

#*(421)-1

G = merge(path([ 0, 6, 1, 5, 2]), path([ 8, 10, 9]), path([ 4, 11]))

F_421_1 = [ G]
#visualize(14*t+7, F_421_1,  '(421)-1',  graph_path)

#*(421)-2

G = merge(path([ 5, 8, 1, 7]), path([ 1, 6]), path([ 10, 9, 11]), path([ 0, 4]))

F_421_2 = [ G]
#visualize(14*t+7, F_421_2,  '(421)-2',  graph_path)

#*(421)-3

G = merge(star(0, [ 6, 5, 4, 3]), path([ 1, 8, 7]), path([ 9, 11]))

F_421_3 = [ G]
#visualize(14*t+7, F_421_3,  '(421)-3',  graph_path)

#! (331)

#*(331)-1

G = merge(path([ 0, 6, 1, 5]), path([ 2, 9, 7, 10]), path([ 3, 4]))

F_331_1 = [ G]
#visualize(14*t+7, F_331_1,  '(331)-1',  graph_path)

#*(331)-2

G = merge(star(0, [ 4, 5, 6]), path([ 11, 9, 10, 7]), path([ 1, 8]))

F_331_2 = [ G]
#visualize(14*t+7, F_331_2,  '(331)-2',  graph_path)

#*(331)-3

G = merge(star(0, [ 4, 5, 6]), star(9, [ 10, 11, 12]), path([ 1, 8]))

F_331_3 = [ G]
#visualize(14*t+7, F_331_3,  '(331)-3',  graph_path)

#! (322)

#*(322)-1

G = merge(path([ 0, 6, 1, 5]), path([ 8, 10, 9]), path([ 11, 4, 7]))

F_322_1 = [ G]
#visualize(14*t+7, F_322_1,  '(322)-1',  graph_path)

#*(322)-2

G = merge(star(0, [ 6, 5, 4]), path([ 1, 8, 7]), path([ 11, 9, 12]))

F_322_2 = [ G]
#visualize(14*t+7, F_322_2,  '(322)-2',  graph_path)

#! (3211)

#*(3211)-1

G = merge(path([ 0, 6, 1, 5]), path([ 8, 10, 7]), path([ 4, 11]), path([ 2, 3]))

F_3211_1 = [ G]
#visualize(14*t+7, F_3211_1,  '(3211)-1',  graph_path)

#*(3211)-2

G = merge(star(0, [ 4, 5, 6]), path([ 11, 9, 12]), path([ 2, 3]), path([ 1, 8]))

F_3211_2 = [ G]
#visualize(14*t+7, F_3211_2,  '(3211)-2',  graph_path)

#! (4111)

#*(4111)-1

G = merge(path([ 0, 6, 1, 5, 2]), path([ 3, 10]), path([ 7, 9]), path([ 11, 12]))

F_4111_1 = [ G]
#visualize(14*t+7, F_4111_1,  '(4111)-1',  graph_path)

#*(4111)-2

G = merge(path([ 4, 8, 1, 7]), path([ 1, 6]), path([ 3, 5]), path([ 9, 12]), path([ 10, 11]))

F_4111_2 = [ G]
#visualize(14*t+7, F_4111_2,  '(4111)-2',  graph_path)

#*(4111)-3

G = merge(star(0, [ 6, 5, 4, 3]), path([ 1, 8]), path([ 10, 11]), path([ 7, 9]))

F_4111_3 = [ G]
#visualize(14*t+7, F_4111_3,  '(4111)-3',  graph_path)

#! (2221)

#*(2221)-1

G = merge(path([ 0, 6, 1]), path([ 4, 8, 5]), path([ 2, 9, 7]), path([ 10, 11]))

F_2221_1 = [ G]
#visualize(14*t+7, F_2221_1,  '(2221)-1',  graph_path)


#! (31111)

#*(31111)-1

G = merge(path([ 0, 6, 1, 5]), path([ 2, 9]), path([ 8, 10]), path([ 4, 7]), path([ 11, 12]))

F_31111_1 = [ G]
#visualize(14*t+7, F_31111_1,  '(31111)-1',  graph_path)

#*(31111)-2

G = merge(star(0, [ 6, 5, 4]), path([ 2, 3]), path([ 9, 11]), path([ 1, 8]), path([ 7, 10]))

F_31111_2 = [ G]
#visualize(14*t+7, F_31111_2,  '(31111)-2',  graph_path)


#! (22111)

#*(22111)-1

G = merge(path([ 0, 6, 1]), path([ 4, 8, 5]), path([ 3, 10]), path([ 7, 9]), path([ 11, 12]))

F_22111_1 = [ G]
#visualize(14*t+7, F_22111_1,  '(22111)-1',  graph_path)

#! (211111)

#*(211111)-1

G = merge(path([ 0, 6, 1]), path([ 4, 8]), path([ 2, 5]), path([ 3, 10]), path([ 7, 9]), path([ 11, 12]))

F_211111_1 = [ G]
#visualize(14*t+7, F_211111_1,  '(211111)-1',  graph_path)
#& above here is generalized