import networkx as nx
import sys

def gvpath(i):
    if i == 0:
        return 'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\Graph Theory'
    if i == 1:
        return 'C:\\Users\\Danny\\Desktop\\Git\\research'
    else: 
        return False
sys.path.append(gvpath(1)) # here is the path with GVIS
import graph_visualization
from graph_visualization import visualize
from itertools import combinations
import math

# how to merge two graphs:
'''
# Create two graphs
G1 = nx.Graph()
G1.add_edges_from([('a',  'b'),  ('b',  'c')])

G2 = nx.Graph()
G2.add_edges_from([('c',  'd'),  ('d',  'e')])

# Merge the graphs
G_merged = nx.compose(G1,  G2)
'''
#how to merge multiple graphs
'''
# Create multiple graphs
G1 = nx.Graph()
G1.add_edges_from([('a',  'b'),  ('b',  'c')])

G2 = nx.Graph()
G2.add_edges_from([('c',  'd'),  ('d',  'e')])

G3 = nx.Graph()
G3.add_edges_from([('e',  'f'),  ('f',  'g')])

# Merge the graphs
G_merged = nx.compose_all([G1,  G2,  G3])
'''

def inspect(G):
    nodes = list(G.nodes)
    edges = list(G.edges)
    
    Ginfo = {
        "Nodes": nodes, 
        "Edges": edges
    }
    
    return Ginfo

def build(vertices,  edges):
    """
    Create a graph using NetworkX from a list of vertices and edges.
    build([u, v, w, ...],  [ (u,  v),  (w,  v),  ...])
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
        G.add_edge(C[i],  C[i + 1])
    return G

def cycle(cycle):

    C = list(cycle)
    G = nx.Graph()
    G.add_nodes_from(C)
    for i in range(len(C)):
        G.add_edge(C[i],  C[(i + 1) % len(C)])
    return G

def star(hub,  neighbors):

    leaves = list(neighbors)
    G = nx.Graph()
    G.add_node(hub)
    for node in leaves:
        G.add_node(node)
        G.add_edge(hub,  node)
    return G

def trees(n):
    def is_isomorphic_to_any(graph,  graph_list):
        for g in graph_list:
            if nx.is_isomorphic(graph,  g):
                return True
        return False

    all_trees = [ ]
    nodes = list(range(n + 1))  # A tree with n edges has n+1 nodes

    for edges in combinations(combinations(nodes,  2),  n):
        G = nx.Graph()
        G.add_edges_from(edges)
        if nx.is_tree(G) and not is_isomorphic_to_any(G,  all_trees):
            all_trees.append(G)

    return all_trees

#notebook
t = 2
inc = 14*(t-1)

#? 7 (mod 14)

#! (61)

#*(61)-1
G1  = merge(path([0,1,2,4,6,9,12]),  path([13,14]))
G2  = merge(path([3,4,7,9,10,13,15]),  path([5,8]))
G3  = merge(path([8,11,12,10,7,5,6]),  path([1,3]))
G4 = merge(path([0,4,9,15,8,16,7]),  path([1,11]))

F_61_1 = [ G1, G2, G3, G4] #defines the decomposition 'object' a list of graph labelings
visualize(14*t+7, F_61_1,  '(61)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(61)-2
G1 = merge(path([1,2,4,6,9,12]), path([6,7]),  path([ 14,15]))
G2 = merge(path([4,7,9,10,13,15]),  path([10,11]),path([5,8]))
G3 = merge(path([5,7,10,12,11,8]), path([12,13]),  path([1,3]))
G4 = merge(path([0,4,9,15,8,16]), path([15,6]),  path([1, 11]))

F_61_2 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_61_2,  '(61)-2',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')


#*(61)-3
G1  = merge(path([9,6,4,2,1,0]), path([1,3]),  path([16, 19]))
G2  = merge(path([4,7,9,10,13,15]), path([13,14]),  path([17,18]))
G3  = merge(path([11,12,10,7,5,6]), path([5,8]),  path([15,18]))
G4  = merge(path([4,9,15,8,16,7]), path([16,12]),  path([1,11]))

F_61_3 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_61_3,  '(61)-3',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(61)-4
G1 = merge(star(6, [8,9,7,4]), path([4,2,1]),  path([14,15]))
G2 = merge(star(10, [8,13,11,9]), path([9,7,4]),  path([12,15]))
G3 = merge(star(12, [9,11,13,10]), path([10,7,5]),  path([1,4]))
G4 = merge(star(15, [7,8,6,9]), path([ 9, 4, 0]),  path([1, 11]))

F_61_4 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_61_4,  '(61)-4',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(61)-5
G1 = merge(path([2,4,6,9,12]), path([8,6,7]),  path([11,14]))
G2 = merge(path([0,2,3,6,5]), path([1,3,4]),  path([7,8]))
G3 = merge(path([0,3,5,4,1]), path([7,5,8]),  path([15,16]))
G4 = merge(path([4,9,15,8,12]), path([7,15,6]),  path([1,11]))

F_61_5 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_61_5,  '(61)-5',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(61)-6
G1 = merge(star(2, [1,5,4]), star(6, [4,8,9]), path([12,15]))
G2 = merge(star(7, [4,8,9]), star(10, [9,11,13]), path([1,3]))
G3 = merge(star(7, [5,6,10]), star(12, [10,13,11]), path([1,4]))
G4 = merge(star(4, [0,9,12]), star(15, [9,6,8]), path([1,11]))

F_61_6 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_61_6,  '(61)-6',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(61)-7
G1 = merge(star(6, [8,9,7,4]), path([2,4,5]), path([12,14]))
G2 = merge(star(3, [1,6,4,2]), path([5,2,0]), path([10,12]))
G3 = merge(star(8, [9,11,5,7]), path([4,7,10]), path([12,13]))
G4 = merge(star(15, [7,8,6,9]), path([4,9,13]), path([1,11]))

F_61_7 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_61_7,  '(61)-7',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(61)-8
G1 = merge(star(6, [7,8,9,5,4]), path([4,2]), path([12,14]))
G2 = merge(star(3, [2,0,6,5,4]), path([4,7]), path([9,12]))
G3 = merge(star(8, [7,9,11,10,5]), path([5,4]), path([0,2]))
G4 = merge(star(15, [6,7,8,11,9]), path([9,4]), path([2,12]))

F_61_8 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_61_8,  '(61)-8',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(61)-9
G1 = merge(path([2,4,6,9,12]), path([6,8,7]), path([13,14]))
G2 = merge(path([0,2,3,6,5]), path([3,4,7]), path([8,10]))
G3 = merge(path([0,3,5,4,1]), path([5,8,9]), path([12,14]))
G4 = merge(path([4,9,15,8,12]), path([15,7,16]), path([1,11]))

F_61_9 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_61_9,  '(61)-9',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(61)-10
G1 = merge(star(4, [2,1,6]), path([8,6,9,12]), path([14,15]))
G2 = merge(star(6, [5,7,3]), path([4,3,2,0]), path([8,9]))
G3 = merge(star(3, [0,1,5]), path([8,5,4,7]), path([12,14]))
G4 = merge(star(9, [4,18,15]),  path([7,15,8,12]), path([1,11]))

F_61_10 = [G1, G2, G3, G4]
#visualize(14*t+7, F_61_10,  '(61)-10',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')


#! (52)

#*(52)-1
G1 = merge(path([1,2,4,6,9,12]),  path([13,14,15]))
G2 = merge(path([3,4,7,9,10,13]),  path([5,8,6]))
G3 = merge(path([11,12,10,7,5,6]),  path([4,1,3]))
G4 = merge(path([0,4,9,15,8,16]), path([1,11,2]))

F_52_1 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_52_1,  '(52)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(52)-2
G1 = merge(path([1,2,4,6,9]), path([2,5]), path([13,14,15]))
G2 = merge(path([13,10,9,7,4]), path([10,11]), path([5,8,6]))
G3 = merge(path([11,12,10,7,5]), path([12,13]), path([4,1,3]))
G4 = merge(path([0,4,9,15,8]), path([4,12]), path([1,11,2]))

F_52_2 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_52_2,  '(52)-2',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(52)-3
G1 = merge(path([0,1,2,4,6]), path([2,5]), path([16,13,14]))
G2 = merge(path([8,6,3,2,0]), path([3,4]), path([14,12,15]))
G3 = merge(path([7,4,5,3,0]), path([5,6]), path([11,8,10]))
G4 = merge(path([7,0,4,9,15]), path([4,12]), path([1,11,2]))

F_52_3 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_52_3,  '(52)-3',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(52)-4
G1 = merge(path([1,2,4,6]), path([5,2,4,7]),  path([13,14,16]))
G2 = merge(path([8,6,3,2]), path([9,6,3,4]),  path([14,12,15]))
G3 = merge(path([4,5,3,0]), path([6,5,3,1]),  path([11,8,7]))
G4 = merge(path([7,0,4,9]), path([6,0,4,12]),  path([1,11,2]))

F_52_4 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_52_4,  '(52)-4',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(52)-5
G1 = merge(star(2, [0,1,5,4]), path([4,7]), path([12,11,13]))
G2 = merge(star(6, [7,8,9,3]), path([3,2]), path([14,12,15]))
G3 = merge(star(3, [4,0,1,5]), path([5,6]), path([11,8,7]))
G4 = merge(star(0, [8,7,6,4]), path([4,9]), path([1,11,2]))

F_52_5 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_52_5,  '(52)-5',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(52)-6
G1 = merge(star(2, [0,1,3,4,5]), path([12,11,14]))
G2 = merge(star(6, [4,5,7,8,9]), path([14,12,15]))
G3 = merge(star(3, [0,1,4,5,6]), path([11,8,7]))
G4 = merge(star(0, [4,5,6,7,8]), path([1,11,2]))

F_52_6 = [ G1, G2, G3, G4]
visualize(14*t+7, F_52_6,  '(52)-6',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#! (43)

#*(43)-1
G1 = merge(path([2,4,6,9,12]), path([13,14,15,16]))
G2 = merge(path([3,4,7,9,10]), path([11,12,15,13]))
G3 = merge(path([12,10,7,5,6]), path([20,17,15,18]))
G4 = merge(path([4,9,15,8,16]), path([5,1,11,2]))

F_43_1 = [ G1, G2, G3, G4]

#visualize(14*t+7, F_43_1,  '(43)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(43)-2
G1 = merge(path([12,9,6,4]), path([9,11]), path([14,15,16,17]))
G2 = merge(path([9,7,4,3]), path([7,6]), path([11,12,15,13]))
G3 = merge(path([6,5,7,10]), path([5,3]), path([18,15,17,20]))
G4 = merge(path([16,8,15,9]), path([8,12]),  path([2,11,1,6]))

F_43_2 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_43_2,  '(43)-2',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(43)-3
G1 = merge(path([4,6,9,11]), path([6,8]), star(15, [14,16,18]))
G2 = merge(path([9,7,4,3]), path([7,6]), star(17, [15,16,20]))
G3 = merge(path([6,5,7,10]), path([5,3]), star(12, [9,11,15]))
G4 = merge(path([16,8,15,9]), path([8,12]), star(1, [11,10,6]))

F_43_3 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_43_3,  '(43)-3',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(43)-4
G1 = merge(star(15, [13,14,16,18]), path([7,6,9,11]))
G2 = merge(star(17, [14,15,16,20]), path([9,7,4,3]))
G3 = merge(star(12, [9,10,11,15]), path([4,6,5,7]))
G4 = merge(star(1, [5,6,10,11]), path([16,8,15,9]))

F_43_4 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_43_4,  '(43)-4',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(43)-5
G1 = merge(star(15, [13,14,16]), path([7,6,9,11,8]))
G2 = merge(star(17, [15,16,20]), path([9,7,4,3,5]))
G3 = merge(star(12, [9,11,15]), path([4,6,5,7,10]))
G4 = merge(star(1, [6,10,11]), path([16,8,15,9,5]))

F_43_5 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_43_5,  '(43)-5',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(43)-6
G1 = merge(star(15, [13,14,16,18]), star(9, [6,11,12]))
G2 = merge(star(17, [18,15,16,20]), star(7, [4,9,10]))
G3 = merge(star(12, [10,11,14,15]), star(6, [4,5,7]))
G4 = merge(star(1, [5,6,11,10]), star(8, [14,15,16]))

F_43_6 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_43_6,  '(43)-6',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#! (511)

#*(511)-1
G1 = merge(path([1,2,4,6,9,12]),  path([13,14]),path([7,8]))
G2 = merge(path([3,4,7,9,10,13]),  path([6,8]),path([12,15]))
G3 = merge(path([11,12,10,7,5,6]),  path([4,1]),path([15,17]))
G4 = merge(path([0,4,9,15,8,16]), path([1,11]),path([3,12]))


F_511_1 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_511_1,  '(511)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(511)-2
G1 = merge(path([1,2,4,6,9]), path([2,5]), path([13,14]),path([7,8]))
G2 = merge(path([13,10,9,7,4]), path([10,11]), path([6,8]),path([12,15]))
G3 = merge(path([11,12,10,7,5]), path([12,13]), path([4,1]),path([15,17]))
G4 = merge(path([0,4,9,15,8]), path([4,12]), path([1,11]),path([5,14]))

F_511_2 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_511_2,  '(511)-2',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(511)-3
G1 = merge(path([0,1,2,4,7]), path([2,5]), path([6,9]),path([8,10]))
G2 = merge(path([8,6,3,2,0]), path([3,4]), path([5,7]),path([12,13]))
G3 = merge(path([6,4,5,3,0]), path([5,8]), path([13,14]),path([15,18]))
G4 = merge(path([7,0,4,9,15]), path([4,12]), path([1,11]),path([5,14]))

F_511_3 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_511_3,  '(511)-3',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(511)-4
G1 = merge(path([1,2,4,6]), path([5,2,4,7]),  path([13,14]),path([12,15]))
G2 = merge(path([8,6,3,2]), path([9,6,3,4]),  path([12,14]),path([15,18]))
G3 = merge(path([4,5,3,0]), path([6,5,3,1]),  path([7,8]),path([14,16]))
G4 = merge(path([7,0,4,9]), path([6,0,4,12]),  path([1,11]),path([5,14]))

F_511_4 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_511_4,  '(511)-4',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(511)-5
G1 = merge(star(2, [0,1,5,4]), path([4,7]), path([11,13]),path([12,15]))
G2 = merge(star(6, [7,8,9,3]), path([3,2]), path([11,12]),path([1,4]))
G3 = merge(star(3, [4,0,1,5]), path([5,6]), path([7,8]),path([12,14]))
G4 = merge(star(0, [8,7,6,4]), path([4,9]), path([1,11]),path([5,14]))

F_511_5 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_511_5,  '(511)-5',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(511)-6
G1 = merge(star(2, [0,1,3,4,5]), path([12,14]),path([18,19]))
G2 = merge(star(6, [4,5,7,8,9]), path([12,15]),path([11,14]))
G3 = merge(star(3, [0,1,4,5,6]), path([8,11]),path([14,15]))
G4 = merge(star(0, [4,5,6,7,8]), path([1,11]),path([3,12]))

F_511_6 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_511_6,  '(511)-6',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#! (421)

#*(421)-1
G1 = merge(path([2,4,6,9,12]), path([13,14,15]),path([18,19]))
G2 = merge(path([3,4,7,9,10]), path([12,15,13]),path([1,2]))
G3 = merge(path([12,10,7,5,6]), path([20,17,15]),path([1,4]))
G4 = merge(path([4,9,15,8,16]), path([5,1,11]),path([3,12]))

F_421_1 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_421_1,  '(421)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(421)-2
G1 = merge(path([12,9,6,4]), path([9,11]), path([15,16,17]),path([1,0]))
G2 = merge(path([9,7,4,3]), path([7,6]), path([12,15,13]),path([18,19]))
G3 = merge(path([6,5,7,10]), path([5,3]), path([15,17,20]),path([1,4]))
G4 = merge(path([16,8,15,9]), path([8,12]),  path([2,11,1]),path([0,5]))

F_421_2 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_421_2,  '(421)-2',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(421)-3
G1 = merge(star(15, [13,14,16,18]), path([7,6,9]),path([2,4]))
G2 = merge(star(17, [14,15,16,20]), path([7,4,3]),path([11,13]))
G3 = merge(star(12, [9,10,11,15]), path([6,5,7]),path([0,2]))
G4 = merge(star(1, [5,6,10,11]), path([8,15,9]),path([4,12]))

F_421_3 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_421_3,  '(421)-3',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#! (331)

#*(331)-1
G1 = merge(path([4,6,9,12]), path([13,14,15,16]),path([19,20]))
G2 = merge(path([3,4,7,9]), path([11,12,15,13]),path([16,17]))
G3 = merge(path([12,10,7,5]), path([20,17,15,18]),path([9,11]))
G4 = merge(path([9,15,8,16]), path([5,1,11,2]),path([7,12]))

F_331_1 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_331_1,  '(331)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(331)-2
G1 = merge(star(15, [13,14,16]), path([7,6,9,11]),path([1,4]))
G2 = merge(star(17, [15,16,20]), path([7,4,3,5]),path([0,2]))
G3 = merge(star(12, [9,11,15]), path([4,6,5,7]),path([0,3]))
G4 = merge(star(1, [6,10,11]), path([16,8,15,9]),path([0,4]))

F_331_2 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_331_2,  '(331)-2',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(331)-3
G1 = merge(star(15, [13,14,18]), star(9, [6,11,12]),path([1,2]))
G2 = merge(star(17, [18,15,20]), star(7, [4,9,10]),path([2,3]))
G3 = merge(star(12, [11,14,15]), star(6, [4,5,7]),path([17,19]))
G4 = merge(star(1, [5,6,11]), star(8, [14,15,16]),path([0,9]))

F_331_3 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_331_3,  '(331)-3',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#! (322)

#*(322)-1
G1 = merge(path([16,15,14,13]), path([0,3,5]), path([6,9,12]))
G2 = merge(path([11,12,15,13]), path([7,9,10]), path([16,18,20]))
G3 = merge(path([18,15,17,20]), path([10,11,14]), path([6,5,7]))
G4 = merge(path([2,12,3,11]), path([7,1,8]), path([4,0,5]))

F_322_1 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_322_1,  '(322)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(322)-2
G1 = merge(star(15, [13,18]), star(9, [6,11,12]),path([0,1,2]))
G2 = merge(star(17, [18,20]), star(7, [4,9,10]),path([2,3,1]))
G3 = merge(star(12, [11,14,15]), star(6, [4,7]),path([17,19,20]))
G4 = merge(star(1, [6,11]), star(8, [14,15,16]),path([4,0,9]))

F_322_2 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_322_2,  '(322)-2',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#! (3211)

#*(3211)-1
G1 = merge(path([8,6,9,11]),path([0,1,2]),path([16,19]),path([15,18]))
G2 = merge(path([9,7,10,8]),path([18,17,20]),path([11,14]),path([2,3]))
G3 = merge(path([13,11,12,14]),path([17,19,20]),path([6,7]),path([5,8]))
G4 = merge(path([0,5,1,7]),path([3,10,2]),path([4,13]),path([6,16]))

F_3211_1 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_3211_1,  '(3211)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(3211)-2
G1 = merge(star(9, [6,11,12]), path([0,1,2]), path([15,18]), path([13,14]))
G2 = merge(star(7, [4,9,10]), path([18,17,20]), path([11,13]), path([2,3]))
G3 = merge(star(12, [11,14,15]), path([17,19,20]), path([6,8]), path([1,3]))
G4 = merge(star(0,[4,5,6]), path([9,1,8]), path([3,12]),path([7,17]))

F_3211_2 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_3211_2,  '(3211)-2',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#! (4111)

#*(4111)-1
G1 = merge(path([2,4,6,9,12]), path([13,14]),path([18,19]),path([0,1]))
G2 = merge(path([3,4,7,9,10]), path([15,13]),path([1,2]),path([8,5]))
G3 = merge(path([12,10,7,5,6]), path([20,17]),path([8,11]),path([1,3]))
G4 = merge(path([4,9,15,8,16]), path([1,11]),path([3,12]),path([2,6]))

F_4111_1 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_4111_1,  '(4111)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(4111)-2
G1 = merge(path([12,9,6,4]), path([9,11]), path([15,16]),path([8,10]),path([2,3]))
G2 = merge(path([9,7,4,3]), path([7,6]), path([15,13]),path([18,19]),path([5,8]))
G3 = merge(path([6,5,7,10]), path([5,3]), path([20,17]),path([8,11]),path([0,1]))
G4 = merge(path([16,8,15,9]), path([8,12]),  path([2,11]),path([0,5]),path([3,13]))
F_4111_2 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_4111_2,  '(4111)-2',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(4111)-3
G1 = merge(star(15, [13,14,16,18]), path([6,9]),path([2,4]),path([5,7]))
G2 = merge(star(17, [14,15,16,20]), path([7,4]),path([11,13]),path([5,6]))
G3 = merge(star(12, [9,10,11,15]), path([6,7]),path([0,2]),path([3,4]))
G4 = merge(star(1, [5,6,10,11]), path([15,9]),path([4,12]),path([0,7]))

F_4111_3 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_4111_3,  '(4111)-3',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')


#! (2221)

#*(2221)-1
G1 = merge(star(15, [13,18]), star(9, [6,11]),path([0,1,2]),path([16,19]))
G2 = merge(star(17, [18,20]), star(7, [9,10]),path([2,3,1]),path([11,14]))
G3 = merge(star(12, [11,14]), star(6, [4,7]),path([17,19,20]),path([5,8]))
G4 = merge(star(1, [6,11]), star(8, [14,16]),path([4,0,9]),path([3,10]))

F_2221_1 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_2221_1,  '(2221)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#! (31111)

#*(31111)-1
G1 = merge(path([2,4,6,9]), path([13,14]),path([18,19]),path([0,1]),path([10,12]))
G2 = merge(path([3,4,7,9]), path([15,13]),path([1,2]),path([8,5]),path([16,17]))
G3 = merge(path([10,7,5,6]), path([20,17]),path([8,11]),path([1,3]),path([9,12]))
G4 = merge(path([9,15,8,16]), path([1,11]),path([3,12]),path([2,6]),path([0,5]))
F_31111_1 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_31111_1,  '(31111)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#*(31111)-2
G1 = merge(star(15, [13,16,18]), path([6,9]),path([2,4]),path([5,7]),path([0,1]))
G2 = merge(star(17, [14,16,20]), path([7,4]),path([11,13]),path([5,6]),path([1,3]))
G3 = merge(star(12, [9,10,11]), path([6,7]),path([0,2]),path([3,4]),path([5,8]))
G4 = merge(star(1, [5,10,11]), path([15,9]),path([4,12]),path([0,7]),path([3,8]))

F_31111_2 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_31111_2,  '(31111)-2',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#! (22111)

#*(22111)-1
G1 = merge(star(15, [18]), star(9, [6,11]),path([0,1,2]),path([16,19]),path([17,20]))
G2 = merge(star(17, [18]), star(7, [9,10]),path([2,3,1]),path([11,14]),path([5,8]))
G3 = merge(star(12, [11,14]), star(6, [4,7]),path([19,20]),path([13,15]),path([5,3]))
G4 = merge(star(1, [6,11]), star(8, [14,16]),path([0,9]),path([3,10]),path([13,17]))

F_22111_1 = [ G1, G2, G3, G4]
#visualize(14*t+7, F_22111_1,  '(22111)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#! (211111)

#*(211111)-1
G1 = merge(star(15, [18]), star(9, [11]),path([0,1,2]),path([16,19]),path([6,5]),path([7,10]))
G2 = merge(star(17, [18]), star(7, [9]),path([2,3,1]),path([11,14]),path([5,8]),path([16,13]))
G3 = merge(star(12, [14]), star(6, [4,7]),path([5,3]),path([13,15]),path([17,20]),path([18,19]))
G4 = merge(star(1, [11]), star(8, [14,16]),path([0,9]),path([3,10]),path([13,17]),path([2,7]))

F_211111_1 = [ G1, G2, G3, G4]
visualize(14*t+7, F_211111_1,  '(211111)-1',  'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research\\7 (mod 14)\\texgraph')

#^ done up to here