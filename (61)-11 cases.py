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
G1.add_edges_from([('a', 'b'), ('b', 'c')])

G2 = nx.Graph()
G2.add_edges_from([('c', 'd'), ('d', 'e')])

# Merge the graphs
G_merged = nx.compose(G1, G2)
'''
#how to merge multiple graphs
'''
# Create multiple graphs
G1 = nx.Graph()
G1.add_edges_from([('a', 'b'), ('b', 'c')])

G2 = nx.Graph()
G2.add_edges_from([('c', 'd'), ('d', 'e')])

G3 = nx.Graph()
G3.add_edges_from([('e', 'f'), ('f', 'g')])

# Merge the graphs
G_merged = nx.compose_all([G1, G2, G3])
'''

def inspect(G):
    nodes = list(G.nodes)
    edges = list(G.edges)
    
    Ginfo = {
        "Nodes": nodes,
        "Edges": edges
    }
    
    return Ginfo

def build(vertices, edges):
    """
    Create a graph using NetworkX from a list of vertices and edges.
    build([u,v,w,...], [(u, v), (w, v), ...])
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
        G.add_edge(C[i], C[i + 1])
    return G

def cycle(cycle):

    C = list(cycle)
    G = nx.Graph()
    G.add_nodes_from(C)
    for i in range(len(C)):
        G.add_edge(C[i], C[(i + 1) % len(C)])
    return G

def star(hub, neighbors):

    leaves = list(neighbors)
    G = nx.Graph()
    G.add_node(hub)
    for node in leaves:
        G.add_node(node)
        G.add_edge(hub, node)
    return G

def K(n):
    G = nx.Graph()
    G.add_nodes_from(range(0,n))
    for node in range(0,n):
        for neighbor in range(0,n):
            if node != neighbor:
                G.add_edge(node,neighbor)
    return G

def trees(n):
    def is_isomorphic_to_any(graph, graph_list):
        for g in graph_list:
            if nx.is_isomorphic(graph, g):
                return True
        return False

    all_trees = []
    nodes = list(range(n + 1))  # A tree with n edges has n+1 nodes

    for edges in combinations(combinations(nodes, 2), n):
        G = nx.Graph()
        G.add_edges_from(edges)
        if nx.is_tree(G) and not is_isomorphic_to_any(G, all_trees):
            all_trees.append(G)

    return all_trees
#notebook

K21 = nx.complete_graph(21,create_using=None)
K22 = nx.complete_graph(22,create_using=None)
#visualize(21,[K21], 'control', 'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research')

#arbirarily taking vertices from Pauline Cain's construction to be Z_21 
x1,x2,x3,x4,x5,x6,x7 = 0,1,2,3,4,5,6
w1,w2,w3,w4,w5,w6,w7 = 7,8,9,10,11,12,13
v1,v2,v3,v4,v5,v6,v7 = 14,15,16,17,18,19,20

#! (61)-11 for K_21
V1 = merge(star(v1,[v2,v3,v4,v5,v6,v7]),path([x3,x1]))
V2 = merge(star(v2,[w7,v3,v4,v5,v6,v7]),path([x7,x1]))
V3 = merge(star(v3,[w2,w6,v4,v5,v6,v7]),path([w3,x4]))
V4 = merge(star(v4,[w2,w3,w5,v5,v6,v7]),path([v3,x1]))
V5 = merge(star(v5,[w2,w3,w5,w7,v6,v7]),path([x2,x1]))
V6 = merge(star(v6,[w2,w4,w5,w6,w7,v7]),path([v2,x1]))
X2 = merge(star(x2,[w2,w3,w4,w5,w6,w7]),path([v5,w1]))
X3 = merge(star(x3,[x2,w3,w4,w5,w6,w7]),path([v1,w1]))
X4 = merge(star(x4,[x1,x3,x7,w5,w6,w7]),path([w1,w2]))
X5 = merge(star(x5,[x1,x3,x4,w5,w6,w7]),path([w2,w3]))
X6 = merge(star(x6,[x1,x3,x4,x5,w6,w7]),path([w3,w4]))
X7 = merge(star(x7,[x2,x3,x5,x6,w6,w7]),path([v2,w1]))
W1 = merge(star(w1,[x2,x3,x4,x5,x6,x7]),path([v1,x1]))
W2 = merge(star(w2,[x4,x5,x6,x7,v1,v7]),path([w6,v2]))
W3 = merge(star(w3,[x5,x6,x7,v1,v2,v7]),path([v3,w1]))
W4 = merge(star(w4,[v2,v3,v7,x5,x6,x7]),path([v5,x1]))
W5 = merge(star(w5,[v2,v3,v7,x1,x6,x7]),path([v4,x2]))
W6 = merge(star(w6,[v1,v4,v5,v7,x1,w5]),path([w2,x3]))
W7 = merge(star(w7,[v3,v4,v7,x1,w5,w6]),path([v6,x2]))
U1 = merge(star(v1,[x2,x3,x4,x5,x6,x7]),path([w1,v7]))
U2 = merge(star(v2,[x2,x3,x4,x5,x6,x7]),path([v6,w1]))
U3 = merge(star(v3,[x2,x3,x4,x5,x6,x7]),path([v4,w1]))
U4 = merge(star(v4,[x1,x3,x4,x5,x6,x7]),path([w5,v1]))
U5 = merge(star(v5,[x2,x3,x4,x5,x6,x7]),path([w4,v1]))
U6 = merge(star(v6,[x1,x3,x4,x5,x6,x7]),path([w7,v1]))
U7 = merge(star(v7,[x1,x3,x4,x5,x6,x7]),path([w4,w5]))
Y1 = merge(star(w1,[w3,w4,w5,w6,w7,x1]),path([x4,x2]))
Y2 = merge(star(w2,[w4,w5,w6,w7,x1,v2]),path([x5,x2]))
Y3 = merge(star(w3,[w5,w6,w7,x1,v3,v6]),path([x6,x2]))
Y4 = merge(star(w4,[w6,w7,x1,x4,v4,v5]),path([v7,x2]))

K21_611_11 = [V1,V2,V3,V4,V5,V6,X2,X3,X4,X5,X6,X7,W1,W2,W3,W4,W5,W6,W7,U1,U2,U3,U4,U5,U6,U7,Y1,Y2,Y3,Y4]
F_61_11_7mod14 = merge(V1,V2,V3,V4,V5,V6,X2,X3,X4,X5,X6,X7,W1,W2,W3,W4,W5,W6,W7,U1,U2,U3,U4,U5,U6,U7,Y1,Y2,Y3,Y4)

#visualize(21,[F_61_11_7mod14], 'K21', 'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research')
#! (61)-11 for K_22
#modified forests:

nX2 = merge(star(x2,[w2,w3,w4,w5,w6,w7]),path([math.inf,x7]))
nY1 = merge(star(w1,[w3,w4,w5,w6,w7,x1]),path([math.inf,v7]))
nW1 = merge(star(w1,[x2,x3,x4,x5,x6,x7]),path([math.inf,w7]))

infx = merge(star(math.inf,[x1,x2,x3,x4,x5,x6]),path([v5,w1]))
infv = merge(star(math.inf,[v1,v2,v3,v4,v5,v6]),path([x4,x2]))
infw = merge(star(math.inf,[w1,w2,w3,w4,w5,w6]),path([v1,x1]))

K22_611_11 = [V1,V2,V3,V4,V5,V6,nX2,X3,X4,X5,X6,X7,nW1,W2,W3,W4,W5,W6,W7,U1,U2,U3,U4,U5,U6,U7,nY1,Y2,Y3,Y4,infx,infv,infw]
F_61_11_8mod14 = merge(V1,V2,V3,V4,V5,V6,nX2,X3,X4,X5,X6,X7,nW1,W2,W3,W4,W5,W6,W7,U1,U2,U3,U4,U5,U6,U7,nY1,Y2,Y3,Y4,infx,infv,infw)
#visualize(21,[F_61_11_8mod14], 'K22', 'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research')

#& check cumulative list of stars in both decompositions

stars = [V1,V2,V3,V4,V5,V6,nX2,X2,X3,X4,X5,X6,X7,nW1,W1,W2,W3,W4,W5,W6,W7,U1,U2,U3,U4,U5,U6,U7,nY1,Y1,Y2,Y3,Y4,infx,infv,infw]

'''for star in stars:
    visualize(21,[star], 'whatever', 'C:\\Users\\baneg\\OneDrive\\Desktop\\Git\\Python\\Research')'''

#&last check for isomorphism

#print(nx.is_isomorphic(K21, F_61_11_7mod14 , node_match=None, edge_match=None))
#print(nx.is_isomorphic(K22, F_61_11_8mod14 , node_match=None, edge_match=None))

